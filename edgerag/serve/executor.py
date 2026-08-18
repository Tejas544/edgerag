"""The real :class:`~edgerag.serve.engine.BatchExecutor`: our decoder over the paged pool.

This is the one place in the serving stack that touches tensors, and by construction it runs only
on the engine's worker thread (``BUGS.md`` P-17). Everything above it -- HTTP, SSE, retrieval --
is arithmetic-free.

Three things it owns that the scheduler deliberately does not, because the scheduler is pure
decision-making and has no tensors in it:

* **One cache per request, one pool for all of them.** ``PagedKVCache`` takes its pools as
  arguments precisely so several sequences can page into a single physical arena -- that arena is
  allocated once, here, and every admitted request gets a block table into it. Allocating a pool
  per request is ``BUGS.md`` B-05, which cost a whole T4 session.
* **Chunked prefill.** A request contributes ``chunk_size`` tokens per iteration rather than its
  whole prompt, so a 7,000-token RAG prefill cannot monopolise the GPU while other requests decode
  (``BUGS.md`` P-18, ``CONTEXT.md`` D18).
* **The prompt's embeddings.** Requests carrying ``prompt_embeds`` (a retrieved page's visual
  tokens, already merged) are sliced from that tensor; text-only requests are embedded from ids.
  Both take the same path from there.

**Known limitation, stated rather than discovered:** block growth calls ``ensure_capacity``
directly rather than routing through ``Scheduler.try_allocate``, so a request that exhausts the
pool mid-generation raises ``OutOfBlocksError`` and fails *that request* instead of preempting a
younger one (D16). Admission already budgets each request's full prompt-plus-generation before
letting it in, so this only bites under copy-on-write pressure -- the case D16's pending note
flags. Wiring preemption in means giving the executor the scheduler, which is a coupling worth
paying for only once there is a measurement showing it matters.

Batching note, stated rather than glossed: prefill requests are run **one at a time** within an
iteration. Each has its own cache and its own sequence length, so they cannot share a forward pass
without padding to the longest -- and padding is precisely the waste D14 finding 4 measured as
superlinear TTFT degradation. Decode is the same: one token per request, each against its own
paged cache. The batching that matters here is *iteration-level* -- new requests join every step
without waiting for the batch to drain -- which is what continuous batching means and what the
scheduler already implements.
"""

from __future__ import annotations

from typing import Any

import torch

from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.paged import PagedKVCache
from edgerag.core.spec import ModelSpec
from edgerag.sched.request import Request
from edgerag.sched.scheduler import Batch
from edgerag.serve.engine import StepOutput

#: Prompt tokens processed per request per iteration. 512 is D18's measured default: far enough
#: from the noisy end that numerical drift stays ~1e-4 on fp32 logits and invisible to greedy
#: decoding, small enough that a 7k prefill yields to decoding requests ~14 times on the way.
DEFAULT_CHUNK_SIZE = 512


class ModelExecutor:
    """Runs scheduled batches on a real decoder. Constructed once; lives on the worker thread."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        spec: ModelSpec,
        allocator: BlockAllocator,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.decoder = decoder
        self.spec = spec
        self.allocator = allocator
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size

        # The arena. Allocated once, shared by every request's block table (B-05).
        shape = (spec.n_kv_heads, allocator.num_blocks, allocator.block_size, spec.head_dim)
        self.key_pool = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)
        ]
        self.value_pool = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(spec.n_layers)
        ]

    @property
    def pool_bytes(self) -> int:
        """What the arena costs, for the memory ledger. Allocated whether or not it is used."""
        per_tensor = self.key_pool[0].numel() * self.key_pool[0].element_size()
        return per_tensor * len(self.key_pool) * 2

    def _cache_for(self, request: Request) -> PagedKVCache:
        if request.cache is None:
            request.cache = PagedKVCache(
                self.spec,
                self.allocator,
                self.device,
                self.dtype,
                pool=self.key_pool,
                value_pool=self.value_pool,
            )
        return request.cache

    def _chunk_embeds(self, request: Request, start: int, end: int) -> torch.Tensor:
        """The prompt slice ``[start:end)``, as embeddings the decoder can consume."""
        if request.prompt_embeds is not None:
            return request.prompt_embeds[:, start:end].to(self.device)
        ids = torch.tensor(
            [request.prompt_token_ids[start:end]], dtype=torch.long, device=self.device
        )
        return self.decoder.embed_tokens(ids)

    @torch.inference_mode()
    def execute(self, batch: Batch) -> StepOutput:
        output = StepOutput()

        for request in batch.prefill:
            cache = self._cache_for(request)
            start = request.prefill_offset
            end = min(start + self.chunk_size, request.prompt_len)
            n_tokens = end - start
            if n_tokens <= 0:
                continue

            cache.table.ensure_capacity(cache.seq_len + n_tokens)
            embeds = self._chunk_embeds(request, start, end)
            logits = self.decoder(
                inputs_embeds=embeds, cache=cache, last_token_only=True
            )
            output.prefilled[request.request_id] = n_tokens

            # The final chunk produces the first generated token. Emitting it here rather than
            # waiting for the next iteration is not an optimisation -- the request transitions to
            # DECODING at the end of this step, and a decode step re-runs the *last* position,
            # which would generate the same token twice.
            if end >= request.prompt_len:
                output.tokens[request.request_id] = int(logits[0, -1].argmax())

        for request in batch.decode:
            cache = self._cache_for(request)
            if not request.generated_token_ids:
                # Unreachable if prefill behaved: its final chunk emits the first token above.
                # Raising beats the alternative, which is to re-feed `prompt_token_ids[-1]` -- a
                # token already in the cache -- and emit a duplicate of it as the answer's first
                # word. That is silently wrong output, which this project treats as worse than a
                # failed request (B-05).
                raise RuntimeError(
                    f"{request.request_id} reached decode with no generated token: prefill "
                    "completed without emitting one"
                )
            last_token = request.generated_token_ids[-1]
            cache.table.ensure_capacity(cache.seq_len + 1)
            ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
            logits = self.decoder(input_ids=ids, cache=cache, last_token_only=True)
            output.tokens[request.request_id] = int(logits[0, -1].argmax())

        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "pool_blocks": self.allocator.num_blocks,
            "block_size": self.allocator.block_size,
            "pool_bytes": self.pool_bytes,
            "dtype": str(self.dtype).replace("torch.", ""),
        }
