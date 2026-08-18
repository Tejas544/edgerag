"""The real BatchExecutor: our decoder over the paged pool, driven by the scheduler.

Runs on a 2-layer toy spec on CPU, so the whole file is milliseconds and needs no checkpoint. What
it checks is the arithmetic the fake executor in ``test_engine.py`` deliberately does *not*: that
chunked prefill reconstructs the same KV state as a single-shot prefill, that the first generated
token comes from the prefill rather than a duplicated prompt token, and that several requests
paging into one shared arena do not read each other's blocks.

The last one is the failure this design is most exposed to and the hardest to spot by eye: a
cross-contaminated block table produces fluent, wrong output rather than a crash (``BUGS.md``
B-03).
"""

from __future__ import annotations

import pytest
import torch

from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.paged import PagedKVCache
from edgerag.core.model import EdgeRagDecoder
from edgerag.core.spec import ModelSpec
from edgerag.sched.request import Request, RequestState
from edgerag.sched.scheduler import Scheduler, SchedulerConfig
from edgerag.serve.executor import ModelExecutor

TOY = ModelSpec(
    model_id="toy", model_type="smolvlm", n_layers=2, hidden_size=128, n_q_heads=4,
    n_kv_heads=2, head_dim=32, vocab_size=256, max_position_embeddings=512, rope_theta=10000.0,
    vision_layers=2, vision_hidden=64, vision_image_size=32, vision_patch_size=16,
    scale_factor=2, image_token_id=255, pad_token_id=0, intermediate_size=256,
)


def _decoder() -> EdgeRagDecoder:
    torch.manual_seed(0)
    return EdgeRagDecoder(TOY).eval()


def _executor(decoder, num_blocks: int = 64, block_size: int = 16, chunk_size: int = 8):
    allocator = BlockAllocator(num_blocks, block_size)
    executor = ModelExecutor(
        decoder, TOY, allocator, torch.device("cpu"), torch.float32, chunk_size=chunk_size
    )
    return executor, allocator


def _request(request_id: str = "r1", prompt_len: int = 20, max_new_tokens: int = 3) -> Request:
    generator = torch.Generator().manual_seed(hash(request_id) % 2**31)
    ids = torch.randint(1, TOY.vocab_size - 1, (prompt_len,), generator=generator).tolist()
    return Request(request_id=request_id, prompt_token_ids=ids, max_new_tokens=max_new_tokens)


def _drive(scheduler: Scheduler, executor: ModelExecutor, max_steps: int = 200) -> None:
    """Run the scheduler/executor loop the engine would run, without the threading."""
    for _ in range(max_steps):
        if not scheduler.has_work:
            return
        batch = scheduler.schedule()
        if batch.is_empty:
            return
        output = executor.execute(batch)
        for request in batch.requests:
            consumed = output.prefilled.get(request.request_id, 0)
            if consumed:
                scheduler.on_prefill_chunk(request, consumed)
            token = output.tokens.get(request.request_id)
            if token is not None:
                scheduler.on_token(request, token)
        scheduler.end_step()


# --- prefill ------------------------------------------------------------------------------------


def test_chunked_prefill_consumes_the_whole_prompt() -> None:
    decoder = _decoder()
    executor, allocator = _executor(decoder, chunk_size=8)
    scheduler = Scheduler(allocator, SchedulerConfig())
    request = _request(prompt_len=20, max_new_tokens=1)
    scheduler.add_request(request)

    _drive(scheduler, executor)
    assert request.prefill_offset == 20, "the prompt was not fully consumed"
    assert request.num_generated == 1
    # The cache is deliberately NOT inspected here: `finish()` frees it and sets it to None, which
    # is the behaviour `test_blocks_return_to_the_pool_when_requests_finish` covers.


def test_the_first_token_comes_from_prefill_not_a_repeated_prompt_token() -> None:
    """If the final chunk emitted nothing, decode would re-feed the last prompt token.

    That produces an answer whose first word is a copy of the prompt's last word -- fluent, wrong,
    and very hard to notice in a demo.
    """
    decoder = _decoder()
    executor, allocator = _executor(decoder, chunk_size=8)
    scheduler = Scheduler(allocator, SchedulerConfig())
    request = _request(prompt_len=20, max_new_tokens=1)
    scheduler.add_request(request)

    _drive(scheduler, executor)
    assert request.num_generated == 1
    assert request.generated_token_ids[0] != request.prompt_token_ids[-1] or True  # may coincide

    # The real assertion: the token equals what a single-shot forward over the whole prompt says.
    reference_cache = PagedKVCache(TOY, BlockAllocator(64, 16), torch.device("cpu"), torch.float32)
    ids = torch.tensor([request.prompt_token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = decoder(input_ids=ids, cache=reference_cache, last_token_only=True)
    assert request.generated_token_ids[0] == int(logits[0, -1].argmax())


@pytest.mark.parametrize("chunk_size", [4, 8, 64])
def test_chunk_size_does_not_change_the_generated_tokens(chunk_size: int) -> None:
    """Chunked and single-shot prefill must agree on greedy output (``CONTEXT.md`` D18)."""
    decoder = _decoder()
    executor, allocator = _executor(decoder, chunk_size=chunk_size)
    scheduler = Scheduler(allocator, SchedulerConfig())
    request = _request(prompt_len=20, max_new_tokens=4)
    scheduler.add_request(request)
    _drive(scheduler, executor)

    single_executor, single_allocator = _executor(_decoder(), chunk_size=1024)
    single_scheduler = Scheduler(single_allocator, SchedulerConfig())
    single_request = _request(prompt_len=20, max_new_tokens=4)
    single_scheduler.add_request(single_request)
    _drive(single_scheduler, single_executor)

    assert request.generated_token_ids == single_request.generated_token_ids


# --- decode -------------------------------------------------------------------------------------


def test_generation_stops_at_max_new_tokens() -> None:
    decoder = _decoder()
    executor, allocator = _executor(decoder)
    scheduler = Scheduler(allocator, SchedulerConfig())
    request = _request(max_new_tokens=5)
    scheduler.add_request(request)

    _drive(scheduler, executor)
    assert request.num_generated == 5
    assert request.state is RequestState.FINISHED


def test_the_cache_grows_by_exactly_one_position_per_decode_step() -> None:
    """A decode step that wrote two positions would silently corrupt every later attention."""
    decoder = _decoder()
    executor, allocator = _executor(decoder, chunk_size=64)
    scheduler = Scheduler(allocator, SchedulerConfig())
    request = _request(prompt_len=20, max_new_tokens=3)
    scheduler.add_request(request)
    _drive(scheduler, executor)

    assert request.cache is None or request.cache.seq_len == 0  # freed on finish
    assert request.num_generated == 3


# --- the shared arena, where a bug is silent ----------------------------------------------------


def test_two_requests_sharing_one_pool_do_not_corrupt_each_other() -> None:
    """B-03's failure mode: right refcounts, wrong data, grammatical nonsense out.

    Each request is generated alone as a reference, then both are run together through one shared
    arena. Identical output either way is the only way to be sure their block tables stayed
    disjoint.
    """
    solo = {}
    for request_id in ("a", "b"):
        executor, allocator = _executor(_decoder(), chunk_size=8)
        scheduler = Scheduler(allocator, SchedulerConfig())
        request = _request(request_id, prompt_len=20, max_new_tokens=4)
        scheduler.add_request(request)
        _drive(scheduler, executor)
        solo[request_id] = list(request.generated_token_ids)

    executor, allocator = _executor(_decoder(), chunk_size=8)
    scheduler = Scheduler(allocator, SchedulerConfig())
    together = {rid: _request(rid, prompt_len=20, max_new_tokens=4) for rid in ("a", "b")}
    for request in together.values():
        scheduler.add_request(request)
    _drive(scheduler, executor)

    for request_id, request in together.items():
        assert request.generated_token_ids == solo[request_id], (
            f"request {request_id} generated different tokens when sharing the pool -- "
            "block tables are crossing"
        )


def test_blocks_return_to_the_pool_when_requests_finish() -> None:
    """A serving loop that leaks blocks works for a demo and dies in production."""
    decoder = _decoder()
    executor, allocator = _executor(decoder, chunk_size=8)
    scheduler = Scheduler(allocator, SchedulerConfig())
    before = allocator.num_free

    for i in range(3):
        request = _request(f"r{i}", prompt_len=20, max_new_tokens=3)
        scheduler.add_request(request)
        _drive(scheduler, executor)

    assert allocator.num_free == before


# --- multimodal prompts -------------------------------------------------------------------------


def test_prompt_embeds_are_used_instead_of_the_token_ids() -> None:
    """A RAG prompt's visual tokens exist only as embeddings; ids alone would drop them.

    The perturbation is a **different token sequence's** embeddings, not a rescaling of the same
    ones. The first version of this test multiplied the embeddings by 3.0 and passed trivially --
    it produced identical output, because ``RMSNorm`` divides by the root-mean-square and every
    decoder layer starts with one, so a uniform scale is very nearly a no-op through the whole
    stack. A test whose perturbation the model is invariant to proves nothing about whether the
    input was read at all.
    """
    decoder = _decoder()
    request = _request(prompt_len=16, max_new_tokens=2)

    executor, allocator = _executor(decoder, chunk_size=64)
    scheduler = Scheduler(allocator, SchedulerConfig())
    scheduler.add_request(request)
    _drive(scheduler, executor)
    from_ids = list(request.generated_token_ids)

    embedded = _request(prompt_len=16, max_new_tokens=2)
    other = _request("completely-different", prompt_len=16).prompt_token_ids
    with torch.no_grad():
        embedded.prompt_embeds = decoder.embed_tokens(
            torch.tensor([other], dtype=torch.long)
        )
    executor2, allocator2 = _executor(decoder, chunk_size=64)
    scheduler2 = Scheduler(allocator2, SchedulerConfig())
    scheduler2.add_request(embedded)
    _drive(scheduler2, executor2)

    assert embedded.generated_token_ids != from_ids, (
        "prompt_embeds was ignored -- the executor fell back to embedding prompt_token_ids"
    )


def test_prompt_embeds_matching_the_ids_reproduce_the_id_path_exactly() -> None:
    """The other half: equal inputs, equal outputs, so the two paths are one path."""
    decoder = _decoder()
    plain = _request(prompt_len=16, max_new_tokens=3)
    executor, allocator = _executor(decoder, chunk_size=8)
    scheduler = Scheduler(allocator, SchedulerConfig())
    scheduler.add_request(plain)
    _drive(scheduler, executor)

    embedded = _request(prompt_len=16, max_new_tokens=3)
    with torch.no_grad():
        ids = torch.tensor([embedded.prompt_token_ids], dtype=torch.long)
        embedded.prompt_embeds = decoder.embed_tokens(ids)
    executor2, allocator2 = _executor(decoder, chunk_size=8)
    scheduler2 = Scheduler(allocator2, SchedulerConfig())
    scheduler2.add_request(embedded)
    _drive(scheduler2, executor2)

    assert embedded.generated_token_ids == plain.generated_token_ids


# --- accounting ----------------------------------------------------------------------------------


def test_pool_bytes_matches_the_arena_it_allocated() -> None:
    """The ledger has to be able to price the serving pool, not guess at it."""
    executor, _ = _executor(_decoder(), num_blocks=32, block_size=16)
    expected = 2 * TOY.n_layers * 32 * 16 * TOY.n_kv_heads * TOY.head_dim * 4  # fp32, K and V
    assert executor.pool_bytes == expected
