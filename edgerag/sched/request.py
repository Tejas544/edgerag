"""Request lifecycle for the continuous-batching scheduler.

Phase 5a. No tensors, no GPU -- the same discipline the allocator was built with, and for the same
reason: a scheduler bug that presents as wrong output is misdiagnosed as a cache bug for hours.
Proving the state machine in isolation means Phase 5b can exclude it.

The state machine is small but the transitions are where the interesting failures live:

* ``WAITING -> PREFILLING`` is admission, and it can fail on block exhaustion.
* ``PREFILLING -> PREFILLING`` is chunked prefill making progress across iterations. This is the
  transition ``01_EDGERAG.md`` does not mention and the one that fixes D14's 25-second TTFT.
* ``* -> SWAPPED -> WAITING`` is preemption (D16). A swapped request re-enters admission rather
  than resuming directly, because ``swap_in`` needs blocks that may no longer exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RequestState(StrEnum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    SWAPPED = "swapped"
    FINISHED = "finished"
    REJECTED = "rejected"


#: States from which a request still holds KV blocks and can therefore be preempted.
PREEMPTABLE = (RequestState.PREFILLING, RequestState.DECODING)

#: States a request never leaves.
TERMINAL = (RequestState.FINISHED, RequestState.REJECTED)


@dataclass
class Request:
    """One in-flight generation request.

    ``prefill_offset`` is what makes chunked prefill expressible: the request remembers how much of
    its prompt has been processed, so the scheduler can hand it a slice per iteration rather than
    monopolising the GPU for a 6,800-token prefill while every decoding request stalls.
    """

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int = 64
    arrival_step: int = 0
    state: RequestState = RequestState.WAITING

    prefill_offset: int = 0
    generated_token_ids: list[int] = field(default_factory=list)

    #: Set when the request is admitted; the scheduler owns the type.
    cache: Any = None

    #: Iteration indices, for TTFT and end-to-end latency.
    admitted_step: int | None = None
    first_token_step: int | None = None
    finished_step: int | None = None
    preemptions: int = 0

    # --- progress -----------------------------------------------------------------------------

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def prefill_done(self) -> bool:
        return self.prefill_offset >= self.prompt_len

    @property
    def num_generated(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_tokens(self) -> int:
        """Positions occupied in the KV cache: the whole prompt plus everything generated."""
        return self.prompt_len + self.num_generated

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL

    def next_prefill_chunk(self, chunk_size: int) -> list[int]:
        """The next slice of prompt to process, without advancing.

        Separate from :meth:`advance_prefill` so a failed forward pass does not leave the request
        believing it made progress it did not make.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        end = min(self.prefill_offset + chunk_size, self.prompt_len)
        return self.prompt_token_ids[self.prefill_offset : end]

    def advance_prefill(self, n_tokens: int) -> None:
        if n_tokens < 0:
            raise ValueError(f"cannot advance prefill by {n_tokens}")
        if self.prefill_offset + n_tokens > self.prompt_len:
            raise ValueError(
                f"{self.request_id}: prefill would advance to "
                f"{self.prefill_offset + n_tokens} past prompt length {self.prompt_len}"
            )
        self.prefill_offset += n_tokens

    def append_token(self, token_id: int, step: int) -> None:
        if self.first_token_step is None:
            self.first_token_step = step
        self.generated_token_ids.append(token_id)

    def should_stop(self, eos_token_id: int | None = None) -> bool:
        if self.num_generated >= self.max_new_tokens:
            return True
        return bool(
            eos_token_id is not None
            and self.generated_token_ids
            and self.generated_token_ids[-1] == eos_token_id
        )

    # --- preemption ---------------------------------------------------------------------------

    def mark_preempted(self) -> None:
        """Send a running request back to the queue.

        ``prefill_offset`` is deliberately **not** reset: a swapped request keeps its KV, so it
        resumes where it stopped. A recompute-policy preemption must reset it explicitly, and that
        asymmetry is why the two policies cannot share one code path (``CONTEXT.md`` D16).
        """
        self.state = RequestState.SWAPPED
        self.preemptions += 1

    def reset_for_recompute(self) -> None:
        """Discard cached progress so the prompt is prefilled again from scratch."""
        self.prefill_offset = 0
        self.cache = None
        self.state = RequestState.WAITING

    # --- reporting ----------------------------------------------------------------------------

    def ttft_steps(self) -> int | None:
        if self.first_token_step is None:
            return None
        return self.first_token_step - self.arrival_step

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "state": self.state.value,
            "prompt_len": self.prompt_len,
            "prefill_offset": self.prefill_offset,
            "generated": self.num_generated,
            "total_tokens": self.total_tokens,
            "preemptions": self.preemptions,
            "ttft_steps": self.ttft_steps(),
            "arrival_step": self.arrival_step,
            "finished_step": self.finished_step,
        }
