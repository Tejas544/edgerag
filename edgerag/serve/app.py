"""OpenAI-compatible HTTP surface. Phase 7's gate: ``/v1/chat/completions`` streams tokens.

Compatible with the OpenAI chat-completions shape on purpose, not to be fashionable: it means
every existing client, benchmark harness and ``curl`` snippet works against this server unchanged,
and it removes "invent an API" from a phase whose actual content is the threading underneath
(``edgerag/serve/engine.py``, ``BUGS.md`` P-17).

**Nothing in this module touches a tensor.** Handlers translate JSON into a
:class:`~edgerag.sched.request.Request`, hand it to the engine, and format whatever comes back.
The engine's worker thread does the arithmetic. That separation is the whole point of the phase,
so it is worth stating where a reader will look for it.

The tokenizer is injected rather than imported for the same reason the executor is: it keeps the
entire HTTP surface testable without ``transformers``, a checkpoint, or a GPU.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from edgerag.sched.request import Request
from edgerag.serve.engine import InferenceEngine

#: OpenAI's terminator. Clients watch for exactly this line, so it is not ours to redesign.
SSE_DONE = "data: [DONE]\n\n"


class Tokenizer(Protocol):
    """The two operations serving needs. ``transformers`` tokenizers satisfy this already."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """The subset of OpenAI's schema this server honours.

    Unsupported knobs are accepted and ignored rather than rejected -- a client that always sends
    ``temperature`` should not get a 422 from a server that happens to decode greedily. What is
    *not* ignored is anything that would silently change the answer: see ``sampling`` below.
    """

    model: str = "edgerag"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=4096)
    stream: bool = False
    #: Accepted for compatibility and **not implemented**: decoding here is greedy (argmax).
    #: Returning silently-greedy output to a client that asked for temperature=1.5 would be a
    #: quiet lie, so a request that actually asks for sampling is refused in `_reject_sampling`.
    temperature: float | None = None
    top_p: float | None = None


@dataclass
class ServerState:
    """Everything a handler needs. Assembled by the caller so tests can substitute all of it."""

    engine: InferenceEngine
    tokenizer: Tokenizer
    model_id: str = "edgerag"
    #: Turns a chat conversation into prompt token ids. Defaults to a plain concatenation.
    build_prompt: Any = None
    #: A :class:`~edgerag.serve.pipeline.RagPipeline`. When set, the last user message is treated
    #: as a question: pages are retrieved, encoded, and merged into the prompt embeddings, and the
    #: keys of the pages actually used are reported back on the response. Absent, the server is a
    #: plain text completion endpoint -- which is what every test that does not need a checkpoint
    #: exercises.
    rag: Any = None


def _last_user_question(messages: list[ChatMessage]) -> str:
    """The question to retrieve against: the last user turn, or the last turn if none is user."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content


def default_prompt_builder(tokenizer: Tokenizer, messages: Iterable[ChatMessage]) -> list[int]:
    """Flatten a conversation into tokens, with no chat template.

    Deliberately dull. A real template belongs to the checkpoint, and the RAG pipeline overrides
    this entirely -- putting a guessed template here would produce plausible-looking output that
    silently disagrees with how the model was trained.
    """
    text = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return tokenizer.encode(text)


def _reject_sampling(payload: ChatCompletionRequest) -> None:
    """Refuse what is not implemented instead of quietly doing something else.

    ``temperature=0`` and ``top_p=1`` mean greedy, which is what this server does, so those pass.
    Anything else asks for sampling that does not exist here.
    """
    if payload.temperature not in (None, 0, 0.0):
        raise HTTPException(
            status_code=400,
            detail="sampling is not implemented: decoding is greedy. Send temperature=0 or omit.",
        )
    if payload.top_p not in (None, 1, 1.0):
        raise HTTPException(
            status_code=400,
            detail="top_p is not implemented: decoding is greedy. Send top_p=1 or omit.",
        )


def _chunk(completion_id: str, model: str, created: int, **choice: Any) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, **choice}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="EdgeRAG", version="0.1.0")
    build_prompt = state.build_prompt or default_prompt_builder

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Reports whether the *worker thread* is alive, not merely whether HTTP is up.

        A server whose engine thread has died still answers HTTP instantly and hangs every
        generation -- which is the failure a naive health check is guaranteed to miss.
        """
        return {
            "status": "ok" if state.engine.is_running else "degraded",
            "engine_running": state.engine.is_running,
            "model": state.model_id,
            **state.engine.stats.to_dict(),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest) -> Any:
        _reject_sampling(payload)
        if not payload.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if not state.engine.is_running:
            raise HTTPException(status_code=503, detail="engine is not running")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        retrieved: list[str] = []

        if state.rag is not None:
            # Retrieval and the vision tower run once per request, here -- not per prefill chunk
            # inside the executor, which would re-encode the same pages ~14 times for a 7k prompt
            # (see edgerag/serve/pipeline.py).
            built = await run_in_threadpool(state.rag.build, _last_user_question(payload.messages))
            prompt_ids, prompt_embeds, retrieved = built.token_ids, built.embeds, built.doc_keys
        else:
            prompt_ids, prompt_embeds = build_prompt(state.tokenizer, payload.messages), None

        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt encoded to zero tokens")

        request = Request(
            request_id=completion_id,
            prompt_token_ids=list(prompt_ids),
            max_new_tokens=payload.max_tokens,
            prompt_embeds=prompt_embeds,
        )

        if payload.stream:
            return StreamingResponse(
                _stream(state, request, completion_id, created, retrieved),
                media_type="text/event-stream",
                # Proxies that buffer will hold every token until the response completes, which
                # turns a streaming endpoint into a slow non-streaming one with no visible error.
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await _collect(
            state, request, completion_id, created, len(prompt_ids), retrieved
        )

    return app


async def _stream(
    state: ServerState, request: Request, completion_id: str, created: int,
    retrieved: list[str] | None = None,
) -> AsyncIterator[str]:
    """SSE, decoded incrementally.

    Tokens are decoded against the growing sequence rather than one at a time, because a decoder
    that sees one token in isolation cannot reconstruct multi-token characters -- byte-level BPE
    splits a single emoji or accented letter across several ids, and per-token decoding emits
    replacement characters where the text should be.
    """
    # The retrieved pages ride on the opening chunk: a client that streams an answer and never
    # learns which documents produced it cannot check the answer against them.
    opening: dict[str, Any] = {"role": "assistant"}
    if retrieved:
        opening["retrieved"] = retrieved
    yield _chunk(completion_id, state.model_id, created, delta=opening, finish_reason=None)

    token_ids: list[int] = []
    text_so_far = ""
    async for event in state.engine.stream(request):
        if event.error:
            # The stream has already started, so the status code is long gone -- the only honest
            # place left to report a failure is in the stream itself.
            yield _chunk(completion_id, state.model_id, created,
                         delta={}, finish_reason="error")
            yield f"data: {json.dumps({'error': {'message': event.error}})}\n\n"
            break
        if event.token_id is not None:
            token_ids.append(event.token_id)
            text = state.tokenizer.decode(token_ids, skip_special_tokens=True)
            delta, text_so_far = text[len(text_so_far):], text
            if delta:
                yield _chunk(completion_id, state.model_id, created,
                             delta={"content": delta}, finish_reason=None)
        if event.done:
            yield _chunk(completion_id, state.model_id, created,
                         delta={}, finish_reason=event.finish_reason or "stop")
            break

    yield SSE_DONE


async def _collect(
    state: ServerState, request: Request, completion_id: str, created: int, prompt_tokens: int,
    retrieved: list[str] | None = None,
) -> dict[str, Any]:
    token_ids: list[int] = []
    finish_reason = "stop"
    async for event in state.engine.stream(request):
        if event.error:
            raise HTTPException(status_code=500, detail=event.error)
        if event.token_id is not None:
            token_ids.append(event.token_id)
        if event.done:
            finish_reason = event.finish_reason or "stop"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": state.model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": state.tokenizer.decode(token_ids, skip_special_tokens=True),
                },
                "finish_reason": finish_reason,
            }
        ],
        "retrieved": retrieved or [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_tokens + len(token_ids),
        },
    }
