"""Phase 7 gate: ``/v1/chat/completions`` streams tokens.

Driven through FastAPI's ``TestClient`` against the *real* engine and a fake executor, so the HTTP
surface, the SSE framing and the thread bridge are all exercised together -- with no checkpoint, no
``transformers``, and no GPU. What is asserted is what a client actually depends on: the wire
format, incremental delivery, and that failures arrive as failures rather than as a hang.
"""

from __future__ import annotations

import json
import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from edgerag.cache.allocator import BlockAllocator  # noqa: E402
from edgerag.sched.scheduler import Batch, Scheduler, SchedulerConfig  # noqa: E402
from edgerag.serve.app import ServerState, create_app  # noqa: E402
from edgerag.serve.engine import InferenceEngine, StepOutput  # noqa: E402

EOS = 2


class FakeTokenizer:
    """Character-level, so decoded output is predictable and multi-token behaviour is visible."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text][:64] or [65]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t) for t in token_ids if not (skip_special_tokens and t == EOS))


class ScriptedExecutor:
    def __init__(self, script: list[int]) -> None:
        self.script = script

    def execute(self, batch: Batch) -> StepOutput:
        output = StepOutput()
        for request in batch.prefill:
            output.prefilled[request.request_id] = request.prompt_len - request.prefill_offset
        for request in batch.decode:
            if request.num_generated < len(self.script):
                output.tokens[request.request_id] = self.script[request.num_generated]
        return output


class ExplodingExecutor:
    def execute(self, batch: Batch) -> StepOutput:
        raise RuntimeError("CUDA out of memory")


@pytest.fixture
def client_factory():
    engines = []

    def make(executor=None, start: bool = True):
        scheduler = Scheduler(BlockAllocator(64, 16), SchedulerConfig(eos_token_id=EOS))
        engine = InferenceEngine(
            scheduler, executor or ScriptedExecutor([72, 73, EOS]), idle_poll_seconds=0.001
        )
        if start:
            engine.start()
        engines.append(engine)
        state = ServerState(engine=engine, tokenizer=FakeTokenizer(), model_id="edgerag-test")
        return TestClient(create_app(state)), engine

    yield make
    for engine in engines:
        engine.stop()


def _sse_events(body: str) -> list[dict]:
    """Parse the payloads out of an SSE body, skipping the [DONE] terminator."""
    events = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


# --- the gate ----------------------------------------------------------------------------------


def test_streaming_delivers_tokens_and_terminates_with_done(client_factory) -> None:
    client, _ = client_factory()
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 5},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert body.endswith("data: [DONE]\n\n")
    events = _sse_events(body)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events if "choices" in e
    )
    assert text == "HI", f"expected the decoded script, got {text!r}"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_streaming_chunks_are_openai_shaped(client_factory) -> None:
    """Clients parse these fields by name; getting the envelope wrong breaks every one of them."""
    client, _ = client_factory()
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        events = _sse_events("".join(response.iter_text()))

    for event in events:
        assert event["object"] == "chat.completion.chunk"
        assert event["id"].startswith("chatcmpl-")
        assert event["model"] == "edgerag-test"
        assert isinstance(event["created"], int)
        assert event["choices"][0]["index"] == 0


def test_non_streaming_returns_one_completion_with_usage(client_factory) -> None:
    client, _ = client_factory()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "HI"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3  # includes EOS, which decodes to nothing
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_max_tokens_is_honoured_and_reported_as_length(client_factory) -> None:
    client, _ = client_factory(ScriptedExecutor([72, 73, 74, 75, 76]))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
    )
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "HI"
    assert body["choices"][0]["finish_reason"] == "length"


# --- the event loop must stay free -------------------------------------------------------------


def test_the_forward_pass_does_not_run_on_the_event_loop(client_factory) -> None:
    """P-17 through the full HTTP path, not just the engine unit test."""
    seen: set[str] = set()

    class ThreadRecordingExecutor(ScriptedExecutor):
        def execute(self, batch: Batch) -> StepOutput:
            seen.add(threading.current_thread().name)
            return super().execute(batch)

    client, _ = client_factory(ThreadRecordingExecutor([72, EOS]))
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert seen == {"edgerag-engine"}, f"executor ran on {seen}"


# --- failure paths -----------------------------------------------------------------------------


def test_a_crash_mid_stream_is_reported_in_the_stream(client_factory) -> None:
    """The status code is already sent by then, so the error has to arrive as an SSE event."""
    client, _ = client_factory(ExplodingExecutor())
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "CUDA out of memory" in body
    assert body.endswith("data: [DONE]\n\n"), "the stream must still terminate cleanly"


def test_a_crash_without_streaming_is_a_500(client_factory) -> None:
    client, _ = client_factory(ExplodingExecutor())
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 500
    assert "CUDA out of memory" in response.json()["detail"]


def test_requests_are_refused_when_the_engine_is_not_running(client_factory) -> None:
    client, engine = client_factory()
    engine.stop()
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 503


def test_empty_messages_are_rejected(client_factory) -> None:
    client, _ = client_factory()
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


@pytest.mark.parametrize(
    "payload", [{"temperature": 0.7}, {"top_p": 0.9}]
)
def test_sampling_is_refused_rather_than_silently_ignored(client_factory, payload) -> None:
    """Greedy output returned to a client that asked for temperature=0.7 is a quiet lie."""
    client, _ = client_factory()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **payload},
    )
    assert response.status_code == 400
    assert "not implemented" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{"temperature": 0}, {"top_p": 1}, {}])
def test_greedy_equivalent_parameters_are_accepted(client_factory, payload) -> None:
    """temperature=0 IS what this server does, so rejecting it would be pedantry."""
    client, _ = client_factory()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **payload},
    )
    assert response.status_code == 200


# --- health ------------------------------------------------------------------------------------


def test_health_reports_the_worker_thread_not_just_http(client_factory) -> None:
    """A dead engine thread still answers HTTP instantly and hangs every generation."""
    client, engine = client_factory()
    assert client.get("/health").json()["status"] == "ok"

    engine.stop()
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["engine_running"] is False


def test_health_exposes_engine_counters(client_factory) -> None:
    client, _ = client_factory()
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    body = client.get("/health").json()
    assert body["submitted"] == 1
    assert body["completed"] == 1


# --- the real executor through the full HTTP path -----------------------------------------------


def test_a_request_generates_through_the_real_decoder_over_the_paged_pool() -> None:
    """No fake executor: our decoder, the paged arena, the scheduler, the engine, and HTTP.

    A toy 2-layer spec keeps it to milliseconds. What this covers that the fake cannot is that the
    pieces *fit together* -- prompt ids in, chunked prefill, first token from prefill, decode
    against a real KV cache, tokens out as SSE. Every earlier layer of this stack was tested with
    the layer below it stubbed; this is the one test where nothing is.
    """
    import torch

    from edgerag.core.model import EdgeRagDecoder
    from edgerag.core.spec import ModelSpec
    from edgerag.serve.executor import ModelExecutor

    spec = ModelSpec(
        model_id="toy", model_type="smolvlm", n_layers=2, hidden_size=128, n_q_heads=4,
        n_kv_heads=2, head_dim=32, vocab_size=256, max_position_embeddings=512,
        rope_theta=10000.0, vision_layers=2, vision_hidden=64, vision_image_size=32,
        vision_patch_size=16, scale_factor=2, image_token_id=255, pad_token_id=0,
        intermediate_size=256,
    )
    torch.manual_seed(0)
    decoder = EdgeRagDecoder(spec).eval()
    allocator = BlockAllocator(64, 16)
    executor = ModelExecutor(
        decoder, spec, allocator, torch.device("cpu"), torch.float32, chunk_size=8
    )
    scheduler = Scheduler(allocator, SchedulerConfig())
    engine = InferenceEngine(scheduler, executor, idle_poll_seconds=0.001)
    engine.start()

    class ToyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [(ord(c) % 200) + 1 for c in text]

        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            return " ".join(str(t) for t in token_ids)

    try:
        client = TestClient(
            create_app(ServerState(engine=engine, tokenizer=ToyTokenizer(), model_id="toy"))
        )
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "what is the total"}],
                  "max_tokens": 4},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        assert content, "the model produced no tokens"
        assert len(content.split()) == 4, f"expected 4 generated tokens, got {content!r}"
        assert body["usage"]["completion_tokens"] == 4
        assert body["choices"][0]["finish_reason"] == "length"
    finally:
        engine.stop()

    # Every block handed out must have come back, or a real server leaks its pool one request at
    # a time and dies hours in.
    assert allocator.num_free == allocator.num_blocks


def test_retrieval_sources_are_reported_on_the_response() -> None:
    """A RAG answer whose sources are not returned cannot be checked against them."""

    class StubRag:
        def build(self, question: str):
            from edgerag.serve.pipeline import RetrievedPrompt

            return RetrievedPrompt(
                token_ids=[5, 6, 7, 8], embeds=None, doc_keys=["docvqa:1:0", "docvqa:2:1"]
            )

    scheduler = Scheduler(BlockAllocator(64, 16), SchedulerConfig(eos_token_id=EOS))
    engine = InferenceEngine(
        scheduler, ScriptedExecutor([72, EOS]), idle_poll_seconds=0.001
    )
    engine.start()
    try:
        client = TestClient(
            create_app(
                ServerState(engine=engine, tokenizer=FakeTokenizer(), rag=StubRag())
            )
        )
        body = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "total?"}]}
        ).json()
        assert body["retrieved"] == ["docvqa:1:0", "docvqa:2:1"]
        assert body["usage"]["prompt_tokens"] == 4, "usage must count the RETRIEVED prompt"
    finally:
        engine.stop()


def test_streaming_reports_its_sources_in_the_opening_chunk() -> None:
    """Streaming clients see the answer before the response ends; sources must arrive too."""

    class StubRag:
        def build(self, question: str):
            from edgerag.serve.pipeline import RetrievedPrompt

            return RetrievedPrompt(token_ids=[5, 6], embeds=None, doc_keys=["infographic:9:0"])

    scheduler = Scheduler(BlockAllocator(64, 16), SchedulerConfig(eos_token_id=EOS))
    engine = InferenceEngine(scheduler, ScriptedExecutor([72, EOS]), idle_poll_seconds=0.001)
    engine.start()
    try:
        client = TestClient(
            create_app(ServerState(engine=engine, tokenizer=FakeTokenizer(), rag=StubRag()))
        )
        with client.stream(
            "POST", "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "q"}], "stream": True},
        ) as response:
            events = _sse_events("".join(response.iter_text()))
        assert events[0]["choices"][0]["delta"]["retrieved"] == ["infographic:9:0"]
    finally:
        engine.stop()


@pytest.mark.slow
def test_the_whole_server_answers_a_document_question_on_the_fixture(tmp_path) -> None:
    """Everything at once: retrieval, a quantized model, the paged cache, and HTTP.

    ``scripts/serve_rag.py`` is the assembly point, and an assembly point that is never assembled
    is exactly the shape of BUGS.md B-05. This boots it for real on the 256M fixture -- INT8
    language beside INT4 vision, the same mixed configuration D24 measured -- puts a question in,
    and requires a non-empty answer with the pages it came from.
    """
    pytest.importorskip("transformers")
    import torch

    from edgerag.core.loader import FIXTURE_MODEL
    from edgerag.retrieval.corpus import CORPUS_PATH
    from scripts.serve_rag import build_server

    if not CORPUS_PATH.exists():
        pytest.skip("corpus not built; run scripts.build_corpus")

    app, engine = build_server(
        model_id=FIXTURE_MODEL, arm="LM8+ViT4", group_size=64, num_blocks=512,
        chunk_size=256, k=2, device="cpu", dtype=torch.float32,
    )
    try:
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "ok"

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "What is the total revenue?"}],
                  "max_tokens": 8},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["retrieved"], "no pages were retrieved"
        assert len(body["retrieved"]) == 2, "k=2 was requested"
        assert body["choices"][0]["message"]["content"].strip(), "the model answered nothing"
        # The prompt is the retrieved pages, not the bare question -- if retrieval or the vision
        # merge silently dropped out, this collapses to a handful of tokens.
        assert body["usage"]["prompt_tokens"] > 500, (
            f"prompt was only {body['usage']['prompt_tokens']} tokens -- retrieved pages missing"
        )
    finally:
        engine.stop()
