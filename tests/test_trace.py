"""Tests for the frozen workload trace.

Determinism is the whole point. The trace is built on Windows and replayed on a Colab T4; if the
two diverge, local correctness runs and published numbers describe different workloads and every
comparison in the project is quietly invalid.
"""

from __future__ import annotations

from edgerag.retrieval.corpus import CorpusDoc, QueryItem
from edgerag.retrieval.trace import (
    build_prompt_messages,
    build_trace,
    trace_fingerprint,
)


def _corpus(n: int = 20) -> list[CorpusDoc]:
    return [
        CorpusDoc(
            doc_key=f"src:{i}:0",
            source="src",
            doc_id=str(i),
            page_no=0,
            image_path=f"data/corpus/{i}.jpg",
            width=800,
            height=1000,
            text=f"text for doc {i}" if i % 2 == 0 else "",
            n_text_chars=16 if i % 2 == 0 else 0,
        )
        for i in range(n)
    ]


def _queries(n: int = 10) -> list[QueryItem]:
    return [
        QueryItem(
            query_id=f"q{i:03d}",
            source="src",
            question=f"question {i}?",
            answers=[f"a{i}"],
            gold_doc_key=f"src:{i}:0",
        )
        for i in range(n)
    ]


# --- determinism --------------------------------------------------------------------------


def test_trace_is_byte_identical_across_builds() -> None:
    a = build_trace(_corpus(), _queries(), k=5, seed=1234)
    b = build_trace(_corpus(), _queries(), k=5, seed=1234)
    assert trace_fingerprint(a) == trace_fingerprint(b)


def test_trace_is_independent_of_input_ordering() -> None:
    """Dict and set iteration order must not leak into the trace.

    The corpus arrives from a streamed dataset and the queries from a JSONL file; neither has a
    guaranteed order across machines.
    """
    corpus, queries = _corpus(), _queries()
    forward = build_trace(corpus, queries, k=5, seed=1234)
    reversed_ = build_trace(corpus[::-1], queries[::-1], k=5, seed=1234)
    assert trace_fingerprint(forward) == trace_fingerprint(reversed_)


def test_per_query_rng_keeps_entries_stable_when_a_query_is_inserted() -> None:
    """A shared RNG stream would shift every subsequent query's draw when one is added.

    That would silently re-baseline the whole project on a corpus edit.
    """
    queries = _queries(10)
    base = {e.query_id: e.retrieved_doc_keys for e in build_trace(_corpus(), queries, seed=1234)}

    extra = QueryItem(
        query_id="q005b",
        source="src",
        question="inserted?",
        answers=["x"],
        gold_doc_key="src:5:0",
    )
    after = {
        e.query_id: e.retrieved_doc_keys
        for e in build_trace(_corpus(), [*queries, extra], seed=1234)
    }

    for qid, retrieved in base.items():
        assert after[qid] == retrieved, f"{qid} changed when an unrelated query was inserted"


def test_different_seeds_produce_different_traces() -> None:
    a = build_trace(_corpus(), _queries(), seed=1)
    b = build_trace(_corpus(), _queries(), seed=2)
    assert trace_fingerprint(a) != trace_fingerprint(b)


def test_fingerprint_tracks_prompt_format_version() -> None:
    """A prompt-format change alters every token count, so it must invalidate the fingerprint."""
    import edgerag.retrieval.trace as trace_mod

    entries = build_trace(_corpus(), _queries())
    before = trace_fingerprint(entries)
    original = trace_mod.PROMPT_FORMAT_VERSION
    try:
        trace_mod.PROMPT_FORMAT_VERSION = original + 1
        assert trace_fingerprint(entries) != before
    finally:
        trace_mod.PROMPT_FORMAT_VERSION = original


# --- trace shape --------------------------------------------------------------------------


def test_gold_document_is_always_present_and_first() -> None:
    for entry in build_trace(_corpus(), _queries(), k=5):
        assert entry.retrieved_doc_keys[0] == entry.gold_doc_key
        assert len(entry.retrieved_doc_keys) == 5


def test_retrieved_documents_are_unique() -> None:
    """A duplicated page would double-count its visual tokens and corrupt the KV accounting."""
    for entry in build_trace(_corpus(), _queries(), k=5):
        assert len(set(entry.retrieved_doc_keys)) == len(entry.retrieved_doc_keys)


def test_queries_with_missing_gold_document_are_dropped() -> None:
    queries = [*_queries(3), QueryItem("qX", "src", "?", ["a"], gold_doc_key="src:999:0")]
    entries = build_trace(_corpus(), queries, k=3)
    assert {e.query_id for e in entries} == {"q000", "q001", "q002"}


def test_k_larger_than_corpus_is_clamped_not_crashed() -> None:
    entries = build_trace(_corpus(3), _queries(3), k=10)
    assert all(len(e.retrieved_doc_keys) == 3 for e in entries)


# --- prompt assembly ----------------------------------------------------------------------


def test_prompt_interleaves_one_image_per_document() -> None:
    docs = _corpus(3)
    messages = build_prompt_messages(docs, "what is the total?")
    content = messages[0]["content"]
    assert sum(1 for c in content if c["type"] == "image") == 3
    assert content[-1]["text"].endswith("Answer:")
    assert "what is the total?" in content[-1]["text"]


def test_prompt_omits_the_text_block_for_documents_without_ocr() -> None:
    """DocVQA pages ship no OCR. An empty text block would still cost tokens for nothing."""
    with_text = build_prompt_messages([_corpus(2)[0]], "q")  # doc 0 has text
    without = build_prompt_messages([_corpus(2)[1]], "q")  # doc 1 does not
    assert len(with_text[0]["content"]) == len(without[0]["content"]) + 1


def test_prompt_truncates_long_ocr_text() -> None:
    doc = _corpus(1)[0]
    doc.text = "x" * 10_000
    content = build_prompt_messages([doc], "q", max_text_chars=100)[0]["content"]
    body = next(c for c in content if c["type"] == "text" and c["text"].startswith("x"))
    assert len(body["text"]) == 100
