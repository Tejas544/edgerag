"""The frozen workload trace, and the one place RAG prompts are assembled.

Every benchmark from Phase 1 to Phase 8 replays this trace. That is ``00_FOUNDATIONS.md`` §4
rule 5 -- "fix everything you're not measuring" -- applied at the project level: an ablation cell
run on Aug 17 must have received byte-identical inputs to the baseline run on Aug 9, or the
comparison measures two things at once.

**Retrieval here is a deterministic stub: the gold document plus a seeded sample of others.**
That is not a shortcut being hidden -- it is ``CONTEXT.md`` D8. Retrieval *quality* (hybrid
scoring, recall@k) is Phase 7 work. What Phases 1-6 need from retrieval is a realistic *shape*:
k documents of realistic size and page-type mix. Token counts, KV bytes, block occupancy, and
throughput are all governed by that shape and are entirely indifferent to which specific pages
retrieval returns.

When Phase 7 replaces the stub with real embeddings, the trace is regenerated once and every
benchmark is re-baselined deliberately -- not silently.

Prompt assembly lives here rather than in each script, because the gate, the baseline, and every
later ablation must build byte-identical prompts. Two copies of this function would drift, and the
drift would be invisible in the numbers.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from edgerag.retrieval.corpus import DATA_DIR, CorpusDoc, QueryItem

TRACE_PATH = DATA_DIR / "trace.jsonl"

#: Bump when prompt assembly changes in any way that alters token counts. Every trace records it,
#: and the baseline refuses to compare across versions.
PROMPT_FORMAT_VERSION = 1

DEFAULT_K = 5
DEFAULT_SEED = 1234
DEFAULT_MAX_TEXT_CHARS = 1500


@dataclass
class TraceEntry:
    """One replayable request."""

    query_id: str
    question: str
    answers: list[str]
    gold_doc_key: str
    retrieved_doc_keys: list[str]
    split: str
    k: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def build_trace(
    corpus: list[CorpusDoc],
    queries: list[QueryItem],
    k: int = DEFAULT_K,
    seed: int = DEFAULT_SEED,
) -> list[TraceEntry]:
    """Assemble the frozen trace.

    Determinism is load-bearing: this must produce identical output on the Windows dev box and on
    a Colab T4, or local correctness runs and published numbers describe different workloads.
    So the corpus is sorted by key before sampling (dict/set iteration order must not leak in),
    and each query draws from its own seeded RNG keyed by ``query_id`` rather than from one shared
    stream -- otherwise inserting a query anywhere shifts every subsequent draw.
    """
    keys = sorted(d.doc_key for d in corpus)
    entries: list[TraceEntry] = []

    for q in sorted(queries, key=lambda x: x.query_id):
        if q.gold_doc_key not in set(keys):
            continue
        # Per-query RNG: a stable trace under corpus or query-set edits.
        rng = random.Random(f"{seed}:{q.query_id}")
        pool = [key for key in keys if key != q.gold_doc_key]
        others = rng.sample(pool, min(k - 1, len(pool)))
        entries.append(
            TraceEntry(
                query_id=q.query_id,
                question=q.question,
                answers=list(q.answers),
                gold_doc_key=q.gold_doc_key,
                retrieved_doc_keys=[q.gold_doc_key, *others],
                split=q.split,
                k=k,
            )
        )
    return entries


def build_prompt_messages(
    docs: list[CorpusDoc],
    question: str,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> list[dict[str, Any]]:
    """Assemble the chat message list for one RAG request.

    The single source of truth for prompt shape. Changing anything here changes every token count
    in the project, so ``PROMPT_FORMAT_VERSION`` must be bumped alongside.
    """
    content: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        content.append({"type": "text", "text": f"Document {i + 1}:"})
        content.append({"type": "image"})
        if doc.text:
            content.append({"type": "text", "text": doc.text[:max_text_chars]})
    content.append({"type": "text", "text": f"Question: {question}\nAnswer:"})
    return [{"role": "user", "content": content}]


def trace_fingerprint(entries: list[TraceEntry]) -> str:
    """Content hash of the trace.

    Stamped into every benchmark record. Two results carrying different fingerprints were measured
    against different workloads and must not share a table -- which is the one failure mode that
    ``check_comparable`` cannot catch, because the held-constant manifest would look identical.
    """
    digest = hashlib.sha256()
    digest.update(f"v{PROMPT_FORMAT_VERSION}\n".encode())
    for entry in entries:
        digest.update(entry.to_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def write_trace(entries: list[TraceEntry], path: Path = TRACE_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(entry.to_json() + "\n")
    return trace_fingerprint(entries)


def load_trace(path: Path = TRACE_PATH) -> list[TraceEntry]:
    with path.open("r", encoding="utf-8") as fh:
        return [TraceEntry(**json.loads(line)) for line in fh if line.strip()]
