"""Freeze the workload trace.

    python -m scripts.build_trace --k 5

Writes ``data/trace.jsonl`` and prints its fingerprint. The trace is committed to git, so the
Colab measurement tier and the local correctness tier replay byte-identical requests.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from edgerag.retrieval.corpus import load_corpus, load_queries
from edgerag.retrieval.trace import (
    DEFAULT_K,
    DEFAULT_SEED,
    PROMPT_FORMAT_VERSION,
    TRACE_PATH,
    build_trace,
    write_trace,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the EdgeRAG workload trace")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    corpus = load_corpus()
    queries = load_queries()
    entries = build_trace(corpus, queries, k=args.k, seed=args.seed)
    fingerprint = write_trace(entries)

    splits = Counter(e.split for e in entries)
    shared = Counter(e.gold_doc_key for e in entries)
    n_shared_docs = sum(1 for c in shared.values() if c > 1)
    max_share = max(shared.values()) if shared else 0

    print(f"  entries         {len(entries)} (k={args.k})")
    print(f"  splits          {dict(splits)}")
    print(f"  prompt format   v{PROMPT_FORMAT_VERSION}")
    print(f"  gold docs with >1 query: {n_shared_docs} (max {max_share})")
    print(f"\n  FINGERPRINT     {fingerprint}")
    print(f"\nwrote {TRACE_PATH}")
    print(
        "\nThis fingerprint is stamped into every benchmark record. Two results with different\n"
        "fingerprints were measured against different workloads and must not share a table."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
