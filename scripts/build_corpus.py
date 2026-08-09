"""Build and freeze the EdgeRAG corpus.

    python -m scripts.build_corpus --infographic 250 --docvqa 400

Writes ``data/corpus/*.jpg``, ``data/corpus.jsonl``, ``data/queries.jsonl``. Run once; the corpus
is frozen after that so every benchmark from Phase 1 onward replays identical inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from edgerag.retrieval.corpus import (
    CORPUS_PATH,
    DATA_DIR,
    QUERIES_PATH,
    assign_heldout,
    build_corpus,
    write_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen EdgeRAG corpus")
    parser.add_argument("--infographic", type=int, default=250, help="InfographicVQA rows to pull")
    parser.add_argument("--docvqa", type=int, default=400, help="DocVQA rows to pull")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)

    print(f"streaming {args.infographic} InfographicVQA + {args.docvqa} DocVQA rows ...")
    docs, queries = build_corpus(
        n_infographic=args.infographic, n_docvqa=args.docvqa, split=args.split
    )
    assign_heldout(queries, fraction=args.heldout_fraction)

    write_jsonl(CORPUS_PATH, docs)
    write_jsonl(QUERIES_PATH, queries)

    by_source = Counter(d.source for d in docs)
    with_text = sum(1 for d in docs if d.n_text_chars > 0)
    q_per_doc = Counter(q.gold_doc_key for q in queries)
    shared = sum(1 for c in q_per_doc.values() if c > 1)
    heldout = sum(1 for q in queries if q.split == "heldout")

    summary = {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "docs_by_source": dict(by_source),
        "docs_with_ocr_text": with_text,
        "mean_text_chars": round(sum(d.n_text_chars for d in docs) / max(len(docs), 1), 1),
        "docs_with_multiple_queries": shared,
        "max_queries_per_doc": max(q_per_doc.values()) if q_per_doc else 0,
        "queries_eval": len(queries) - heldout,
        "queries_heldout": heldout,
    }

    # BUGS.md B-01: the OCR parser fails by returning "" rather than raising, so absence of an
    # exception proves nothing. Coverage is the only signal that actually catches it.
    infographic_docs = [d for d in docs if d.source == "infographicvqa"]
    if infographic_docs:
        covered = sum(1 for d in infographic_docs if d.n_text_chars > 0)
        coverage = covered / len(infographic_docs)
        if coverage < 0.80:
            print(
                f"\nFAILED: OCR text extracted for only {covered}/{len(infographic_docs)} "
                f"({coverage:.1%}) InfographicVQA pages. Expected >=80%. "
                "The parser degrades to an empty string on unexpected shapes -- see BUGS.md B-01.",
                file=sys.stderr,
            )
            return 1

    (DATA_DIR / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n  {len(docs)} unique pages from {len(queries)} questions")
    print(f"  by source: {dict(by_source)}")
    print(f"  with OCR text: {with_text} (mean {summary['mean_text_chars']:.0f} chars)")
    print(
        f"  pages with >1 question: {shared} (max {summary['max_queries_per_doc']}) "
        "<- the prefix-sharing workload"
    )
    print(f"  eval/heldout queries: {summary['queries_eval']}/{heldout}")
    print(f"\nwrote {CORPUS_PATH}, {QUERIES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
