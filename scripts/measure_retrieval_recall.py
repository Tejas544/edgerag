"""Phase 7: recall@k for hybrid retrieval, against text-only and image-only controls.

    python -m scripts.measure_retrieval_recall
    python -m scripts.measure_retrieval_recall --model HuggingFaceTB/SmolVLM2-2.2B-Instruct

Runs entirely on the local tier (``CONTEXT.md`` D4) -- this is a *correctness and quality*
measurement, not a latency one, so it needs no T4. It does need real time: embedding every corpus
document costs one vision-tower forward per page, measured at ~8s/image on a GTX 1650 with the
256M fixture (``embed_image`` is not the fast path anything else in this project calls; nothing
upstream of it was built for throughput). 362 documents is comfortably under an hour; budget more
for the 2.2B headline model's larger tower.

**This script exists to answer one question honestly: does the image-space query signal
(``embed_query_for_image_space``, projecting query text through ``embed_tokens`` into the same
space the connector emits image vectors in) carry any real relevance signal, or does it not.**
``edgerag/retrieval/embed.py`` states plainly that nothing trained this projection contrastively,
so the answer is not assumed -- it is read off ``recall_at_k``'s three columns, same as every other
ablation in this project compares against a naive baseline rather than trusting its own headline
number (FastV's ``attention`` vs ``uniform``, D20).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.retrieval.corpus import load_corpus
from edgerag.retrieval.embed import embed_image, embed_query_for_image_space
from edgerag.retrieval.index import DEFAULT_ALPHA, FlatIndex, recall_at_k
from edgerag.retrieval.trace import load_trace

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7 retrieval recall (local tier)")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--n-queries", type=int, default=0, help="0 = all held-out queries")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    lm = load_model(args.model, device=device, dtype=dtype)

    docs = load_corpus()
    trace = load_trace()
    queries = [e for e in trace if e.split == "heldout"]
    if args.n_queries:
        queries = queries[: args.n_queries]
    print(f"{len(docs)} corpus documents, {len(queries)} held-out queries")

    print(f"embedding {len(docs)} documents ({args.model}, {device})...")
    t0 = time.time()
    image_embeddings: dict[str, object] = {}
    for i, doc in enumerate(docs):
        image_path = REPO_ROOT / doc.image_path
        with Image.open(image_path).convert("RGB") as image:
            image_embeddings[doc.doc_key] = embed_image(lm.model, lm.processor, image, device)
        if (i + 1) % 50 == 0 or i + 1 == len(docs):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(docs) - i - 1) / rate if rate else 0
            print(f"  {i + 1}/{len(docs)}  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)")

    index = FlatIndex.build(docs, image_embeddings, alpha=args.alpha)
    print(f"index built: {len(docs)} docs, vocab_size={index.vectorizer.vocab_size}, "
          f"{sum(1 for d in docs if d.text)} with OCR text\n")

    print(f"embedding {len(queries)} query questions...")
    query_vectors = {
        q.query_id: embed_query_for_image_space(lm.model, lm.processor, q.question, device)
        for q in queries
    }

    # TraceEntry doesn't carry `answers`/`question` under the same names load_queries()'s
    # QueryItem does in every field -- recall_at_k only reads .query_id/.question/.gold_doc_key,
    # which both share, so either input type works without adapting the harness.
    results = {}
    for k in args.k:
        results[k] = recall_at_k(index, queries, query_vectors, k)
        print(f"recall@{k}:  hybrid={results[k]['hybrid']:.3f}  "
              f"text_only={results[k]['text_only']:.3f}  "
              f"image_only={results[k]['image_only']:.3f}")

    verdict = "the image-space signal contributes" if any(
        results[k]["hybrid"] > results[k]["text_only"] + 1e-9 for k in args.k
    ) else "the image-space signal does not beat text-only alone"
    print(f"\n{verdict} at alpha={args.alpha} on this corpus.")

    out = REPO_ROOT / "results" / "retrieval_recall.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "alpha": args.alpha,
                "n_docs": len(docs),
                "n_docs_with_text": sum(1 for d in docs if d.text),
                "n_queries": len(queries),
                "vocab_size": index.vectorizer.vocab_size,
                "recall": {str(k): v for k, v in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
