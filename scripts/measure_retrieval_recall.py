"""Phase 7: recall@k for hybrid retrieval, against text-only and image-only controls.

    python -m scripts.measure_retrieval_recall
    python -m scripts.measure_retrieval_recall --model HuggingFaceTB/SmolVLM2-2.2B-Instruct

Runs entirely on the local tier (``CONTEXT.md`` D4) -- this is a *correctness and quality*
measurement, not a latency one, so it needs no T4. It does need real time: embedding every corpus
document costs one vision-tower forward per page, measured at ~9s/image on a GTX 1650 with the
256M fixture (``embed_image`` is not the fast path anything else in this project calls; nothing
upstream of it was built for throughput). 362 documents is under an hour; budget more for the
2.2B headline model's larger tower.

**Document embeddings are cached to disk, one line per document, flushed as each is computed.**
The first run of this script cost that hour and then crashed one line later on an unrelated bug
in the query-embedding step -- correct in the loop that had already spent 56 minutes, wrong in
the seconds-long loop after it, and because nothing had been persisted, the expensive half had to
be redone from nothing. That is ``BUGS.md`` B-05's shape again: a long-running measurement with no
checkpoint turns any late failure into a full re-run. The cache is what the Colab scripts already
do for exactly this reason (append-and-flush per arm/config); a local script that runs for the
better part of an hour deserves the same discipline.

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
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.core.model import load_from_hf
from edgerag.retrieval.corpus import load_corpus
from edgerag.retrieval.embed import embed_image, embed_query_for_image_space
from edgerag.retrieval.index import DEFAULT_ALPHA, FlatIndex, recall_at_k
from edgerag.retrieval.trace import load_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "results" / "embedding_cache"


def _cache_path(model_id: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", model_id)
    return CACHE_DIR / f"{slug}.jsonl"


def _load_cache(model_id: str) -> dict[str, np.ndarray]:
    path = _cache_path(model_id)
    if not path.exists():
        return {}
    cached = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row["doc_key"]] = np.array(row["embedding"], dtype=np.float32)
    return cached


def _append_cache(model_id: str, doc_key: str, vector: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _cache_path(model_id).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"doc_key": doc_key, "embedding": vector.tolist()}) + "\n")
        fh.flush()


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
    # embed_image takes the raw HF module (it drives encode_images_chunked, which expects HF's
    # tree). embed_query_for_image_space needs embed_tokens on *our* decoder -- the HF model has
    # no such top-level attribute (it is nested under model.text_model), so this has to be built.
    decoder = load_from_hf(lm.spec, lm.model)

    docs = load_corpus()
    trace = load_trace()
    queries = [e for e in trace if e.split == "heldout"]
    if args.n_queries:
        queries = queries[: args.n_queries]
    print(f"{len(docs)} corpus documents, {len(queries)} held-out queries")

    image_embeddings = _load_cache(args.model)
    todo = [d for d in docs if d.doc_key not in image_embeddings]
    if image_embeddings:
        print(f"{len(image_embeddings)}/{len(docs)} document embeddings already cached")
    if todo:
        print(f"embedding {len(todo)} documents ({args.model}, {device})...")
        t0 = time.time()
        for i, doc in enumerate(todo):
            image_path = REPO_ROOT / doc.image_path
            with Image.open(image_path).convert("RGB") as image:
                vector = embed_image(lm.model, lm.processor, image, device)
            image_embeddings[doc.doc_key] = vector
            _append_cache(args.model, doc.doc_key, vector)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(todo) - i - 1) / rate if rate else 0
                print(f"  {i + 1}/{len(todo)}  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)")

    index = FlatIndex.build(docs, image_embeddings, alpha=args.alpha)
    print(f"index built: {len(docs)} docs, vocab_size={index.vectorizer.vocab_size}, "
          f"{sum(1 for d in docs if d.text)} with OCR text\n")

    print(f"embedding {len(queries)} query questions...")
    query_vectors = {
        q.query_id: embed_query_for_image_space(decoder, lm.processor, q.question, device)
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

    # Why, not just whether: a hybrid score that never beats text-only could mean the image side
    # is a genuine zero, or it could mean alpha is too small to let a real signal through. The
    # spread of each side's similarity scores across documents answers that without ambiguity --
    # a signal with real ranking power varies a lot document-to-document; noise clusters near
    # zero regardless of how it is weighted (CONTEXT.md D22).
    text_matrix = np.stack([index.vectorizer.transform(q.question) for q in queries])
    text_sim = text_matrix @ index.text_vectors.T  # (n_queries, n_docs)
    image_ids = [q.query_id for q in queries if q.query_id in query_vectors]
    diagnostics = {
        "text_sim_mean_std": float(text_sim.std(axis=1).mean()),
        "text_sim_mean_max": float(text_sim.max(axis=1).mean()),
    }
    if image_ids and index.image_vectors.shape[1]:
        image_matrix = np.stack([query_vectors[qid] for qid in image_ids])
        image_sim = image_matrix @ index.image_vectors.T
        diagnostics["image_sim_mean_std"] = float(image_sim.std(axis=1).mean())
        diagnostics["image_sim_mean_max"] = float(image_sim.max(axis=1).mean())
        ratio = diagnostics["image_sim_mean_std"] / diagnostics["text_sim_mean_std"]
        print(f"  score spread: text std={diagnostics['text_sim_mean_std']:.4f}  "
              f"image std={diagnostics['image_sim_mean_std']:.4f}  "
              f"({ratio:.2f}x) -- image_sim mean max={diagnostics['image_sim_mean_max']:+.4f}")

    docs_by_key = {d.doc_key: d for d in docs}
    with_text = sum(1 for q in queries if docs_by_key[q.gold_doc_key].text)
    text_ceiling = with_text / max(len(queries), 1)
    diagnostics["text_reachable_ceiling"] = text_ceiling
    print(f"  structural ceiling for text-only: {text_ceiling:.1%} of queries have a gold doc "
          "with OCR text at all")

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
                "diagnostics": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
