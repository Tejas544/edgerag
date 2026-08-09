"""PHASE 1 GATE -- measure the assumption the Phase 4 differentiator rests on.

    python -m scripts.measure_token_ratio --k 5 --n-queries 40

``01_EDGERAG.md`` §5 Day 7 asserts *"a single image costs 500-2000 tokens ... the KV cache is
dominated by visual tokens."* That is true of Qwen2-VL. It is **not automatically true of
SmolVLM**, which pixel-shuffles the vision tower's output down by ``scale_factor**2`` -- 729
patches become 81 tokens on the 2.2B checkpoint.

So the differentiator survives only if image *splitting* puts the token count back. This script
measures that on the real corpus, before six days of work are spent on the assumption.

**Gate:** visual tokens / total prefill tokens at k=5.

* >= 50%  -> thesis holds, proceed to Phase 4 as planned
* <  50%  -> escalate: raise resolution, split harder, and only then consider the D1 model switch

Runs on the **processor alone** -- no model weights are downloaded or loaded, so the headline
2.2B numbers are measurable on the local tier despite the 2.2B not fitting in 4 GB of VRAM.
Token composition is a property of the processor and config, not of the weights.

Retrieved-document *identity* is stubbed (gold document plus a seeded sample of others). That is
deliberate and does not weaken the result: the ratio is governed by ``k`` and by the page mix,
not by which specific pages retrieval happens to return. Real retrieval lands with the frozen
trace; this gate only needs the composition.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Any

from PIL import Image

from edgerag.core.loader import FIXTURE_MODEL, HEADLINE_MODEL, load_spec
from edgerag.core.spec import ModelSpec
from edgerag.retrieval.corpus import DATA_DIR, CorpusDoc, load_corpus, load_queries

GATE_THRESHOLD = 0.50
MIB = 1024**2


@dataclass
class Composition:
    """Token composition of one assembled RAG prompt."""

    query_id: str
    total_tokens: int
    image_tokens: int
    text_tokens: int
    n_subimages: int
    k: int

    @property
    def visual_share(self) -> float:
        return self.image_tokens / self.total_tokens


def build_prompt_messages(docs: list[CorpusDoc], question: str, max_text_chars: int) -> list[dict]:
    """Assemble a realistic RAG prompt: k retrieved pages, their OCR text, then the question."""
    content: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        content.append({"type": "text", "text": f"Document {i + 1}:"})
        content.append({"type": "image"})
        if doc.text:
            content.append({"type": "text", "text": doc.text[:max_text_chars]})
    content.append({"type": "text", "text": f"Question: {question}\nAnswer:"})
    return [{"role": "user", "content": content}]


def measure(
    model_id: str,
    spec: ModelSpec,
    corpus: list[CorpusDoc],
    queries: list[Any],
    k: int,
    n_queries: int,
    max_text_chars: int,
    seed: int,
) -> tuple[list[Composition], dict[str, Any]]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    by_key = {d.doc_key: d for d in corpus}
    rng = random.Random(seed)

    sample = rng.sample(queries, min(n_queries, len(queries)))
    results: list[Composition] = []

    for q in sample:
        gold = by_key.get(q.gold_doc_key)
        if gold is None:
            continue
        others = rng.sample([d for d in corpus if d.doc_key != q.gold_doc_key], k - 1)
        docs = [gold, *others]

        images = [Image.open(DATA_DIR.parent / d.image_path).convert("RGB") for d in docs]
        messages = build_prompt_messages(docs, q.question, max_text_chars)
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        batch = processor(text=prompt, images=images, return_tensors="pt")

        ids = batch["input_ids"][0]
        n_image = int((ids == spec.image_token_id).sum())
        n_total = int(ids.numel())
        pv = batch.get("pixel_values")
        n_sub = int(pv.shape[1]) if pv is not None and pv.dim() >= 2 else 0

        for im in images:
            im.close()

        results.append(
            Composition(
                query_id=q.query_id,
                total_tokens=n_total,
                image_tokens=n_image,
                text_tokens=n_total - n_image,
                n_subimages=n_sub,
                k=k,
            )
        )

    shares = [r.visual_share for r in results]
    totals = [r.total_tokens for r in results]
    per_tok = spec.kv_bytes_per_token("float16")

    summary = {
        "model_id": model_id,
        "k": k,
        "n_queries_measured": len(results),
        "visual_share_mean": round(fmean(shares), 4),
        "visual_share_median": round(median(shares), 4),
        "visual_share_min": round(min(shares), 4),
        "visual_share_max": round(max(shares), 4),
        "prefill_tokens_median": int(median(totals)),
        "prefill_tokens_max": max(totals),
        "image_tokens_median": int(median(r.image_tokens for r in results)),
        "subimages_median": int(median(r.n_subimages for r in results)),
        "tokens_per_subimage": spec.visual_tokens_per_subimage,
        "kv_bytes_per_token": per_tok,
        "kv_mib_median_prompt": round(median(totals) * per_tok / MIB, 1),
        "kv_mib_from_visual_only": round(
            median(r.image_tokens for r in results) * per_tok / MIB, 1
        ),
        "gate_threshold": GATE_THRESHOLD,
        "gate_passed": bool(median(shares) >= GATE_THRESHOLD),
    }
    return results, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 gate: visual token share")
    parser.add_argument("--k", type=int, default=5, help="retrieved documents per query")
    parser.add_argument("--n-queries", type=int, default=40)
    parser.add_argument("--max-text-chars", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[HEADLINE_MODEL, FIXTURE_MODEL],
        help="headline first -- it is the one the gate verdict applies to",
    )
    args = parser.parse_args(argv)

    corpus = load_corpus()
    queries = load_queries()
    print(f"corpus: {len(corpus)} pages, {len(queries)} queries\n")

    all_summaries = []
    per_query: dict[str, list[dict[str, Any]]] = {}
    for model_id in args.models:
        spec = load_spec(model_id)
        results, summary = measure(
            model_id,
            spec,
            corpus,
            queries,
            args.k,
            args.n_queries,
            args.max_text_chars,
            args.seed,
        )
        all_summaries.append(summary)
        # Kept per-query so Phase 4 can plot the distribution rather than a single median --
        # the spread across page types is part of the result.
        per_query[model_id] = [
            {**asdict(r), "visual_share": round(r.visual_share, 4)} for r in results
        ]

        verdict = "PASS" if summary["gate_passed"] else "FAIL"
        print(f"=== {model_id} ===")
        print(
            f"  attention          {'MHA' if not spec.uses_gqa else f'GQA {spec.n_rep}:1'}"
            f"  |  {spec.kv_bytes_per_token() / 1024:.1f} KiB KV per token"
        )
        print(f"  tokens/sub-image   {spec.visual_tokens_per_subimage}")
        print(f"  sub-images (med)   {summary['subimages_median']}  for k={args.k} pages")
        print(
            f"  prefill tokens     median {summary['prefill_tokens_median']:,}  "
            f"max {summary['prefill_tokens_max']:,}"
        )
        print(
            f"  image tokens       median {summary['image_tokens_median']:,}"
        )
        print(
            f"  VISUAL SHARE       median {summary['visual_share_median']:.1%}  "
            f"(mean {summary['visual_share_mean']:.1%}, "
            f"range {summary['visual_share_min']:.1%}-{summary['visual_share_max']:.1%})"
        )
        print(
            f"  KV for one prompt  {summary['kv_mib_median_prompt']:.0f} MiB  "
            f"of which {summary['kv_mib_from_visual_only']:.0f} MiB is visual"
        )
        print(f"  GATE ({GATE_THRESHOLD:.0%})         {verdict}\n")

    out = DATA_DIR / "token_ratio_gate.json"
    out.write_text(
        json.dumps({"summaries": all_summaries, "per_query": per_query}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")

    return 0 if all_summaries[0]["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
