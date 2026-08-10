"""Paged vs naive KV memory on the real workload, and what prefix ordering costs.

    python -m scripts.measure_paged_memory

Memory only -- no latency. Allocation sizes are exact and device-independent, so this runs on the
local tier under ``CONTEXT.md`` D4 ("correctness and memory accounting only") and needs neither a
T4 nor model weights. Per-document token counts come from the real processor; everything else is
arithmetic over the frozen trace.

Three strategies are compared at the headline model's 192 KiB/token (D10):

1. **naive** -- one contiguous reservation of ``max_seq_len`` per sequence, whatever it uses.
2. **paged** -- ``ceil(tokens / block_size)`` blocks per sequence.
3. **paged + prefix sharing** -- queries whose prompts begin with the same documents share those
   blocks copy-on-write.

The third exposes something the plan does not anticipate. Sharing requires a common *prefix*, and
our prompts place the gold document first followed by different neighbours, so queries about the
same page share only that one document. Reordering retrieved documents into a **canonical**
(id-sorted) order instead of relevance order lengthens the common prefix substantially. That is a
real, cheap lever, and it is measured here rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from edgerag.core.loader import HEADLINE_MODEL, load_spec
from edgerag.core.spec import ModelSpec
from edgerag.retrieval.corpus import DATA_DIR, load_corpus
from edgerag.retrieval.trace import load_trace

MIB = 1024**2
GIB = 1024**3
TOKENS_CACHE = DATA_DIR / "doc_token_costs.json"


def measure_document_tokens(model_id: str, corpus: list, force: bool = False) -> dict[str, int]:
    """Token cost of each corpus page, measured with the real processor.

    Cached to disk: it is a pure function of the corpus and the processor config, and it is the
    slow part of this script.
    """
    if TOKENS_CACHE.exists() and not force:
        return json.loads(TOKENS_CACHE.read_text(encoding="utf-8"))

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    costs: dict[str, int] = {}

    for i, doc in enumerate(corpus):
        image = Image.open(DATA_DIR.parent / doc.image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Document 1:"},
                    {"type": "image"},
                    *([{"type": "text", "text": doc.text[:1500]}] if doc.text else []),
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        batch = processor(text=prompt, images=[image], return_tensors="pt")
        costs[doc.doc_key] = int(batch["input_ids"].shape[-1])
        image.close()
        if (i + 1) % 50 == 0:
            print(f"  measured {i + 1}/{len(corpus)} pages")

    TOKENS_CACHE.write_text(json.dumps(costs), encoding="utf-8")
    return costs


def blocks_for(tokens: int, block_size: int) -> int:
    return (tokens + block_size - 1) // block_size


def common_prefix_length(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def analyse(
    spec: ModelSpec,
    entries: list,
    doc_tokens: dict[str, int],
    block_size: int,
    canonical_order: bool,
) -> dict[str, Any]:
    """Blocks required to hold every request concurrently, with and without sharing."""
    per_token = spec.kv_bytes_per_token("float16")

    def doc_list(entry) -> list[str]:
        keys = [k for k in entry.retrieved_doc_keys if k in doc_tokens]
        return sorted(keys) if canonical_order else keys

    sequences = [(e, doc_list(e)) for e in entries]
    sequences = [(e, d) for e, d in sequences if d]

    # --- unshared: every sequence pays for all its documents ---
    total_tokens = sum(sum(doc_tokens[k] for k in docs) for _, docs in sequences)
    paged_blocks = sum(blocks_for(sum(doc_tokens[k] for k in docs), block_size)
                       for _, docs in sequences)

    # --- shared: group by leading document, credit the common prefix once per group ---
    groups: dict[str, list[list[str]]] = defaultdict(list)
    for _, docs in sequences:
        groups[docs[0]].append(docs)

    shared_blocks = 0
    shared_prefix_docs = 0
    for docs_in_group in groups.values():
        prefix = docs_in_group[0]
        for other in docs_in_group[1:]:
            prefix = prefix[: common_prefix_length(prefix, other)]

        prefix_tokens = sum(doc_tokens[k] for k in prefix)
        shared_prefix_docs += len(prefix) * len(docs_in_group)
        # The shared prefix is stored once, in whole blocks.
        shared_blocks += blocks_for(prefix_tokens, block_size)
        # Each member then stores its own suffix.
        for docs in docs_in_group:
            suffix_tokens = sum(doc_tokens[k] for k in docs[len(prefix) :])
            shared_blocks += blocks_for(suffix_tokens, block_size)

    capacity_tokens = paged_blocks * block_size
    return {
        "block_size": block_size,
        "canonical_order": canonical_order,
        "n_sequences": len(sequences),
        "total_tokens": total_tokens,
        "paged_blocks": paged_blocks,
        "paged_gib": paged_blocks * block_size * per_token / GIB,
        "shared_blocks": shared_blocks,
        "shared_gib": shared_blocks * block_size * per_token / GIB,
        "sharing_saving_pct": 100 * (1 - shared_blocks / paged_blocks) if paged_blocks else 0.0,
        "internal_frag_pct": 100 * (capacity_tokens - total_tokens) / capacity_tokens
        if capacity_tokens
        else 0.0,
        "mean_shared_prefix_docs": shared_prefix_docs / len(sequences) if sequences else 0.0,
    }


def concurrency(
    spec: ModelSpec, kv_budget_bytes: int, tokens_per_seq: int, block_size: int
) -> dict:
    """How many sequences fit in a KV budget, naive vs paged."""
    per_token = spec.kv_bytes_per_token("float16")
    naive_per_seq = spec.max_position_embeddings * per_token
    paged_per_seq = blocks_for(tokens_per_seq, block_size) * block_size * per_token
    return {
        "naive_per_seq_gib": naive_per_seq / GIB,
        "paged_per_seq_gib": paged_per_seq / GIB,
        "naive_max_seqs": kv_budget_bytes // naive_per_seq,
        "paged_max_seqs": kv_budget_bytes // paged_per_seq,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paged vs naive KV memory on the frozen trace")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--kv-budget-gib", type=float, default=2.5,
                        help="4 GiB budget minus ~1.05 GiB INT4 weights and activation headroom")
    parser.add_argument("--force-remeasure", action="store_true")
    args = parser.parse_args(argv)

    spec = load_spec(args.model)
    corpus = load_corpus()
    entries = load_trace()

    print(f"model {spec.model_id}")
    print(f"  {spec.kv_bytes_per_token() / 1024:.0f} KiB KV per token "
          f"({'MHA' if not spec.uses_gqa else f'GQA {spec.n_rep}:1'})")
    print(f"corpus {len(corpus)} pages | trace {len(entries)} requests\n")

    print("measuring per-document token cost with the real processor ...")
    doc_tokens = measure_document_tokens(args.model, corpus, force=args.force_remeasure)
    values = sorted(doc_tokens.values())
    print(f"  median {values[len(values) // 2]} tokens/page, "
          f"range {values[0]}-{values[-1]}\n")

    # --- block size sweep -------------------------------------------------------------------
    print("=== block size sweep (all 650 requests resident) ===")
    print(f"{'block':>6} {'paged GiB':>10} {'int.frag':>9} "
          f"{'shared GiB':>11} {'saving':>8} {'canon GiB':>10} {'canon saving':>13}")
    rows = []
    for bs in args.block_sizes:
        relevance = analyse(spec, entries, doc_tokens, bs, canonical_order=False)
        canonical = analyse(spec, entries, doc_tokens, bs, canonical_order=True)
        rows.append({"relevance": relevance, "canonical": canonical})
        print(f"{bs:>6} {relevance['paged_gib']:>10.2f} "
              f"{relevance['internal_frag_pct']:>8.1f}% "
              f"{relevance['shared_gib']:>11.2f} "
              f"{relevance['sharing_saving_pct']:>7.1f}% "
              f"{canonical['shared_gib']:>10.2f} "
              f"{canonical['sharing_saving_pct']:>12.1f}%")

    # --- concurrency ------------------------------------------------------------------------
    median_tokens = sorted(
        sum(doc_tokens[k] for k in e.retrieved_doc_keys if k in doc_tokens) for e in entries
    )[len(entries) // 2]
    budget = int(args.kv_budget_gib * GIB)
    print(f"\n=== concurrency within a {args.kv_budget_gib:.2f} GiB KV budget ===")
    print(f"median request: {median_tokens} tokens")
    conc = {}
    for bs in (8, 16, 32):
        c = concurrency(spec, budget, median_tokens, bs)
        conc[bs] = c
        print(f"  block {bs:>2}: naive {c['naive_per_seq_gib']:.2f} GiB/seq -> "
              f"{c['naive_max_seqs']} seqs | paged {c['paged_per_seq_gib']:.2f} GiB/seq -> "
              f"{c['paged_max_seqs']} seqs")

    out = Path("results/paged_memory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": spec.model_id,
                "kv_bytes_per_token": spec.kv_bytes_per_token(),
                "median_request_tokens": median_tokens,
                "kv_budget_gib": args.kv_budget_gib,
                "block_size_sweep": rows,
                "concurrency": {str(k): v for k, v in conc.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
