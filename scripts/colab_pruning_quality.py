"""Phase 4 quality curve: answer quality versus visual-token pruning ratio. T4 only.

    python -m scripts.colab_pruning_quality --drive /content/drive/MyDrive/edgerag

The companion to ``scripts/measure_pruning_memory.py``. That one computes MiB reclaimed exactly
and locally; this one measures what the reclaim costs. **Neither is publishable alone** -- the
memory curve without the quality curve is the half of the result that flatters us.

Runs our own decoder and our own greedy loop (``01_EDGERAG.md`` §2 forbids ``generate()``), with
the compressor wired in, so what is measured is the code that ships.

Scored with **ANLS** (Average Normalized Levenshtein Similarity), the standard DocVQA metric.
Exact match is too brittle for generative answers -- "0.28" versus "0.28%" is a real answer, and
exact match calls it a total failure. ANLS with the usual 0.5 threshold scores near-misses
partially and outright wrong answers zero.

Two controls, both necessary:

* ``uniform`` at every ratio. If attention-based selection does not beat evenly-spaced selection,
  the honest finding is "visual tokens are redundant", not "FastV works".
* ``keep_ratio=1.0``, which the tests prove is bit-identical to no compressor at all, so the
  baseline row is the true baseline rather than a near-miss.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from bench.metrics import assert_device_trusted, sync
from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.compressed import CompressedKVCache
from edgerag.compress.fastv import FastVCompressor, FastVConfig, build_visual_mask
from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.core.model import encode_images_chunked, load_from_hf, merge_image_features
from edgerag.retrieval.corpus import load_corpus
from edgerag.retrieval.trace import build_prompt_messages, load_trace, trace_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance, iterative to avoid recursion limits on long answers."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity against the best of several gold answers.

    Below ``threshold`` the score is zeroed rather than allowed to decay smoothly, which stops a
    long wrong answer from collecting partial credit for incidental character overlap.
    """
    prediction = prediction.strip().lower()
    best = 0.0
    for answer in answers:
        gold = answer.strip().lower()
        if not gold and not prediction:
            best = max(best, 1.0)
            continue
        denom = max(len(prediction), len(gold))
        if denom == 0:
            continue
        similarity = 1.0 - levenshtein(prediction, gold) / denom
        best = max(best, similarity)
    return best if best >= threshold else 0.0


@torch.inference_mode()
def generate(
    decoder: torch.nn.Module,
    hf_model: torch.nn.Module,
    processor: Any,
    spec: Any,
    entry: Any,
    docs_by_key: dict,
    config: FastVConfig,
    strategy: str,
    max_new_tokens: int,
    device: torch.device,
    cache: CompressedKVCache,
) -> tuple[str, dict[str, Any]]:
    """Greedy decode one trace request under a pruning configuration.

    ``cache`` is supplied by the caller and **reused across requests**. Constructing it here --
    which is what this function did originally -- allocates a fresh block pool per request:
    1024 blocks x 16 tokens x 192 KiB/token is **3.2 GiB**, on top of 4.18 GiB of weights, and it
    OOM'd on every single call. See ``BUGS.md`` B-05.
    """
    docs = [docs_by_key[k] for k in entry.retrieved_doc_keys if k in docs_by_key]
    images = [Image.open(REPO_ROOT / d.image_path).convert("RGB") for d in docs]
    prompt = processor.apply_chat_template(
        build_prompt_messages(docs, entry.question), add_generation_prompt=True
    )
    batch = processor(text=prompt, images=images, return_tensors="pt")
    for image in images:
        image.close()

    input_ids = batch["input_ids"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    features = encode_images_chunked(hf_model, pixel_values)
    embeds = decoder.embed_tokens(input_ids)
    embeds = merge_image_features(input_ids, embeds, features, spec.image_token_id)
    visual_mask = build_visual_mask(input_ids, spec.image_token_id)

    cache.reset()
    compressor = FastVCompressor(config, strategy=strategy) if config.enabled else None

    t0 = time.perf_counter()
    logits = decoder(
        inputs_embeds=embeds, cache=cache, compressor=compressor, visual_mask=visual_mask
    )
    sync()  # not torch.cuda.synchronize(): this must be runnable on CPU so a test can cover it
    ttft = time.perf_counter() - t0

    tokens: list[int] = []
    next_id = int(logits[0, -1].argmax())
    eos = processor.tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        if next_id == eos:
            break
        tokens.append(next_id)
        step = decoder(
            input_ids=torch.tensor([[next_id]], device=device), cache=cache
        )
        next_id = int(step[0, -1].argmax())

    saving = cache.savings()
    return processor.tokenizer.decode(tokens, skip_special_tokens=True), {
        "ttft_s": ttft,
        "prefill_tokens": int(input_ids.shape[1]),
        **saving,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 quality curve (T4)")
    parser.add_argument("--drive", default="")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument("--n-queries", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--score-layer", type=int, default=2)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=576,
        help=(
            "block pool size. 576 x 16 tokens = 9,216 slots ~= 1.8 GiB at 192 KiB/token, which "
            "covers the longest observed prompt (~8k) plus generation. The pool is allocated ONCE "
            "and reused; oversizing it wastes VRAM the model needs (BUGS.md B-05)."
        ),
    )
    parser.add_argument(
        "--keep-ratios", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.375, 0.25, 0.125]
    )
    parser.add_argument("--strategies", nargs="+", default=["attention", "uniform"])
    parser.add_argument("--allow-untrusted-device", action="store_true")
    args = parser.parse_args(argv)

    info = assert_device_trusted(allow_untrusted=args.allow_untrusted_device)
    print(f"device: {info.name}")

    corpus = load_corpus()
    docs_by_key = {d.doc_key: d for d in corpus}
    trace = load_trace()
    fingerprint = trace_fingerprint(trace)
    # Held-out only: these documents were split off by document, never by question, so no
    # question about a held-out page appears in the eval split (CONTEXT.md D8 / corpus.py).
    heldout = [e for e in trace if e.split == "heldout"][: args.n_queries]
    print(f"trace {fingerprint} | {len(heldout)} held-out requests\n")

    lm = load_model(args.model, device="cuda", dtype=torch.float16)
    decoder = load_from_hf(lm.spec, lm.model)

    out_dir = Path(args.drive) if args.drive else (REPO_ROOT / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pruning_quality.jsonl"

    # One pool for the whole run. Allocated here, reset between requests.
    cache = CompressedKVCache(
        lm.spec,
        BlockAllocator(args.num_blocks, 16),
        lm.device,
        torch.float16,
        score_layer=args.score_layer,
    )
    print(f"block pool: {args.num_blocks} blocks x 16 tokens = "
          f"{cache.full.allocator.num_blocks * 16:,} slots\n")

    total_scored = 0
    for strategy in args.strategies:
        for ratio in args.keep_ratios:
            if ratio == 1.0 and strategy != args.strategies[0]:
                continue  # the no-op baseline is strategy-independent; measure it once
            config = FastVConfig(
                keep_ratio=ratio, score_layer=args.score_layer, score_mode="last_row"
            )
            scores, savings, ttfts = [], [], []
            oom_count = 0

            for entry in heldout:
                try:
                    text, stats = generate(
                        decoder, lm.model, lm.processor, lm.spec, entry, docs_by_key,
                        config, strategy, args.max_new_tokens, lm.device, cache,
                    )
                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    cache.reset()
                    torch.cuda.empty_cache()
                    continue
                scores.append(anls(text, entry.answers))
                savings.append(stats["mib_reclaimed"])
                ttfts.append(stats["ttft_s"])

            total_scored += len(scores)
            if oom_count:
                print(f"    (skipped {oom_count}/{len(heldout)} requests: out of memory)")

            record = {
                "strategy": strategy,
                "keep_ratio": ratio,
                "score_layer": args.score_layer,
                "n_scored": len(scores),
                "anls": sum(scores) / len(scores) if scores else 0.0,
                "mib_reclaimed": sum(savings) / len(savings) if savings else 0.0,
                "ttft_s_mean": sum(ttfts) / len(ttfts) if ttfts else 0.0,
                "workload_fingerprint": fingerprint,
                "device": info.name,
            }
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
            print(f"  {strategy:>9} keep={ratio:<6.3f} ANLS={record['anls']:.4f} "
                  f"reclaimed={record['mib_reclaimed']:.0f} MiB "
                  f"TTFT={record['ttft_s_mean']:.2f}s  (n={record['n_scored']})")

    print(f"\nwrote {out_path}")

    # A run that scored nothing must FAIL, not emit a tidy file of zeros. The first version of
    # this script OOM'd on every request, swallowed it, and wrote ANLS 0.0 across the board --
    # a plausible-looking "quality curve" showing that pruning destroys quality. A result that is
    # silently empty is more dangerous than a crash, because it gets published (BUGS.md B-05).
    if total_scored == 0:
        print(
            "\nFAILED: not a single request was scored. Every one was skipped, so the numbers "
            "above are zeros by default rather than by measurement. Do not use this file.",
            file=sys.stderr,
        )
        return 1

    expected = len(heldout)
    for line in out_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["n_scored"] < expected * 0.5:
            print(
                f"\nWARNING: {record['strategy']} keep={record['keep_ratio']} scored only "
                f"{record['n_scored']}/{expected} requests -- the mean is over a biased subset.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
