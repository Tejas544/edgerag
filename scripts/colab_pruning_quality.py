"""Phase 4 quality curve: answer quality versus visual-token pruning ratio. T4 only.

    python -m scripts.colab_pruning_quality --drive /content/drive/MyDrive/edgerag

The companion to ``scripts/measure_pruning_memory.py``. That one computes MiB reclaimed exactly
and locally; this one measures what the reclaim costs. **Neither is publishable alone** -- the
memory curve without the quality curve is the half of the result that flatters us.

Runs our own decoder and our own greedy loop (``01_EDGERAG.md`` §2 forbids ``generate()``), with
the compressor wired in, so what is measured is the code that ships.

The request pipeline and the ANLS metric moved to ``bench/`` in Phase 6, when the quantization
ablation needed both. Two experiments that claim to hold everything constant except one variable
have to share the code that holds it constant.

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
from pathlib import Path

import torch

from bench.metrics import anls, assert_device_trusted
from bench.pipeline import free_duplicate_hf_decoder, generate
from edgerag.cache.allocator import BlockAllocator
from edgerag.cache.compressed import CompressedKVCache
from edgerag.compress.fastv import FastVConfig
from edgerag.core.loader import HEADLINE_MODEL, load_model
from edgerag.core.model import load_from_hf
from edgerag.retrieval.corpus import load_corpus
from edgerag.retrieval.trace import load_trace, trace_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        default=1152,
        help=(
            "block pool size. The compressed cache needs blocks for BOTH halves, and at "
            "keep_ratio=1.0 -- the ablation's own baseline -- neither half is pruned, so a "
            "7k-token prompt wants ~440 blocks twice. 1152 covers that with headroom at ~3.5 GiB. "
            "The pool is allocated once and reused (BUGS.md B-05)."
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

    # `load_from_hf` COPIES the decoder weights into ours, so from here the HF text decoder and
    # lm_head are duplicates -- roughly 3.6 GiB of a 14.6 GiB card held for nothing. Keeping both
    # is why every request OOM'd even after the per-request pool was fixed (BUGS.md B-05).
    # Per CONTEXT.md D2 the only things still needed from HF are the vision tower and connector.
    free_duplicate_hf_decoder(lm.model)

    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"after load: {reserved:.2f} GiB reserved of {total:.2f} GiB "
              f"({total - reserved:.2f} GiB free for cache and activations)")

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
    # Preflight: the worst case is keep_ratio=1.0, where NOTHING is pruned and both halves store
    # the whole sequence. Checking it here turns a mid-request OutOfBlocksError into arithmetic
    # printed before any GPU time is spent.
    longest = max(
        sum(len(docs_by_key[k].text) // 4 + 900 for k in e.retrieved_doc_keys if k in docs_by_key)
        for e in heldout
    )
    worst_case_blocks = 2 * ((longest + args.max_new_tokens + 15) // 16)
    print(
        f"block pool: {args.num_blocks} blocks x 16 tokens = {args.num_blocks * 16:,} slots "
        f"(~{args.num_blocks * 3 / 1024:.1f} GiB); worst case needs ~{worst_case_blocks}"
    )
    if worst_case_blocks > args.num_blocks:
        print(
            f"\nFAILED: at keep_ratio=1.0 neither half is pruned, so both store the full "
            f"sequence -- roughly {worst_case_blocks} blocks against a pool of {args.num_blocks}. "
            f"Re-run with --num-blocks {int(worst_case_blocks * 1.3)}.",
            file=sys.stderr,
        )
        return 1
    print()

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
                    # Bail on the first config that cannot fit a single request. If request 1
                    # OOMs, so will requests 2..60 and every later config -- continuing just
                    # spends 25 minutes of scarce T4 quota proving it 660 times. Two OOMs in a
                    # row with nothing scored means the configuration is wrong, not the data.
                    if oom_count >= 2 and not scores:
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(
                            f"\nABORTING: the first {oom_count} requests both ran out of memory, "
                            f"so nothing will fit. {reserved:.2f} GiB reserved of {total:.2f}. "
                            "Reduce --num-blocks or --n-queries, or check that the duplicate HF "
                            "decoder was freed. Not spending the rest of the session on this.",
                            file=sys.stderr,
                        )
                        return 1
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
