"""Boot the RAG server: retrieval, quantized model, paged cache, streaming HTTP.

    python -m scripts.serve_rag                     # ship config, port 8000
    python -m scripts.serve_rag --arm fp16          # unquantized, for comparison

This is the assembly point -- it constructs every piece the earlier phases built and wires them
together, and contains no logic of its own worth testing beyond that. Defaults are the *measured*
ones, not plausible ones:

* **``LM8+ViT4`` by default** (INT8 language, INT4 vision). ``CONTEXT.md`` D24: 2.296 GiB against
  fp16's 4.185, at 0.4193 ANLS against 0.4378 -- a 0.31 sigma difference. The uniform INT4
  alternative is smaller still and keeps 60% of quality, which is not a trade worth defaulting to.
* **Text-only retrieval.** D22 measured the image-space query signal as noise (0.14x the text
  side's spread, mean per-query maximum indistinguishable from zero), so ``FlatIndex`` is built
  with no image embeddings at all. That also makes startup instant: embedding 362 pages costs ~50
  minutes and buys nothing measurable.
* **Chunked prefill at 512** (D18), so one 7,000-token retrieved prompt cannot monopolise the GPU
  while other requests decode (P-18).

* **The pool is sized from the 4 GiB budget, not chosen.** ``--budget-gib 4.0`` subtracts the
  measured weights and the measured activation term and spends what is left on blocks: **471
  blocks at 1.380 GiB, for 4.000 GiB total.** The previous default of 640 blocks was 1.875 GiB and
  put the pipeline at **4.495 GiB -- 507 MiB over the budget this project is named after**. Nobody
  had done the subtraction; the README printed the two numbers a paragraph apart.

  What a budget buys is a *prompt length*, and this one buys **7,472 tokens**. Requests longer
  than that are refused by admission rather than discovered as an OOM mid-decode -- which is 25%
  of the frozen trace, and the honest cost of the headline. ``--budget-gib 0`` restores sizing by
  ``--num-blocks`` for anyone who would rather serve every request and quote 4.5 GiB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from bench.serving import build_stack
from edgerag.core.loader import HEADLINE_MODEL

try:
    from edgerag.serve.app import ServerState, create_app
except ImportError as exc:  # pragma: no cover -- depends on how the env was installed
    # FastAPI is an optional extra, and a bare "No module named 'fastapi'" sends people looking
    # for a bug in their code rather than at their install command. Say what to run.
    raise ImportError(
        f"serving needs the optional extras and {exc.name!r} is missing. Install with:\n"
        '    pip install -e ".[serve]"'
    ) from exc

from scripts.colab_quant_ablation import MIXED_ARMS, arm_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def build_server(
    model_id: str = HEADLINE_MODEL,
    arm: str = "LM8+ViT4",
    bits: int = 4,
    group_size: int = 128,
    num_blocks: int = 640,
    block_size: int = 16,
    chunk_size: int = 512,
    budget_gib: float | None = None,
    k: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Assemble everything and return ``(app, engine)``. The engine is already running.

    The stack itself comes from :func:`bench.serving.build_stack`, which
    ``scripts/colab_poisson.py`` also drives. That is deliberate: a load test measuring a
    separately-assembled lookalike of this server would stop describing it the first time one of
    the two got a fix (``bench/pipeline.py`` makes the same argument about the request path).
    """
    stack = build_stack(
        model_id=model_id,
        quant_spec={} if arm == "fp16" else arm_spec(arm, bits),
        arm=arm,
        group_size=group_size,
        num_blocks=num_blocks,
        block_size=block_size,
        chunk_size=chunk_size,
        budget_gib=budget_gib,
        k=k,
        device=device,
        dtype=dtype,
    )
    app = create_app(
        ServerState(
            engine=stack.engine,
            tokenizer=stack.lm.processor.tokenizer,
            model_id=f"{model_id}:{arm}",
            rag=stack.rag,
        )
    )
    return app, stack.engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve EdgeRAG over HTTP")
    parser.add_argument("--model", default=HEADLINE_MODEL)
    parser.add_argument(
        "--arm", default="LM8+ViT4",
        choices=["fp16", "LM", "LM+ViT", "ViT", *MIXED_ARMS],
        help="quantization configuration; the default is D24's measured ship config",
    )
    parser.add_argument("--bits", type=int, default=4, help="ignored for mixed arms")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--budget-gib", type=float, default=4.0,
        help=(
            "size the block pool from this memory budget instead of from --num-blocks, so the "
            "4 GiB headline holds by construction. Needs the measured activation term from "
            "results/poisson_sweep.jsonl. Pass 0 to size by --num-blocks instead."
        ),
    )
    parser.add_argument(
        "--num-blocks", type=int, default=640,
        help="pool size when --budget-gib is 0. 640 x 16 tokens is 1.875 GiB, which with weights "
             "and activation lands at 4.495 GiB -- over the budget this project is named after.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("-k", type=int, default=5, help="pages retrieved per question")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed: pip install -e '.[serve]'", file=sys.stderr)
        return 1

    app, engine = build_server(
        model_id=args.model, arm=args.arm, bits=args.bits, group_size=args.group_size,
        num_blocks=args.num_blocks, chunk_size=args.chunk_size, k=args.k, device=args.device,
        budget_gib=args.budget_gib or None,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    print(f"\nserving on http://{args.host}:{args.port}  --  POST /v1/chat/completions\n")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
