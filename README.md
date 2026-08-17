# EdgeRAG

**A complete multimodal RAG pipeline — image + text corpus → retrieval → VLM generation — running
entirely within a fixed 4 GB memory budget.**

No LangChain. No LlamaIndex. No `model.generate()`. The paged KV-cache allocator, the
continuous-batching scheduler, the visual-token compressor, and the quantized linear layers are
written here, not imported.

> **Status: Phase 6 of 8 (Aug 17, 2026).** The decoder, paged KV cache, scheduler, visual-token
> compressor and INT4 quantization are built and measured. Retrieval quality and the serving layer
> are Phase 7. Every number below is measured or exactly computed, with its source file named;
> what is still unmeasured is listed under [Known limitations](#known-limitations) rather than
> left blank. See [`PLAN.md`](PLAN.md) for the phase schedule.

---

## Why the constraint is the project

Anyone can wire up RAG. The interesting question is what you have to build differently when the
whole pipeline — model weights, KV cache, vision encoder, embeddings, and index — has to fit in
4 GB. Every design decision in [`CONTEXT.md`](CONTEXT.md) is downstream of that number.

The development GPU for this project is a **GTX 1650 with exactly 4096 MiB of VRAM**. The budget
is not a config flag on a large card; it is the machine.

---

## Architecture

_Diagram lands in Phase 8._

```
corpus (images + text)
   │
   ├─ offline: SigLIP tower ──► image embeddings ──┐
   └─ offline: text encoder ──► text embeddings ───┤
                                                   ▼
query ──► hybrid retrieval ──► top-k docs ──► prefix-shared KV blocks
                                                   │
                                     ┌─────────────┴─────────────┐
                                     │  visual token compression │
                                     └─────────────┬─────────────┘
                                                   ▼
                       paged KV cache ◄──► continuous-batching scheduler
                                                   │
                                         INT4 quantized VLM decode
                                                   ▼
                                         streaming response
```

---

## Benchmarks

### Baseline — HuggingFace `generate()`, SmolVLM2-2.2B, Tesla T4

Every later number is measured against this. Workload is the frozen trace `94b148a0b9f5006e`:
k=5 retrieved document pages per query, ~6,758 prefill tokens median, 75.4% of them visual.

| batch | TTFT p50 | tok/s per seq | tok/s aggregate | peak allocated |
|---:|---:|---:|---:|---:|
| 1 | 3,730 ms | 26.0 | 26.0 | **5.76 GiB** |
| 2 | 11,071 ms | 15.9 | 31.8 | 7.90 GiB |
| 4 | 25,006 ms | 8.6 | 34.6 | 12.41 GiB |

**The baseline does not fit the 4 GB target for even one request.** 5.76 GiB at batch 1, before
any concurrency. The problem is not that the naive pipeline is inefficient — it is that it does
not run on the target device.

Two things this table is saying that are easy to misread:

- **Aggregate throughput rises 1.32× for 4× the batch.** Per-sequence tok/s *falls*; that is
  batching working, not failing. The ceiling is low because this checkpoint is MHA (32 query
  heads, 32 KV heads), so decode is bandwidth-bound on KV reads at 192 KiB/token. Scheduling
  cannot recover that — only cutting KV bytes can.
- **TTFT degrades superlinearly** (1× → 2.97× → 6.7×), because left-padding pads every request to
  the batch maximum. This is what chunked prefill has to fix.

**KV cache on/off** at batch 1: 25.95 tok/s vs **0.26 tok/s — 100×**, against the 2–3× a
short-prompt workload would show. Without a cache each decode step re-attends the whole
6,758-token prefix, so one decode step costs one full prefill. Notably it saves only 0.24 GiB,
so at batch 1 the cache is nearly free in memory terms; paging earns its keep at concurrency.

Methodology is fixed and enforced in code, not by convention:

- warmup discarded, `torch.cuda.synchronize()` on both sides of every timed region
- p50/p95/p99 reported, never a bare mean; p99 flagged when computed from <100 samples
- ≥5 trials with standard deviation
- TTFT and decode throughput aggregated separately — prefill is compute-bound, decode is
  memory-bound, and conflating them hides the thing that matters
- peak-memory counter reset between runs
- every record stamps its device, driver, torch build, and held-constant manifest; the harness
  **refuses to tabulate runs that did not hold the same variables fixed**

**Reference hardware is a Tesla T4.** The harness will not write a performance record on any other
device without an explicit override that stamps the result `trusted: false`. The local GTX 1650
has no tensor cores, so its timings are architecturally incomparable — it is used for correctness
and memory accounting only.

### The memory budget, line by line

`results/memory_ledger.md`, computed exactly from the checkpoint config — no weights loaded, no
GPU involved, because quantized bytes are a function of tensor shape and nothing else. Weights
only, GiB:

| arm | bits | language | vision | connector | total | vs fp16 |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 16 | 3.376 | 0.769 | 0.040 | **4.185** | 1.00× |
| LM | 8 | 1.900 | 0.769 | 0.040 | 2.708 | 1.55× |
| LM+ViT | 8 | 1.900 | 0.515 | 0.020 | 2.435 | 1.72× |
| ViT | 8 | 3.376 | 0.515 | 0.020 | 3.911 | 1.07× |
| LM | 4 | 1.150 | 0.769 | 0.040 | 1.958 | 2.14× |
| **LM+ViT** | **4** | **1.150** | **0.386** | **0.010** | **1.546** | **2.71×** |
| ViT | 4 | 3.376 | 0.386 | 0.010 | 3.772 | 1.11× |

**INT4 buys 2.71×, not 4×, and the shortfall is three named line items** rather than a rounding
error: the `lm_head` at 192 MiB (deliberately fp16 — its error reaches `argmax` with nothing
downstream to attenuate it), embeddings and norms at 193 MiB (not linear layers at all), and 255
MiB of the vision MLP whose `in_features = 4304 = 16 × 269` no power-of-two group size above 16
divides. Quoting 4× means counting only the layers you quantized and calling that the model.

Assembled at LM+ViT/INT4 with one request in flight: **3.12 GiB of the 4 GiB budget**, leaving
0.88 GiB — about 1.7 concurrent requests at 1.25 GiB of KV each. The fp16 arm supports **zero**:
its weights alone exceed the budget before a single KV block is allocated. That is the honest
framing of this project — not "4× less memory" but *runs at all versus does not*.

The budget is defined as `torch.cuda.max_memory_allocated()` and **excludes the CUDA context**
(300–600 MiB on Turing), which is reported separately because it is a driver cost, not a pipeline
cost. One line of that table — the transient activation peak — is inferred from the measured T4
baseline rather than computed, and is labelled as such wherever it appears.

### What paging actually buys, and what it does not

Exact allocator arithmetic over all 650 trace requests (`results/paged_memory.json`):

- **Internal fragmentation is 0.0–0.5% for every block size from 1 to 64.** Block size does not
  matter on this workload; 16 is chosen because fragmentation is free and it minimises block-table
  length.
- **Prefix sharing saves 8.3%** — and **14.8% if retrieved documents are ordered canonically
  rather than by relevance**, a 78% improvement for a one-line change in prompt assembly. Ordering
  the retrieved set is a prefix-cache-hit-rate decision, not only a relevance decision.
- **The 4–8× concurrency multiple is not reachable here, and the README will not claim it.**
  Measured: 1 sequence → 2. RAG requests already sit at 81% of maximum context, so right-sizing
  the reservation can only ever recover 1.24×. What paging genuinely buys on this workload is no
  per-token reallocation and block-granularity admission and eviction — the latter is what makes
  the scheduler possible at all.

**The gather is the cost.** Paged attention itself is free (+0.55% against attention over an
already-contiguous cache), but materialising the gathered KV costs **72.7% of the paged attention
path** at the median request length — 23.5 ms per decode step against 8.8 ms of attention. A
head-major pool layout took ~20% off that and did not change the conclusion: the fused kernel is
required, not optional, and it is not written.

### Visual token pruning: the price, not just the saving

FastV at layer 2, 40 held-out requests (`results/pruning_quality.jsonl`):

| keep | ANLS (attention) | ANLS (uniform) | MiB reclaimed | TTFT |
|---:|---:|---:|---:|---:|
| 1.000 | 0.438 | — | 0 | 2.63 s |
| 0.750 | 0.381 | 0.265 | 224 | 2.00 s |
| 0.500 | 0.203 | 0.207 | 448 | 1.40 s |
| 0.375 | 0.149 | 0.234 | 560 | 1.14 s |
| 0.250 | 0.088 | 0.146 | 671 | 0.92 s |
| 0.125 | 0.053 | 0.076 | 783 | 0.71 s |

**The target was 50–75% of visual tokens removed with quality holding, and it is not reachable on
this workload.** Halving the visual tokens reclaims 448 MiB per request and costs 54% of ANLS;
there is no knee in the curve. The mechanism is the workload: on a document page the answer is
often a single number in one cell of one table, and a budget that drops half the page has an even
chance of dropping the cell. FastV was validated on natural images, where adjacent patches really
are redundant.

Attention-based scoring beats a uniform stride at keep=0.75 and **loses** below half — once the
budget is small, coverage beats salience. Read that table knowing n=40 gives a standard error near
0.06, so gaps under ~0.12 are ties; only the fall from keep=1.0 to keep=0.5 clearly clears it.

---

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev,serve]" --index-url https://download.pytorch.org/whl/cu124
pytest -q
python -m scripts.measure_memory_ledger
```

The memory ledger needs no GPU and no downloaded weights — it instantiates the checkpoint on the
`meta` device and prices it from shapes, in about two seconds. Everything that needs a T4 lives in
`scripts/colab_*.py` and refuses to record a number on any other device;
[`notebooks/COLAB.md`](notebooks/COLAB.md) is the runbook.

---

## Design decisions

Every non-obvious choice, with the alternative it beat and the condition that would reverse it,
is logged in [`CONTEXT.md`](CONTEXT.md).

## Known limitations

Populated as they are discovered rather than at the end. [`BUGS.md`](BUGS.md) carries the bug log
and the predicted failure modes this design is built to avoid.

- **INT4 is expected to be slower than fp16, and the number is not measured yet.** `QuantLinear`
  dequantizes then calls a normal matmul; a T4 has no INT4 tensor cores, so the packed bytes are
  expanded before they reach the multiplier. The memory win is real and immediate; the speed win
  needs the dequantize fused into the GEMV. The throughput and quality columns of the quantization
  ablation are the next T4 run (`scripts/colab_quant_ablation.py`).
- **The fused paged-attention kernel is not written.** The gather is 72.7% of the paged attention
  path, well past the 25% threshold at which the design log said to revisit the decision. Triton
  has no Windows support, so this is Colab-only work that has not been scheduled.
- **The pruning quality curve is n=40.** Standard error ~0.06; most gaps in that table are ties.
  More queries, not more ratios, is where the next hour of T4 time should go.
- **Retrieval is a stub.** The frozen trace plants the gold page first in every request, so
  recall@5 is 100% by construction and the 0.438 ANLS baseline is what the *generator* achieves
  with the right page guaranteed present. Real hybrid retrieval is Phase 7 and can only lower it.
- **The serving layer does not exist yet.** FastAPI, streaming, and the asyncio↔scheduler bridge
  are Phase 7.
- **The CUDA context is excluded from the 4 GB budget** and has not been measured on the T4. At
  300–600 MiB it is 8–15% of the ceiling, so it is reported as a separate line rather than folded
  in or quietly ignored.

## Repo map

| Path | What |
|---|---|
| [`PLAN.md`](PLAN.md) | Phase schedule, gates, cut order |
| [`CONTEXT.md`](CONTEXT.md) | Decision log — every choice with its rejected alternative |
| [`BUGS.md`](BUGS.md) | Bug log + predicted failure modes |
| [`notebooks/COLAB.md`](notebooks/COLAB.md) | T4 runbook — every GPU measurement, in order |
| `bench/` | Benchmark harness (written before any feature work) + the shared request pipeline |
| `scripts/` | Measurement runners. `measure_*` are exact and local; `colab_*` require a T4 |
| `edgerag/core/` | Model, layers, quantized linear, memory budget |
| `edgerag/cache/` | Naive and paged KV cache, block allocator, copy-on-write |
| `edgerag/sched/` | Continuous-batching scheduler, admission control |
| `edgerag/compress/` | Visual token pruning / merging |
| `edgerag/retrieval/` | Embedding, index, RAG pipeline |
| `edgerag/serve/` | FastAPI serving layer |
