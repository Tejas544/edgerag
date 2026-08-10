# EdgeRAG

**A complete multimodal RAG pipeline — image + text corpus → retrieval → VLM generation — running
entirely within a fixed 4 GB memory budget.**

No LangChain. No LlamaIndex. No `model.generate()`. The paged KV-cache allocator, the
continuous-batching scheduler, the visual-token compressor, and the quantized linear layers are
written here, not imported.

> **Status: Phase 0 of 8 (Aug 9, 2026).** Benchmark harness and budget enforcement are in place;
> the model pipeline is not. Numbers below are placeholders until Phase 1. See
> [`PLAN.md`](PLAN.md) for the phase schedule.

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

---

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev,serve]" --index-url https://download.pytorch.org/whl/cu124
pytest -q
python -m bench.bench --dry-run --md --allow-untrusted-device
```

---

## Design decisions

Every non-obvious choice, with the alternative it beat and the condition that would reverse it,
is logged in [`CONTEXT.md`](CONTEXT.md).

## Known limitations

Populated as they are discovered rather than at the end. [`BUGS.md`](BUGS.md) carries the bug log
and the predicted failure modes this design is built to avoid.

## Repo map

| Path | What |
|---|---|
| [`PLAN.md`](PLAN.md) | Phase schedule, gates, cut order |
| [`CONTEXT.md`](CONTEXT.md) | Decision log — every choice with its rejected alternative |
| [`BUGS.md`](BUGS.md) | Bug log + predicted failure modes |
| `bench/` | Benchmark harness (written before any feature work) |
| `edgerag/core/` | Model, layers, quantized linear, memory budget |
| `edgerag/cache/` | Naive and paged KV cache, block allocator, copy-on-write |
| `edgerag/sched/` | Continuous-batching scheduler, admission control |
| `edgerag/compress/` | Visual token pruning / merging |
| `edgerag/retrieval/` | Embedding, index, RAG pipeline |
| `edgerag/serve/` | FastAPI serving layer |
