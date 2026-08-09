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

_Phase 1 populates this._ Methodology is fixed and enforced in code, not by convention:

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
