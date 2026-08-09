# EdgeRAG — Execution Plan

**Status:** ACCEPTED — decisions D1–D9 confirmed, see `CONTEXT.md`.
**Window:** Aug 8 – Aug 18, 2026. Today is **Aug 9 = Day 2**.
**Source of truth for scope:** `01_EDGERAG.md`. This file is *how*, not *what*.

---

## Hardware tiers (D4) — the rule that everything else hangs off

| | Local · GTX 1650 · 4 GB · SM 7.5 | Colab · T4 · 16 GB · SM 7.5 |
|---|---|---|
| **Purpose** | Correctness, memory accounting, iteration | **Every published number** |
| **Models** | SmolVLM2-**256M** (fixture), **500M** (integration) | SmolVLM2-**2.2B** (headline) |
| **Tensor cores** | **No** — GTX 16-series ships without | Yes |
| **Triton** | No (Windows) | Yes |
| **2.2B fp16 fits?** | No (~4.4 GB > 4 GB) | Yes |

**Hard rule: no perf number measured locally may ever reach `results/`, a plot, the README, or the
CV bullet.** Without tensor cores, local timings are architecturally incomparable to a T4.
`bench.py` enforces this in code — it refuses to write a perf record on a non-T4 device unless
`--allow-untrusted-device` is passed, which stamps `"trusted": false` into the JSON.

Same compute capability on both tiers is the lucky part: no bf16 either side, and CUDA code that
runs locally runs on the T4 unchanged. Only the *speed* differs, never the *answer*.

---

## 0. Honest schedule assessment (read this first)

The spec budgets **45 hours across 11 days**. My estimate for the scope as written is
**50–58 hours.** We are over by roughly one full day, before any bug tax.

Two conflicts to resolve now rather than on Aug 15:

| Conflict | Reality | Resolution |
|---|---|---|
| `00_FOUNDATIONS.md` §5 puts the C++ ramp at 2 hrs/day on **Aug 15–18** — the same days as EdgeRAG Phases 4–6 | Aug 15–18 becomes a 7 hr/day double-load. Both degrade. | **RESOLVED (D9): C++ ramp slips to Aug 19–22, VecCore starts Aug 19.** EdgeRAG is the flagship; it does not get degraded to protect a downstream project's warm-up. |
| Day 1 (Aug 8) was supposed to be *both* 6 hrs of front-load reading *and* the start of "Days 1–2 baseline + harness" | Can't be both. | Treat Aug 8 as Day 0 (reading). Build starts today. Ship target is **Aug 18 EOD, with Aug 19 AM as declared slack.** |

**Cut order, decided now so it isn't decided under pressure at 1 a.m. on Aug 17:**

1. Demo GIF + OpenAI-compat response schema details
2. Triton INT4 kernel → fall back to reporting the honest throughput regression with a roofline explanation
3. ToMe (keep FastV only)
4. Continuous batching → keep static batching, report it as a known limitation

**Never cut:** `bench.py`, the equivalence test suite, the paged KV allocator, the memory
accounting table. Those four *are* the project.

---

## Phase gates

Every phase has a **gate**. A gate is a command that either passes or fails. You do not start
phase N+1 until phase N's gate is green and committed. This is the mechanism that prevents the
classic failure mode of this project: a subtly-wrong paged cache that produces plausible text and
poisons every number downstream.

---

## Phase 0 — Rails · Aug 9 AM · ✅ **DONE** (commit `9ac2044`)

Nothing model-related. This is scaffolding that everything else is measured by.

- `git init`, first commit, push to GitHub. Repo is public, pinned.
- Package layout (§ "Repo layout" below), `pyproject.toml`, `ruff`, `mypy`, `pytest`.
- **`bench/` harness — written before the model exists.** Warmup discard, `cuda.synchronize()`
  bracketing, p50/p95/p99, ≥5 trials with std-dev, `max_memory_allocated` reset between runs,
  incremental JSON append (Colab disconnect insurance), markdown table emitter.
  Prove it against a fake model that just `sleep`s — the harness must be trustworthy before it
  measures anything real.
- **`MemoryBudget` context manager** that raises if peak allocated exceeds the configured ceiling.
  Wired into tests from day one. The budget is enforced continuously, not audited at the end.
- `CONTEXT.md`, `BUGS.md`, `PLAN.md` committed.

**Gate: GREEN.** `pytest -q` → 29 passed · `ruff check` clean · `bench.bench --dry-run --md`
emits valid JSONL + markdown table · device gate verified to *refuse* on the GTX 1650 and to stamp
`trusted=false` under override · `MemoryBudget` verified to fire against real VRAM · committed.

**Not done — needs your action:** `git remote add origin … && git push`. The repo is local only.
`00_FOUNDATIONS.md` §6 makes commit history a hiring signal, and it only counts once it's pushed.

---

## Phase 1 — Baseline + frozen workload trace · Aug 9 PM – Aug 10 · ~6 hrs

This phase is reordered from the spec (retrieval was Day 10). Rationale: **the benchmark workload
must exist before the thing being benchmarked.**

- Load the VLM through `transformers`. Run `generate()` **once** — as the baseline and as the
  golden-output reference. (Allowed: it is the thing we beat, not the thing we ship.)
- Corpus: 300–500 document pages with figures/tables (DocVQA / InfographicVQA). Held-out QA set
  carved out now and never touched until Phase 6.
- **Stub retriever**: embed offline, flat cosine, dump `data/trace.jsonl` —
  `{query, retrieved_ids, image_paths, gold_answer}`. **This trace is frozen for the rest of the
  project.** Every benchmark from here replays it. Retrieval *quality* work stays in Phase 7;
  the *trace* has to be real now.
- **MEASUREMENT GATE — the differentiator's load-bearing assumption:**
  measure `visual_tokens / total_prefill_tokens` on the real trace with k=5 retrieved docs.
  Record the number in `CONTEXT.md`.
  - ≥ 50% → thesis holds, proceed as planned.
  - < 50% → the Phase 4 differentiator is undermined. Escalate immediately: enable image
    splitting / raise resolution / switch model (see D1). Do **not** discover this on Aug 15.
- Baseline numbers at batch 1 and batch 4: TTFT p50/p99, decode tok/s, peak memory.
  Record what `use_cache` was set to — see `BUGS.md` P-16.

**Gate:** `results/baseline.json` exists · visual-token ratio recorded in `CONTEXT.md` ·
first README benchmark table rendered.

---

## Phase 2 — Own the forward pass + naive KV cache · Aug 11 · ~5 hrs

- Our decoder forward: RMSNorm, RoPE, GQA attention, SwiGLU, the decode loop, sampling.
- HF retained for: weight loading, tokenizer/processor, and the **vision tower forward**.
  That boundary is deliberate and defended in `CONTEXT.md` D2.
- `core/linear.py` exposes a `QuantLinear` interface with an fp16 passthrough implementation.
  Quantization lands in Phase 6 as a config flag, not a refactor.
- Naive contiguous KV cache — this is the **reference implementation** that paged is validated
  against for the rest of the project. It never gets deleted.
- **`tests/test_equivalence.py`** — our logits vs HF logits, `allclose` at a tolerance justified
  in `CONTEXT.md`. Run against the **256M model** so the suite finishes in seconds and you
  actually run it every commit.

**Gate:** `pytest tests/test_equivalence.py` green · Plot 1 (tok/s vs seq len, cache on/off)
committed.

---

## Phase 3 — Paged KV cache · Aug 12–14 · ~13 hrs · **CENTREPIECE**

An interviewer will spend 15 minutes here. Budget accordingly.

**3a — Allocator, with no model in sight (~4 hrs).**
`BlockAllocator`: free list, refcounts, allocate/free, fragmentation accounting.
Unit-tested standalone with property tests: refcount conservation, no double-free, no leak across
randomized allocate/free/fork sequences. **This is pure data-structure code — test it as such,
before a GPU is involved.** Most of the pain in this phase is allocator bugs misdiagnosed as
attention bugs.

**3b — Block tables + paged attention (~5 hrs).**
Per-sequence logical→physical mapping. Paged attention via gather-into-scratch + SDPA (D3).
The masking of the partial final block is the highest-risk line of code in the project
(`BUGS.md` P-01).

**3c — Copy-on-write prefix sharing (~3 hrs).**
Hash the retrieved-context prefix; share blocks across queries hitting the same document;
copy on divergence. **This is the RAG-specific result** — measure memory saved when 20 queries
share one retrieved doc.

**3d — Preemption (~1 hr).** Recompute-on-evict, with the swap and reject alternatives written up
in `CONTEXT.md` so the "why that one" answer is on record.

**Gate — non-negotiable:** paged logits == naive logits == HF logits, swept across
seq_len ∈ {1, 15, 16, 17, 31, 32, 33, 127, 128, 129} × block_size ∈ {8, 16, 32}, with and without
prefix sharing. The boundary cases are the whole point of that sweep.

**Metrics:** max concurrent seqs before OOM (vs naive) · bytes saved by prefix sharing at k=20 ·
fragmentation % · **gather overhead as a fraction of decode time** (this is what lets you answer
"why didn't you write a fused kernel?" with a number instead of a shrug).

---

## Phase 4 — Visual token compression · Aug 15 · ~5 hrs · **DIFFERENTIATOR**

- FastV-style: rank visual tokens by attention received at layer K, prune the bottom p%.
- **Prune once at end-of-prefill**, not per-layer, so layers ≥K use a second shorter block table.
  Per-layer pruning would make block tables layer-dependent — real complexity, no extra credit.
- Sweep p ∈ {0, 25, 50, 62.5, 75, 87.5}. **The curve is the deliverable**, including where it
  falls off.
- Quality measured as ANLS/exact-match on the held-out set — not vibes.
- Fallback if FastV fights the paged cache: uniform-stride pruning as an honest baseline.

**Gate:** quality-vs-KV-bytes curve plotted with the cliff located and explained.

---

## Phase 5 — Continuous batching · Aug 16 · ~5 hrs

- Iteration-level scheduling; admit at every decode step.
- **Chunked prefill.** Not in the spec, and it matters: without it, one long retrieved-context
  prefill stalls every in-flight decode and your p99 TTFT is meaningless. It is also the direct
  answer to interview question #6.
- Admission control gated on the allocator's free-block watermark — the scheduler and the
  allocator are one system, not two.
- Poisson-arrival replay of the frozen trace.

**Gate:** Plot 2 (throughput **and** p99 TTFT vs concurrency, on the same axes — show the
tradeoff, do not bury it).

---

## Phase 6 — Quantization · Aug 17 · ~5 hrs

- INT8, then INT4 weight-only, group-wise g=128, symmetric, per-output-channel scales.
- Fill in `QuantLinear`. Optional Triton W4A16 GEMV (D7) — time-boxed to 2 hrs, hard stop.
- **The ablation that produces the actual finding:** quantize *LM only* vs *LM + vision tower* vs
  *vision tower only*. The VLM quality cliff lives in that table. This is the standout interview
  moment the spec is angling at, and it only appears if you run all three cells.

**Gate:** memory / tok-s / quality delta table across {fp16, int8, int4} × {LM, LM+ViT, ViT}.

---

## Phase 7 — Retrieval quality + serving · Aug 18 AM · ~4 hrs

- **Image embeddings reuse the VLM's own vision tower** (D6) — a separate SigLIP-SO400M would
  cost ~800 MB fp16 against a 4 GB budget, for a capability already resident.
- Text embeddings computed offline at index time; the text encoder is not resident at serve time.
- Hybrid scoring, recall@5 vs flat exhaustive.
- Index behind an interface with a flat implementation — **VecCore drops into that seam later.**
  That seam is the storyline connecting projects 01 and 02; build it now, cheaply.
- FastAPI: scheduler owns the GPU on its own thread, asyncio bridges via queues. Never run GPU
  work on the event loop (`BUGS.md` P-17).

**Gate:** `/v1/chat/completions` streams tokens · recall@5 within 2% of exhaustive.

---

## Phase 8 — Ship · Aug 18 PM · ~4 hrs

- Full ablation matrix — **generated by the harness from config files, not assembled by hand.**
- Memory accounting table summing under 4 GB, with the measurement definition stated explicitly
  (`max_memory_allocated`, CUDA context excluded and quantified separately).
- 3 plots · architecture diagram · README with design decisions and known limitations.
- `BUGS.md` final pass: the three hardest bugs written up properly. Per `00_FOUNDATIONS.md` §7,
  this is the question candidates fumble most.

---

## Repo layout

```
edgerag/
  core/     config, weights, layers, linear(QuantLinear), model, vision
  cache/    naive, allocator, block_table, paged, cow
  sched/    request, scheduler, admission
  compress/ fastv, tome
  retrieval/embed, index(flat; VecCore seam), pipeline
  serve/    app(FastAPI), engine(asyncio<->scheduler bridge)
bench/      bench, workload, metrics, plots
tests/      test_equivalence, test_allocator, test_cow, test_scheduler
configs/    model/*.yaml, ablation/*.yaml
data/       corpus/, trace.jsonl, heldout.jsonl
results/    *.json, plots/
```

Importable modules, thin notebook driver only. Per `00_FOUNDATIONS.md` §3: the repo must read
like software, not a lab book.

---

## Standing practices

- **Commit at every green gate**, minimum daily. Real messages.
- **`BUGS.md` gets an entry the moment a bug costs more than 20 minutes** — written while the
  diagnosis is fresh, not reconstructed in December.
- **`CONTEXT.md` gets an entry for every decision with a rejected alternative.** If there was no
  alternative, it wasn't a decision and doesn't belong there.
- Every benchmark run writes JSON incrementally. A Colab disconnect at hour 3 must not cost
  hours 1 and 2.
- `nvidia-smi` logged into every results JSON — you don't always get a T4, and silent hardware
  variation will otherwise show up as a phantom regression.
