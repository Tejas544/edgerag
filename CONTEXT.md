# EdgeRAG — Decision Log

Every entry is a decision **with a rejected alternative**. If there was no alternative, it wasn't
a decision and it doesn't belong here.

Purpose: in December, an interviewer asks "why X and not Y?" and the answer must be specific,
immediate, and not "it was easier." This file is where that answer is stored.

**Format:** `Dnn · Title · Status · Date` → Decision / Rejected / Why / Revisit if.
**Statuses:** `PROPOSED` · `ACCEPTED` · `SUPERSEDED by Dnn` · `REVERSED (see BUGS.md P-nn)`

---

## D1 · Base model · **ACCEPTED** · 2026-08-09

**Decision:** SmolVLM2-2.2B-Instruct as the headline model, **with image splitting enabled** on a
document-heavy corpus (DocVQA / InfographicVQA). SmolVLM2-256M as the *test fixture* model for the
equivalence suite.

**Rejected:**
- *Qwen2-VL-2B* — dynamic resolution genuinely produces more visual tokens (better for the Phase 4
  story), but **M-RoPE is 3D (temporal/height/width)**. Re-implementing the forward pass with M-RoPE
  and then making it work inside a paged block table is a multi-day detour on the critical path,
  for a differentiator we can get another way.
- *Moondream2* — `trust_remote_code`, non-standard config surface, thinner documentation. Weight
  introspection cost is unpredictable and this phase is on the critical path.
- *Text-only Qwen2.5-0.5B* — the documented fallback if VLM loading burns two days. Held in reserve.

**Why:** the schedule risk in Phase 2 (own forward pass) dominates. SmolVLM is a standard
SigLIP + SmolLM2 stack — RoPE is 1D, GQA is conventional, the config is legible. The two-model
split (2.2B headline / 256M fixture) is the important part: **an equivalence suite that takes
40 seconds gets run once a day; one that takes 4 seconds gets run every commit.** In a project
whose central risk is a silently-wrong cache, test cycle time is a first-class design concern.

**The catch — this is a live risk, not a settled point:** SmolVLM already applies pixel-shuffle
compression to visual tokens. If the resulting visual-token share is low, **the Phase 4
differentiator ("visual tokens dominate the KV cache") is false for our model** and the project
loses its novel contribution. Image splitting on high-resolution document pages should restore
it (~81 tokens × (N_subimages + 1), N=4–9 on doc pages, × 5 retrieved docs), but *this is an
assumption until measured.*

**Revisit if:** the Phase 1 measurement gate reports visual/total prefill tokens **< 50%**.
Escalation order: raise resolution → increase splitting → switch to Qwen2-VL-2B and pay the
M-RoPE cost.

> ### ✅ MEASURED 2026-08-09 — GATE PASSED
> `visual_tokens / total_prefill_tokens @ k=5` = **75.4% median** on the headline 2.2B
> (mean 75.5%, range **54.7%–88.3%** over 40 queries, 362-page corpus).
> **Even the worst-case query observed clears the 50% threshold.** Fixture agrees: 70.8% median.
> Raw data `data/token_ratio_gate.json`; reproduce with `python -m scripts.measure_token_ratio`.
>
> The thesis is not marginal on this model. It holds with room to spare, without needing any
> escalation on resolution or splitting. D1 stands; no model switch.

---

## D2 · The "from scratch" boundary · **ACCEPTED** · 2026-08-09

**Decision:** We own — decoder forward pass, attention, RoPE/RMSNorm/GQA/SwiGLU, KV cache (naive
and paged), the decode loop, sampling, scheduler, quantized linear, retrieval pipeline, serving.
We use HF for — weight loading, config introspection, tokenizer/image processor, and the
**vision tower forward**.

**Rejected:**
- *Reimplement everything including SigLIP and image preprocessing* — 2–3 days of the 11, spent on
  the least interesting surface. Image preprocessing bugs (resize/normalize/patch order) are
  silent and miserable to debug.
- *Use `model.generate()`* — explicitly forbidden by `01_EDGERAG.md` §2, and correctly so.

**Why:** the memory-management surface is the project. The vision tower is a one-shot prefill-time
compute cost with **no KV cache and no paging implications** — reimplementing it buys zero
insight into the thing being claimed. The interview answer is specific and holds up:
*"I wrote the attention path and the entire serving stack. I used HF for weight loading and the
vision tower forward — the tower is a one-shot prefill cost with no KV state, so it isn't part of
the memory-management surface this project is about."*

**Revisit if:** Phase 6 finds the vision tower is the quantization quality cliff and we need
layer-level control HF doesn't expose.

---

## D3 · Paged attention implementation · **ACCEPTED** · 2026-08-09

**Decision:** Gather physical blocks into a contiguous scratch buffer via `index_select`, then
call standard SDPA. **Measure the gather overhead as a fraction of decode time and publish it.**

**Rejected:**
- *Fused Triton paged-attention kernel* — the right long-term answer, but: Turing (SM 7.5) Triton
  support has degraded across recent releases, debugging a fused attention kernel on a machine you
  get disconnected from is a multi-day risk, and **it does not change the thesis.** The claim is a
  memory claim. Held as an optional Phase 3 stretch only if we are ahead.
- *FlashAttention / xformers paged path* — no dependable paged-KV path on Turing.

**Why:** the entire memory win — non-fragmented allocation, prefix sharing, higher concurrency —
is preserved exactly by the gather approach. Only the copy cost is added, and quantifying that
cost is *better* interview material than hiding it: *"my paged attention gathers then calls SDPA;
the gather is N% of decode time; a fused kernel removes that copy — here's the number I'd expect
to recover."*

**Revisit if:** measured gather overhead exceeds ~25% of decode time — at that point it distorts
every throughput comparison and has to be fixed rather than footnoted.

---

## D4 · Development environment · **ACCEPTED** · 2026-08-09

**Measured local hardware:** NVIDIA GeForce GTX 1650 · **4096 MiB VRAM** · **compute capability
7.5 (Turing, TU11x)** · driver 592.82 · Python 3.13.6 · Windows 11.

**Decision:** Two-tier split.
- **Local GTX 1650 — correctness tier.** Full CUDA dev loop against SmolVLM2-**256M** (fixture) and
  **500M** (integration). All allocator / block-table / CoW / scheduler / equivalence work.
- **Colab T4 — measurement tier.** SmolVLM2-2.2B, and **every number that appears in the README.**

**Rejected:**
- *Colab-only development* — 11 days of allocator iteration against a runtime that disconnects, no
  debugger, 30-second cell latency. Highest-friction possible loop for the highest-iteration-count
  part of the project.
- *Local-only, publish local numbers* — see the hard rule below. Not viable.

**Why this hardware is unusually well-suited, and where it isn't:**

| Property | Local GTX 1650 | Colab T4 | Consequence |
|---|---|---|---|
| Compute capability | 7.5 Turing | 7.5 Turing | **Same arch.** No bf16 on either. CUDA/Triton code that runs locally runs on T4. |
| Tensor cores | **None** (GTX 16-series ships without them) | Yes (~65 TFLOPS fp16) | fp16 matmul takes a different path locally. **Perf is not comparable.** |
| VRAM | **4 GB** | 16 GB | 2.2B fp16 (~4.4 GB) **does not fit locally.** 256M/500M do, comfortably. |

**The happy accident worth using in the README:** the project targets a ≤ 4 GB budget, and the
development GPU physically has exactly 4096 MiB. The constraint is not a config flag on a 16 GB
card — it is the machine. That is a materially stronger version of the same claim.

> ### HARD RULE — no local performance numbers, ever
> The GTX 1650 has no tensor cores. Any latency or throughput figure measured on it is
> architecturally incomparable to a T4 and **must never enter `results/`, a plot, the README, or
> the CV bullet.** Local runs are for *correctness and memory accounting* only.
> `bench.py` enforces this: it reads `torch.cuda.get_device_name()` and refuses to write a
> perf record on a non-T4 device unless `--allow-untrusted-device` is passed, which also stamps
> `"trusted": false` into the JSON. Manual discipline at 2 a.m. is not a control.

**Revisit if:** we need to profile a fused Triton kernel — see D7, that work is Colab-only anyway.

---

## D9 · Schedule: C++ ramp slips · **ACCEPTED** · 2026-08-09

**Decision:** The `00_FOUNDATIONS.md` §5 C++ ramp moves from Aug 15–18 to **Aug 19–22**. VecCore
starts Aug 19. EdgeRAG gets clean, undivided days for Phases 4–6.

**Rejected:**
- *Run both* — 7 hr/day across the three hardest EdgeRAG phases. The phase most likely to be
  degraded is Phase 4, which is the differentiator.
- *Cut EdgeRAG scope to protect the ramp* — inverts the priority. EdgeRAG is the flagship.

**Why:** `00_FOUNDATIONS.md` §1 — *"if you run out of time, cut scope, never depth."* Splitting
attention across two projects during Phase 3→4 cuts depth on both. The ramp is warm-up for a
project that hasn't started; the flagship is on the critical path.

**Revisit if:** EdgeRAG lands early. Then the ramp starts early and nothing was lost.

---

## D5 · Visual compression algorithm · **ACCEPTED (default, unchallenged)** · 2026-08-09

**Decision:** FastV-style attention-based pruning of visual tokens, applied **once at end of
prefill**, producing a second shorter block table for layers ≥ K.

**Rejected:**
- *ToMe (bipartite soft matching)* — merges tokens *inside the ViT*, so it reduces vision-encoder
  **compute**, not the LLM **KV cache**. Our thesis is a KV-cache memory thesis. Kept as the
  fallback if FastV fights the paged cache.
- *FastV pruned per-layer as published* — makes the resident token set layer-dependent, and
  therefore block tables layer-dependent. Real complexity, no additional result.

**Why:** FastV attacks the exact quantity the project is about. Pruning once at the prefill/decode
boundary keeps the block-table model simple (two tables, not L tables) while capturing essentially
all of the memory win, because decode length dominates total KV residency.

**Revisit if:** the quality cliff appears below 25% pruning — at which point the honest curve is
the result, per `01_EDGERAG.md` §10.

---

## D6 · Embedding model memory strategy · **ACCEPTED (default, unchallenged)** · 2026-08-09

**Decision:** Image embeddings are produced by **the VLM's own resident vision tower**. Text
embeddings are computed **offline at index build time**; the text encoder is not resident at serve
time.

**Rejected:**
- *Separate SigLIP-SO400M for image embedding* — ~400M params ≈ **800 MB in fp16**, i.e. 20% of
  a 4 GB budget, for a vision encoder already loaded in memory.
- *Resident sentence-transformer for text* — the query is one short string; encoding it is not
  worth keeping a model resident under a hard budget.

**Why:** in a project where *the constraint is the project*, spending 800 MB on a duplicate
capability is the single most obvious thing an interviewer would attack. Reusing the tower is
free, and "the retrieval embedder and the generator share a vision encoder" is a genuinely good
design line.

**Open question to settle in Phase 7:** query-side text embedding still needs *something* resident
or a CPU-side encoder. Measure the CPU latency before deciding; it may be small enough to ignore.

**Revisit if:** shared-tower embeddings measurably hurt recall@5 vs a dedicated embedder — then
the tradeoff becomes a *measured* memory-vs-recall curve, which is a better result than either
choice alone.

---

## D7 · INT4 kernel strategy · **ACCEPTED, amended** · 2026-08-09

**Decision:** Implement group-wise INT4 pack/unpack + a minimal Triton W4A16 GEMV, **time-boxed to
2 hours**. On timeout, ship pure-PyTorch dequant and report the throughput regression honestly
with the roofline explanation.

**AMENDMENT (2026-08-09, after hardware detection):** **Triton has no official Windows support.**
The kernel work therefore cannot be developed locally — it is **Colab-only**, which removes the
fast local iteration loop from the highest-risk part of Phase 6 and makes the 2-hour time-box
tighter than it looks. Two mitigations, both cheap:
1. The **pack/unpack, group-wise scale computation, and quantize→dequantize round-trip tests are
   pure PyTorch** and get built and tested locally in Phase 2's `QuantLinear` seam. Only the GEMV
   kernel itself is Colab-bound. This is the same "test the data structure without the GPU"
   principle as D4.
2. Because P-20 (wrong scale axis) is caught locally by the round-trip test, the Colab session is
   spent on kernel performance rather than on debugging quantization correctness.

The GTX 1650's lack of tensor cores is irrelevant here — a W4A16 GEMV at batch 1 is
bandwidth-bound, not MMA-bound — but the Windows/Triton gap is decisive.

**Rejected:**
- *`bitsandbytes` NF4* — violates the spirit of §2 for a component that is arguably core.
- *Skipping the kernel and only reporting memory* — leaves the most-asked question
  ("INT4 is fewer bytes over HBM, so why isn't it faster?") answered with a shrug.

**Why — set expectations now:** on T4 there are **no INT4 tensor cores**, and naive PyTorch
dequant-then-matmul is *slower than fp16* because the dequant overhead dominates at batch-1 decode.
The memory win is real and immediate; the **speed** win requires fusing dequant into the GEMV so
you actually move 4× fewer bytes across HBM. Without that kernel the honest report is
"4× memory reduction, N% throughput regression, here is why, and here is what the fused kernel
recovers" — which is still a defensible senior answer, and is the documented cut line.

**Revisit if:** Triton fails on Turing at Phase 6. Do not spend the evening on it — take the
fallback, write it up, move on. QuantKit is where this gets done properly.

---

## D8 · Retrieval built early, tuned late · **ACCEPTED (default, unchallenged)** · 2026-08-09

**Decision:** A stub retriever and a **frozen `trace.jsonl`** are built in Phase 1 (Aug 9–10).
Retrieval *quality* work (hybrid scoring, recall@k) stays in Phase 7 per the original spec.

**Rejected:**
- *Spec's original ordering (retrieval on Day 10)* — every benchmark from Phase 3 onward would run
  against synthetic prompts, and real retrieved contexts have a different length distribution.
  Discovering that on Day 10 invalidates a week of numbers with no time to re-run them.

**Why:** the workload must exist before the thing being benchmarked. Freezing the trace also means
every ablation cell across 8 days is comparing against an identical input — which is
`00_FOUNDATIONS.md` §4 rule 5 ("fix everything you're not measuring") applied at the project level
rather than the function level.

**Revisit if:** the trace turns out to be unrepresentative (e.g. all retrieved contexts land in a
narrow length band, hiding fragmentation behaviour). Then regenerate once, early, and re-baseline
deliberately.

---

## D10 · Checkpoint variants, and the MHA finding · **ACCEPTED** · 2026-08-09

**Decision:** Headline **`HuggingFaceTB/SmolVLM2-2.2B-Instruct`**, fixture
**`HuggingFaceTB/SmolVLM2-256M-Video-Instruct`**. Both are `model_type: smolvlm`.

**Rejected:** `HuggingFaceTB/SmolVLM-Instruct` (2.2B) and the 256M/500M SmolVLM-v1 checkpoints —
these are `model_type: idefics3`. Mixing families across tiers would mean the forward pass is
*tested* on one architecture and *shipped* on another. Matching `model_type` across tiers is worth
more than any difference between the two families.

### Finding 1 — the headline model is MHA, not GQA. This sizes the entire project.

Config probe, 2026-08-09:

| | 2.2B (headline) | 500M | 256M (fixture) |
|---|---|---|---|
| layers | 24 | 32 | 30 |
| query heads | 32 | 15 | 9 |
| **kv heads** | **32** | 5 | 3 |
| attention | **MHA — no GQA saving** | GQA 3:1 | GQA 3:1 |
| KV bytes/token fp16 | **196,608 (192 KiB)** | 40,960 | 23,040 |

`2 × 24 layers × 32 kv-heads × 64 head-dim × 2 bytes = 196,608 B/token`.

- One sequence at seq_len 2048 → **384 MiB**
- **Batch 8 at seq_len 2048 → exactly 3.00 GiB, against a 4 GiB total budget.**

The 2.2B checkpoint spends **8.5× more KV per token than the 256M fixture** despite having fewer
layers, purely because it has no GQA. Encoded and tested in `edgerag/core/spec.py` /
`tests/test_spec.py`.

**Why this is good news:** the project's premise — that KV cache management is the binding
constraint — is not a contrivance on this model, it is arithmetic. Paging, prefix sharing, and
visual-token pruning all have more headroom to win than they would on a GQA model.

**Consequence for testing:** the fixture exercises `n_rep=3` and the headline exercises `n_rep=1`.
**Neither tier covers the other.** MHA is the degenerate case of GQA, so one code path serves both,
but the `n_rep=1` path must be explicitly tested locally or it ships untested (`BUGS.md` P-08).

**Consequence for the CV bullet:** *"my 2.2B VLM uses full MHA, so its KV cache is ~4× a
comparable GQA model's — which is exactly why paging pays for itself here"* is a specific,
checkable claim.

### Finding 2 — visual token counts, and what the Phase 1 gate is really testing

Pixel shuffle divides tower patches by `scale_factor²`:

| | patches/side | scale_factor | **tokens per sub-image** |
|---|---|---|---|
| 2.2B | 384/14 = 27 → 729 | 3 | **81** |
| 256M | 512/16 = 32 → 1024 | 4 | **64** |

At 81 tokens, a *single un-split image* would make the D5 differentiator false. The thesis
survives only through **image splitting**, where a 2×2 split costs 5 sub-images (4 tiles + 1
global view) = **405 tokens/image**, and k=5 retrieval = **2025 visual tokens ≈ 380 MiB of KV**.

So the Phase 1 gate is not really measuring the model — **it is measuring whether our processor
configuration splits aggressively enough.** That reframes the escalation path: tune splitting and
resolution *first*, and only consider the D1 model switch if splitting cannot get us there.

**Revisit if:** the measured ratio lands below 50% even at maximum splitting.

---

## D11 · What the gate measurement forces · **ACCEPTED** · 2026-08-09

The Phase 1 gate did more than clear a threshold — it produced the number the rest of the project
is sized by.

**One k=5 RAG query on the headline model costs 1,267 MiB of KV cache. 987 MiB of that is
visual.** (median 6,758 prefill tokens × 192 KiB/token; 5,265 of those tokens are image tokens.)

Three consequences, all of which change downstream work:

**1. Two concurrent queries do not fit. At all.**
Weights at INT4 (~1.1 GiB) + two queries (2.5 GiB) + activations already exceeds 4 GiB. A naive
contiguous KV cache tops out at **one** in-flight request. The paged allocator is therefore not an
optimisation on this project — it is the difference between a demo and a server. That is a far
stronger framing for the README than "4–8× more concurrent sequences."

**2. Visual token pruning moves from "differentiator" to "load-bearing".**
78% of KV bytes are visual. A 50% visual prune reclaims ~494 MiB per query — more than the entire
INT4 weight saving. The Phase 4 sweep should therefore be treated as a *memory* result first and a
quality result second, and the plot's y-axis should be MiB reclaimed, not just % tokens removed.

**3. Prefix sharing has a concrete, measured target.**
95 of 362 corpus pages carry multiple questions (max 11). Those are real shared prefixes, not a
synthetic scenario. At 11 questions against one page, CoW should save roughly 10 × the per-page
visual KV cost.

**Deferred to Phase 3, now that the numbers exist:** block size. At 192 KiB/token, a 16-token
block is 3 MiB and a 4 GiB pool holds only ~1,300 blocks — so block-table overhead is negligible
and internal fragmentation is expensive. This argues for **smaller** blocks than the default 16.
Decide with a measured sweep, not by analogy to vLLM's defaults, which were tuned on GQA models
with ~8× cheaper tokens.

---

## D12 · The vision tower's activation peak is a budget line item · **ACCEPTED** · 2026-08-10

**Found by:** smoke-testing the Colab runner locally on the 256M fixture before spending scarce
free-tier T4 quota. The run was not intended to produce a finding; the peak-memory number did.

**Measurement (256M fixture, k=5, batch 1, GTX 1650):** peak allocated **2.696 GiB**.

That number does not decompose the way the plan assumed:

| component | size |
|---|---|
| model weights | 0.49 GiB |
| KV cache (5,683 tokens × 22.5 KiB) | 0.12 GiB |
| pixel values (65 sub-images) | 0.10 GiB |
| **unaccounted — vision tower activations** | **~2.0 GiB** |

A k=5 document-RAG prompt splits into ~65 sub-images, and the tower processes **all of them in one
forward pass**. The transient activation peak dominates everything else — on the *smallest* model
in the tiering, where weights and KV are both trivial.

**Why this matters more than it looks:** `01_EDGERAG.md` §11 asks for a memory table summing
weights + KV + vision + embeddings + index under 4 GB. That framing treats the vision encoder as a
*weights* cost. It is not. On this workload it is an **activation** cost, it is transient, and it
sets the peak — and peak is what `max_memory_allocated` measures and what the budget is defined
against (D3/P3). A budget assembled from weights and KV alone would balance on paper and OOM in
practice.

Extrapolating to the 2.2B headline model: 65 sub-images × 729 patches = ~47k tokens through a
27-layer, 1152-wide tower in one pass. The MLP intermediate alone is ~437 MiB transient per layer
boundary. Against a 4 GiB budget already holding ~1.1 GiB of INT4 weights and ~1.27 GiB of KV,
**the vision tower is a plausible cause of the budget failing** — a risk not on the original list.

**Decision: chunk the vision tower forward.** Process sub-images in groups (start at 8) rather
than all 65 at once, bounding peak activation at the cost of some GPU parallelism during prefill.

**Rejected:**
- *Leave it unchunked and buy headroom elsewhere* — cedes 2 GiB of a 4 GiB budget to a transient,
  and the peak scales with `k`, so any increase in retrieval depth reopens it.
- *Gradient checkpointing on the tower* — solves a training problem. There is no backward pass
  here; there is nothing to trade compute against.
- *Downscale images* — directly attacks the visual-token share the Phase 1 gate just confirmed at
  75.4%, and would weaken the Phase 4 result to fix a memory problem that chunking solves for free.

**Why it is free:** each sub-image passes through the ViT independently — there is no
cross-sub-image attention in the tower. Chunking is therefore *exactly* equivalent, not an
approximation, and the equivalence is testable (identical embeddings, any chunk size).

**Implementation:** Phase 2, alongside the forward pass. Chunk size becomes a config knob and gets
a memory-vs-prefill-latency sweep in Phase 8.

**Interview value:** the KV cache gets all the attention in this problem space. "On a
document-RAG workload the vision encoder's transient activation peak was competitive with the KV
cache, so I chunked the tower forward and measured the tradeoff" is a specific, non-obvious
finding that comes only from having measured early.

**Revisit if:** measured chunking cost in prefill latency exceeds ~15%, at which point the chunk
size needs tuning rather than the default.

---

## Pending — decide before the phase that needs it

| # | Question | Needed by |
|---|---|---|
| P1 | Preemption policy: recompute vs swap vs reject. Leaning **recompute-on-evict** (no CPU↔GPU transfer, simplest correctness story, and prefill is compute-bound so recompute is cheap on a T4 with spare FLOPs). | Phase 3d |
| P2 | Equivalence tolerance: what `atol/rtol` and why. fp16 softmax accumulation drift grows with sequence length — a tolerance chosen at seq 32 will fail at seq 2048. Likely: upcast softmax to fp32, set tolerance per-length. | Phase 2 |
| P3 | Memory-budget definition: does the 4 GB include the ~300–600 MB CUDA context? **Recommend: exclude it, state it explicitly, and report it as a separate line.** Silently excluding it is the kind of thing that unravels an otherwise good conversation. | Phase 8 |
| P4 | Quality metric for the pruning sweep: ANLS vs exact-match vs LLM-judge. Leaning **ANLS** (standard for DocVQA, no extra model resident). | Phase 4 |
| P5 | **Batched VLM requests pad the sub-image dimension to the batch maximum.** Measured 2026-08-10: batching a 57-sub-image request with a 61 one yields `pixel_values (2, 61, ...)` — four phantom images pushed through 12–27 tower layers for nothing. Document pages vary widely in aspect ratio, so this waste is the norm, not an edge case. Options: (a) group requests by sub-image count in the scheduler, (b) run the tower **per request, unbatched**, and batch only the decoder — attractive because the tower is prefill-only and D12 already chunks it. Leaning (b). | Phase 5 |
