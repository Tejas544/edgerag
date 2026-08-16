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

> ### ⚠️ MEASURED 2026-08-11 on a Tesla T4 — **the threshold is blown, three times over**
>
> | seq | block | gather | attention | **gather share** |
> |---:|---:|---:|---:|---:|
> | 2048 | 16 | 0.412 ms | 0.188 ms | 68.7% |
> | 4096 | 16 | 0.797 ms | 0.238 ms | 77.0% |
> | 6800 | 16 | 1.214 ms | 0.357 ms | **77.3%** |
> | 8192 | 16 | 1.427 ms | 0.412 ms | **77.6%** |
>
> At the workload's median length the gather costs **3.4× the attention it feeds**. Block size is
> almost irrelevant (52–78% across 8/16/32/64), and contiguous attention costs the same as paged
> (0.362 vs 0.357 ms), so the gather is **pure overhead, not amortised**. Scaled to a full decode
> step: 24 layers × 1.214 ms = **29 ms of gather per token**, against 8.6 ms of attention.
>
> The local GTX 1650 estimate was ~60%; the T4 is worse, exactly as predicted — tensor cores make
> attention faster, so the copy occupies a larger share. Predicting the direction correctly is
> mild consolation for the size of the number.
>
> **D3's reasoning was sound and its conclusion is wrong.** "Only a copy is added" is true; the
> unexamined assumption was that the copy is *small*. It is the dominant cost of the decode
> attention path, because the gather reads the whole KV and writes a copy of it, then attention
> reads that copy — roughly tripling the memory traffic of a step that D14 already established is
> bandwidth-bound.
>
> **Order of response, cheapest first:**
> 1. **P6 — the redundant second copy.** `_gather_pool` copies twice: once for `pool[block_ids]`,
>    again for `.transpose(0,1).contiguous()`, because the pool is token-major and attention wants
>    head-major. A head-major pool makes the reshape a free view. Roughly halves the gather for a
>    layout change. **Do this first.**
> 2. **Re-measure.** If the share is still far above 25%, the fused Triton kernel stops being a
>    stretch goal and becomes the honest fix — with the caveat that Triton is Colab-only (D7).
> 3. **Report it either way.** "The gather is 77% of the attention path and here is what I did
>    about it" is a far better answer than a design note claiming the cost is negligible.

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

> **CORRECTED 2026-08-10 — the reasoning above is wrong, see D15.** "One wasted block is 3 MiB" is
> true in absolute terms and irrelevant in relative ones. Requests on this workload are ~6,625
> tokens, so a wasted 64-token block is **0.95%** of one sequence. Measured internal fragmentation
> across the whole trace is **0.0%–0.5%** for every block size from 1 to 64. Block size is
> effectively free to choose here, and the argument for small blocks evaporates. I reasoned from
> the absolute size of a block without dividing by the size of a request.

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
cross-sub-image attention in the tower. Chunking is therefore mathematically equivalent, not an
approximation.

> **Verified and refined 2026-08-10** (`tests/test_equivalence.py`). **"Exactly equivalent" above
> was overstated.** Chunking agrees with unchunked encoding to **~5e-06**, not bit-exactly: a
> chunk holding a *single* image takes a different GEMM path from a batched one, so reduction
> order and fp32 rounding differ. Any input whose sub-image count is not a multiple of the chunk
> size ends with exactly such a chunk. Same effect as `CACHE_ATOL`.
>
> **I got this wrong twice before getting it right.** A first run on a large image appeared
> bit-exact for chunk sizes ≥3, and I wrote that down as a property — it was a coincidence of an
> input that happened to divide evenly. A smaller image produced a trailing batch-1 chunk and the
> claim collapsed. Reading agreement on one input as a guarantee is precisely the mistake the
> tolerance bands exist to prevent, and it is the second overclaim of bit-exactness in this phase.
>
> The claim also went **unverified from Phase 2 until Phase 4**, and the function it describes was
> broken the whole time (`BUGS.md` B-04). An untested assertion in a decision log is a liability,
> not a record.

**Implementation:** Phase 2, alongside the forward pass. Chunk size becomes a config knob and gets
a memory-vs-prefill-latency sweep in Phase 8.

**Interview value:** the KV cache gets all the attention in this problem space. "On a
document-RAG workload the vision encoder's transient activation peak was competitive with the KV
cache, so I chunked the tower forward and measured the tradeoff" is a specific, non-obvious
finding that comes only from having measured early.

**Revisit if:** measured chunking cost in prefill latency exceeds ~15%, at which point the chunk
size needs tuning rather than the default.

---

## D13 · Correctness gates run in fp32, not fp16 · **ACCEPTED** · 2026-08-10

**Measured 2026-08-10**, our decoder vs HuggingFace, 256M fixture, seq 64, eager attention both
sides:

| precision | max abs Δ logits | mean abs Δ | greedy argmax |
|---|---|---|---|
| **fp32 / CPU** | **0.000e+00** | **0.000e+00** | identical |
| fp16 / CUDA | 1.75e-01 | 1.97e-02 | identical |

**In fp32 the output is bit-identical** — hidden states too, not just logits. The implementation is
not "close to" HF's, it *is* HF's arithmetic. The fp16 divergence is accumulation order: same
operations, different cuBLAS kernel selection driven by tensor layout, compounded over 30 layers.

**Decision:** every correctness gate — Phase 2's, and Phase 3's paged-cache gate — runs in
**fp32**, where the expected difference is *zero* and the tolerance is 1e-6. fp16 is tested
separately and only for the property that actually matters functionally: **identical greedy
token ids**.

**Rejected:** *a single fp16 test with a tolerance loose enough to pass.* That was the original
design and it failed at seq_len=2 on 4 elements out of 98,560. The obvious fix — widen `atol` from
2e-2 to 3e-2 — is the trap: a tolerance wide enough to absorb fp16 noise is wide enough to hide a
real bug, and there is no principled place to stop widening it.

**Why this matters more in Phase 3 than here.** `BUGS.md` P-07 predicted an evening lost to "the
equivalence test fails at long context and it is *not* a bug, it is fp16." Running the gate in
fp32 **eliminates that entire class of confusion**: any nonzero difference is a bug, full stop.
When the paged allocator disagrees with the naive cache by 1e-3, there is no judgement call about
whether 1e-3 is acceptable — it is not, and the search starts immediately. That is worth
considerably more than the wall-clock cost of running the gate on CPU.

**Cost:** fp32 CPU runs on the 256M fixture, which is why D4's two-tier split exists. Sequence
lengths in the fp32 sweep stay modest to keep the suite in the seconds range.

**Interview value:** "how do you know your implementation is correct?" answered with *bit-identical
in fp32* is a materially stronger answer than *within tolerance*.

---

## D14 · What the T4 baseline says · **MEASURED** · 2026-08-10

Tesla T4 (14.6 GiB), SmolVLM2-2.2B (4.18 GiB fp16, 2,246,784,880 params), frozen trace
`94b148a0b9f5006e`, HF `generate()` with `use_cache=True`.

| batch | TTFT p50 | tok/s per seq | **tok/s aggregate** | peak alloc |
|---|---|---|---|---|
| 1 | 3,730 ms | 26.0 | 26.0 | 5.76 GiB |
| 2 | 11,071 ms | 15.9 | 31.8 | 7.90 GiB |
| 4 | 25,006 ms | 8.6 | 34.4 | 12.41 GiB |

### Finding 1 — the baseline does not fit the target budget for even one request

**5.76 GiB for a single k=5 RAG query, against a 4 GiB ceiling.** The naive fp16 pipeline is over
budget at batch 1, before any concurrency. This is a better statement of the problem than
"reduce memory 4×": the starting point is not a working system that is merely inefficient, it is a
system that *does not run* on the target device.

The D11 prediction was 5.7 GiB and the measurement is 5.76. The arithmetic in
`edgerag/core/spec.py` holds at scale.

### Finding 2 — batching barely helps, and MHA is why

Aggregate throughput goes 26.0 → 31.8 → 34.4: **1.32× for 4× the batch.** For a GQA model this
would be near-linear.

The cause is D10: the 2.2B checkpoint is MHA, so every sequence carries 192 KiB of KV per token.
Decode is bandwidth-bound on KV reads, and batch *b* reads *b* times as much. Adding sequences adds
proportional bandwidth demand, so throughput saturates almost immediately.

**This changes the Phase 5 expectation and should be said before anyone asks.** `01_EDGERAG.md` §6
targets 5–11× decode throughput from continuous batching. On this model that ceiling is set by KV
bandwidth, not by scheduling, and no scheduler recovers it. **The lever that does work is cutting
KV bytes** — visual-token pruning (Phase 4) and INT4 (Phase 6) attack the actual bottleneck;
batching does not. Reinforces D11.

### Finding 3 — activation cost per sequence grows with batch, confirming D12

Subtracting weights and predicted KV:

| batch | non-weight, non-KV | per sequence |
|---|---|---|
| 1 | 0.31 GiB | 0.31 |
| 2 | 1.18 GiB | 0.59 |
| 4 | 3.16 GiB | 0.79 |

Per-sequence activation *rises* with batch because left-padding pads every request to the batch
maximum, and the vision tower's sub-image count follows. D12's chunking addresses the tower half;
the padding half is a scheduling concern for Phase 5 (group by similar length).

### Finding 4 — TTFT degrades superlinearly

3,730 → 11,071 → 25,006 ms is 1× → 2.97× → **6.7×** for 1× → 2× → 4× batch. Same padding cause.
A 3.7-second TTFT at batch 1 is already a poor experience; 25 seconds at batch 4 is unusable. This
is the number chunked prefill (Phase 5) has to fix, and it makes the p99-TTFT-vs-throughput plot
the honest centrepiece `01_EDGERAG.md` §5 asks for.

### Finding 5 — max concurrency is exactly 4 ✅ *refined 2026-08-10*

`oom_probe` with linear refinement: batch 1 (5.76 GiB), 2 (7.90), 4 (12.41) succeed; batch 8 OOMs;
**batch 5 also OOMs**. `max_ok: 4, oom_at: 5, refined: true`.

**A naive HF pipeline sustains exactly 4 concurrent k=5 RAG requests on a 16 GB T4** — and cannot
run one within the project's own 4 GiB budget. This is the denominator for every concurrency claim.

The earlier asterisk is resolved. The first probe only doubled (1, 2, 4, 8) and so proved
*"≥4, <8"*; publishing that 4 as a maximum would have been right by luck. It is now measured:
batch 5 fails, so 4 is the true ceiling. The refinement cost eight minutes and converted a lucky
guess into a defensible number.

### Finding 5b — cross-session latency variance is 12–14%; memory variance is zero

Two cells were measured twice across separate Colab sessions, which is free variance data:

| cell | metric | run 1 | run 2 | spread |
|---|---|---|---|---|
| `hf_generate_b4` | TTFT p50 | 25,006 ms | 28,377 ms | **11.9%** |
| `hf_generate_b4` | decode tok/s | 8.64 | 8.27 | 4.3% |
| `hf_generate_b4` | peak allocated | 12.413 GiB | 12.413 GiB | **0.00%** |
| `hf_generate_nocache_b1` | TTFT p50 | 4,125 ms | 4,773 ms | **13.6%** |
| `hf_generate_nocache_b1` | decode tok/s | 0.26 | 0.22 | **13.8%** |
| `hf_generate_nocache_b1` | peak allocated | 5.520 GiB | 5.520 GiB | **0.00%** |

This is `BUGS.md` P-15 quantified, and it sets two rules for the rest of the project:

1. **Any latency claim below ~15% is inside the cross-session noise floor.** A "1.1× faster"
   result measured in two different Colab sessions is indistinguishable from thermal and clock
   variation. Latency comparisons must be made **within a single session**, against a baseline
   re-measured in that same session.
2. **Memory numbers are bit-for-bit reproducible.** Allocation is deterministic; peak allocated
   matched to the byte across sessions and hardware states.

**This retroactively validates the project's framing.** The headline claims are memory claims —
5.76 GiB → 1.86 GiB, 443 MiB reclaimed per request, exactly 4 concurrent requests — and those are
the numbers that survive scrutiny. Had the thesis rested on a modest throughput multiple, 14% of
it would have been weather. Worth saying unprompted: *"I measured my own measurement noise before
trusting any of it."*

### Finding 6 — the KV cache is worth 100× here, not the 2–3× the plan predicted

| batch 1 | decode | peak alloc |
|---|---|---|
| `use_cache=True` | **25.95 tok/s** | 5.76 GiB |
| `use_cache=False` | **0.26 tok/s** | 5.52 GiB |

**100×**, against `01_EDGERAG.md` §5's "expect 2–3×". The plan's estimate assumes short prompts.
The mechanism explains the gap exactly: without a cache every decode step re-attends the entire
6,758-token retrieved prefix, so *one decode step costs one full prefill* — 1/3.73 s = 0.27 tok/s,
which is what was measured. The cache converts O(prefix) per-token work into O(1).

**This is a RAG-specific result, and it is the honest framing:** the longer the retrieved context,
the more the KV cache is worth. A chatbot with a 200-token prompt would see something near the
plan's 2–3×. Quote the ratio *with* the mechanism, never alone.

The memory column is the more interesting half. **Disabling the cache saves only 0.24 GiB (4%)
while costing 100× throughput.** Naively the cache should cost its full 1.27 GiB, but with
`use_cache=False` every step still runs a full-sequence forward whose activations dominate the
peak. So at batch 1 the KV cache is not a memory-vs-speed tradeoff at all — it is nearly free.
Paging earns its keep at *concurrency*, where the KV terms add up and the activation peak does
not, which is precisely the regime Phase 3 targets.

**Harness change made in response:** the markdown table now reports per-sequence *and* aggregate
tok/s. Reporting only per-sequence makes batching look like a regression, which is a self-inflicted
wound in an interview.

---

## D15 · Where paging actually wins on this workload — and where it does not · **MEASURED** · 2026-08-10

Exact allocator arithmetic over all 650 trace requests, with per-document token costs measured by
the real processor (median **1,418** tokens/page, range 651–2,191; median request **6,625**
tokens). Memory only, so it runs on the local tier under D4.

| block size | paged | internal frag | + prefix sharing | saving | + canonical order | saving |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 790.70 GiB | 0.0% | 724.67 | 8.4% | **673.63** | **14.8%** |
| 16 | 791.58 GiB | 0.1% | 725.80 | 8.3% | **674.80** | **14.8%** |
| 64 | 794.27 GiB | 0.5% | 729.27 | 8.2% | **678.36** | **14.6%** |

### Finding 1 — block size does not matter here, and I argued otherwise

Internal fragmentation is **0.0%–0.5% across the entire range 1–64**. At ~6,625 tokens per
request, even a 64-token block wastes under 1%. D11's claim that fragmentation argues for small
blocks was wrong: it reasoned from a block's *absolute* size (3 MiB) without dividing by a
request's size (1.21 GiB). **Decision: block_size = 16.** Not because it is optimal — nothing is —
but because fragmentation is free and 16 minimises block-table length.

### Finding 2 — the 4–8× concurrency target is not reachable on this workload

`01_EDGERAG.md` §6 targets **4–8×** max concurrent sequences from paging. Measured, within a
2.5 GiB KV budget: **naive 1 sequence → paged 2**.

The reason is structural, not fixable. Paging's headline win is not over-reserving: a static cache
reserves `max_position_embeddings` (8,192 tokens = 1.50 GiB) per sequence while paged reserves
`ceil(6625/16)*16` (= 1.22 GiB). But **RAG requests already sit at 81% of the maximum context**, so
right-sizing can only ever recover 8192/6625 = **1.24×**. The 4–8× figure assumes short sequences
against a large reservation — a chatbot workload, not a k=5 document-RAG one.

**And the comparison is against a strawman anyway.** HuggingFace's `DynamicCache` — the actual
measured baseline (D14) — already grows on demand and does *not* pre-reserve `max_seq_len`. So
"paging beats static over-reservation" describes a baseline nobody runs.

### What paging genuinely buys here, stated honestly

1. **Prefix sharing: 8.3%**, and this is the real lever (see Finding 3).
2. **No per-token reallocation.** `DynamicCache` concatenates the whole cache each decode step —
   O(prefix) memory traffic per token. Paged writes one slot. Untested as a *latency* claim; needs
   the T4.
3. **Block-granularity admission and eviction**, which is what makes the Phase 5 scheduler
   possible at all. A contiguous cache cannot hand back a finished sequence's memory mid-batch.

**The README must lead with (2) and (3), not with a concurrency multiple.** Claiming 4–8× here
would not survive one question about sequence length versus max context.

### Finding 3 — ordering retrieved documents canonically nearly doubles the sharing win

Prompts currently place the gold document first, then differing neighbours, so two queries about
the same page share exactly one document. Sorting the retrieved set by document id instead of
relevance makes the common prefix longer:

**8.3% → 14.8% saving, a 78% improvement, for a one-line change in prompt assembly.**

This is a genuine systems lever and I have not seen it in the RAG literature: *retrieved-document
ordering is a prefix-cache-hit-rate decision, not only a relevance decision.* It trades against
position effects in answer quality (documents early in the prompt get attended to differently),
which makes it a measurable quality-vs-memory tradeoff rather than a free win.

**Deferred to Phase 4**, where the quality harness exists to price the tradeoff. Changing prompt
assembly now would bump `PROMPT_FORMAT_VERSION` and invalidate the frozen trace and the T4
baseline — for a change that must be quality-tested before adoption regardless.

---

## D16 · Preemption policy: swap, not recompute · **ACCEPTED** · 2026-08-10 · resolves P1

**Decision:** **swap-to-pinned-host-memory**, victims chosen **newest-first**. `RECOMPUTE` and
`REJECT` are implemented behind the same interface so the choice can be measured rather than
argued.

| policy | cost to resume | why not |
|---|---|---|
| **SWAP** | ~1.27 GiB over PCIe 3.0 x16 ≈ **100 ms** | — |
| RECOMPUTE | one full prefill = **3,730 ms** (D14) | **37× worse** |
| REJECT | request dies | honest under sustained overload, not a first resort |

**This reverses the plan's lean, and the reason is a measurement.** `01_EDGERAG.md` §5 and my own
P1 favoured recompute on the argument that *"prefill is compute-bound, so a T4 has spare FLOPs and
recompute is cheap."* D14 kills it: TTFT is **3.73 s at batch 1**, because a k=5 RAG prompt is
~6,758 tokens of which 75% are visual and must traverse the vision tower again. Recompute here
costs *seconds*, and preempting a request would push p99 far past the point where any scheduler
throughput gain is worth having.

**Where the plan's instinct is right:** for short prompts, a host round-trip costs more than
re-running prefill, and vLLM's recompute default is correct. The deciding variable is prefix
length, and this workload sits at the far end of it. That is why the policy is a parameter with a
measured justification rather than a constant — and it is a good answer to §7 question 4.

**Victim selection is newest-first.** Oldest-first would discard the most accumulated work and
livelock long requests: each preemption resets whichever request has waited longest. Newest-first
penalises arrivals during a spike, which is the better trade.

**Two constraints carried in from Phase 3a/3c:**
- Copy-on-write can itself raise `OutOfBlocksError`, so admission control must reserve CoW headroom
  or deadlock exactly when prefix sharing is helping most.
- `swap_in` is an admission decision, not a guaranteed operation — the pool may have been taken.
  The Phase 5 scheduler has to handle a failed restore.

**Revisit if:** measured swap latency exceeds ~500 ms, at which point rejection beats both.

---

## D17 · Visual pruning: the memory half, measured · **MEASURED** · 2026-08-10

FastV at layer 2 on the headline model. **22 of 24 layers (92%) store the reduced set**, which is
why the MiB saved far exceeds the fraction of tokens dropped. Mean request: 6,804 tokens, 5,159
visual (75.8%).

| keep | visual kept | KV MiB | reclaimed | saving |
|---:|---:|---:|---:|---:|
| 1.000 | 5,159 | 1,276 | 0 | 0.0% |
| 0.750 | 3,869 | 1,054 | 222 | 17.4% |
| **0.500** | 2,579 | 832 | **443** | **34.8%** |
| 0.250 | 1,289 | 611 | 665 | 52.1% |
| 0.125 | 644 | 500 | 776 | 60.8% |

**Pruning half the visual tokens reclaims 443 MiB per request — more than the entire INT4 weight
saving (≈3.1 GiB → 1.05 GiB is a one-off; this is per request and scales with concurrency).**
D11 predicted ~494 MiB at 50%; the measured 443 is close, the gap being the text tokens and the
two layers below the cut.

**Against the 4 GiB ceiling**, INT4 weights (1.05 GiB) plus one request:

| | KV | total | verdict |
|---|---|---|---|
| baseline fp16, no pruning (measured, D14) | — | **5.76 GiB** | does not fit |
| INT4 + no pruning | 1.28 GiB | 2.29 GiB | fits |
| INT4 + 50% pruning | 0.83 GiB | **1.86 GiB** | fits, with room for a second request |

That is the project's thesis in one table: from *does not run on the target device* to *fits with
headroom*, and the two levers are independent.

### This is half a result and must not ship alone

The quality axis needs the 2.2B and therefore a T4 (`scripts/colab_pruning_quality.py`). A memory
curve published without its quality curve is the flattering half. The plan's target is 50–75%
of visual tokens removed at ≥95% quality retention; whether 0.5 clears that is **unmeasured**.

Two controls are built into the quality script:
- **`uniform` stride at every ratio.** If attention-based selection does not beat evenly-spaced
  selection, the honest finding is "visual tokens are redundant on this workload", not "FastV
  works". That distinction is the difference between a result and a citation.
- **`keep_ratio=1.0` proven bit-identical to no compressor** (`atol=0, rtol=0` in
  `tests/test_fastv.py`), so the baseline row is the real baseline.

### Scoring mode is a genuine fork, not a detail

`visual_token_scores` supports two modes and they answer different questions:
- **`last_row`** (default, FastV's choice): score by what the *final* prompt token attends to.
  Every decode query resembles that one, so it is the best available predictor of what generation
  will need. Pruning is a bet about the future and this bets with the closest observation.
- **`mean`**: average over all attending queries, normalised by how many queries could see each
  key. More signal and less sensitive to one odd final token, but it measures what the *prompt*
  attended to, which is not the same question.

The normalisation in `mean` matters: without it, token *j* is scored across all *S* queries when
only *S−j* could see it, which systematically under-scores late tokens — and in a RAG prompt the
late tokens are the most recently retrieved document. Both modes go in the ablation.

---

## D18 · Chunked prefill trades numerical drift for TTFT, and the drift is measurable · **MEASURED** · 2026-08-11

Chunked prefill is the fix for D14's 25-second TTFT at batch 4: a 6,800-token prompt is processed
in slices across iterations instead of monopolising the GPU while every decoding request stalls
(`BUGS.md` P-18).

It is mathematically exact and numerically is not, and the pattern is clean. 40-token prompt,
fp32, vs a single full prefill:

| chunks | max abs Δ logits |
|---:|---:|
| **1** | **0.000e+00** |
| 2–3 | 4.3e-05 |
| 6 | 1.6e-04 |
| 40 | 9.1e-04 |

Single-chunk is **bit-identical**, as is paged-vs-naive single-shot — so the logic is right. Every
additional chunk boundary changes a GEMM shape, and the rounding compounds through 30 layers.
Greedy tokens are identical at every chunk size tested.

**Two consequences:**

1. **A flat tolerance is wrong for this comparison.** `tests/test_batched.py` scales it with chunk
   count (`5e-5 × n_chunks`) and asserts single-chunk exactness separately. A flat band is either
   too loose to catch a real bug at 2 chunks or too tight to pass at 40.
2. **Chunk size is a three-way tradeoff, not a two-way one.** Smaller chunks give finer
   scheduling granularity and lower p99 TTFT, but cost more numerical drift *and* more Python-side
   iteration overhead. The default of 512 sits far from the noisy end — 6,800 tokens is ~14
   chunks, well inside the band where drift is ~1e-4 on fp32 logits and invisible to greedy
   decoding.

**Worth stating unprompted:** "chunking the prefill changes the logits in the last decimal places,
and here is the measured curve" is a stronger answer than asserting it is free.

---

## Pending — decide before the phase that needs it

| # | Question | Needed by |
|---|---|---|
| P1 | Preemption policy: recompute vs swap vs reject. Leaning **recompute-on-evict** (no CPU↔GPU transfer, simplest correctness story). **Amended 2026-08-10 — D14 removes one of the reasons.** "Prefill is compute-bound so recompute is cheap" assumed spare FLOPs; the measured TTFT is **3.7 s at batch 1 and 25 s at batch 4**, so recomputing a preempted sequence is *seconds*, not microseconds, and would wreck p99. Swap-to-host now looks better than it did: 1.27 GiB over PCIe is ~100 ms, an order of magnitude cheaper than recompute. Decide with a measurement, not the vLLM-paper default. **Second constraint, found in Phase 3a:** copy-on-write itself can raise `OutOfBlocksError` — writing to a shared block needs a free block at the moment the pool is fullest. Admission control must reserve CoW headroom, or the system deadlocks exactly when prefix sharing is helping most. | Phase 3d |
| P2 | Equivalence tolerance: what `atol/rtol` and why. fp16 softmax accumulation drift grows with sequence length — a tolerance chosen at seq 32 will fail at seq 2048. Likely: upcast softmax to fp32, set tolerance per-length. | Phase 2 |
| P3 | Memory-budget definition: does the 4 GB include the ~300–600 MB CUDA context? **Recommend: exclude it, state it explicitly, and report it as a separate line.** Silently excluding it is the kind of thing that unravels an otherwise good conversation. | Phase 8 |
| P4 | Quality metric for the pruning sweep: ANLS vs exact-match vs LLM-judge. Leaning **ANLS** (standard for DocVQA, no extra model resident). | Phase 4 |
| ~~P6~~ | ✅ **DONE 2026-08-11.** Pool switched to head-major `(kv_heads, blocks, slots, dim)`, so `pool[:, block_ids]` reshapes to `(kv_heads, tokens, dim)` as a **free view** — no transpose, no second `contiguous()`. Local (untrusted, directional) gather at seq 2048 fell **1.037 ms → 0.698 ms, a 33% reduction**; the fraction of the attention path dropped 61.7% → 51.7%. Less than the 50% hoped, because the remaining copy is the gather itself and is irreducible without fusion. 91 cache tests pass unchanged, which is the real check: the layout touches the write path, CoW block copies, preemption swap, and the batched pool, and every one of those had to be re-indexed. **Needs a T4 re-measure before the number is published** — and if the share is still far above 25% there, the fused Triton kernel becomes the honest fix rather than a stretch goal. |
| ~~P6-orig~~ | *(original entry, kept for the reasoning)* **`_gather_pool` copies twice, and it should copy once.** Writing the D3 measurement exposed it: `pool[block_ids]` produces one copy, then `.transpose(0,1).contiguous()` produces a second, because the pool is token-major `(blocks, slots, kv_heads, dim)` while attention wants head-major. Storing the pool as `(kv_heads, blocks, slots, dim)` makes the gather `pool[:, block_ids]`, whose reshape to `(kv_heads, seq, dim)` is a **free view** — no transpose, one copy. Untested-but-directional local numbers put gather at ~60% of the paged attention path, well above D3's 25% revisit threshold, which is unsurprising in hindsight: the gather reads the KV *and writes a copy*, then attention reads that copy, so it roughly doubles decode's memory traffic. **Do not act before the T4 number** — the local card has no tensor cores, so its attention is slow and the true fraction there is likely *higher*, not lower. If confirmed, this layout change is the cheap fix and a fused kernel is the expensive one. | Phase 5 |
| ~~P5~~ | **WITHDRAWN 2026-08-10 — the claim was wrong.** I reported that batching a 57-sub-image request with a 61 one wastes tower compute on four phantom images, inferred from `pixel_values` having shape `(2, 61, ...)`. Reading `SmolVLMModel.get_image_features` shows it drops all-zero padding images (`real_images_inds`) *before* the tower runs. The padded tensor costs memory; it costs no compute. Lesson: a tensor shape is not a code path. Residual real issue is much smaller and tracked as `BUGS.md` P-26 (an all-black page is indistinguishable from padding). | — |
