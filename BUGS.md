# EdgeRAG — Bug Log

Per `00_FOUNDATIONS.md` §7: *"Name the single hardest bug and how you diagnosed it"* is the
question candidates fumble most and interviewers weight most heavily. You will not remember these
in December. **Write the entry while the diagnosis is fresh.**

**Rule:** any bug costing more than 20 minutes gets an entry, written the same day.

---

## Entry format

```
### B-nn · <one-line symptom> · <date> · <phase>
**Symptom:**    What you observed. The observation, not the cause.
**Wrong theory:** What you believed first, and why it was plausible. ← keep this, it's the
                  most interesting part of the story in an interview.
**Root cause:**  What was actually wrong.
**Diagnosis:**   How you found it. The *method*, not the answer.
**Fix:**         Commit SHA.
**Prevention:**  The test that now exists so it can't come back.
**Cost:**        Hours.
```

---

## Confirmed bugs

### B-01 · OCR text silently empty for every InfographicVQA page · 2026-08-09 · Phase 1

**Symptom:** corpus build completed with exit 0, no exception, no warning — and the summary line
read `with OCR text: 0 (mean 0 chars)` across all 250 InfographicVQA pages. Every document in the
corpus was image-only, so hybrid text+image retrieval had nothing to retrieve on.

**Wrong theory:** that InfographicVQA's `ocr` field was empty or absent for the validation split,
and text would have to come from OCR-ing the images ourselves (a Tesseract dependency and an
afternoon). The field is 208 KB per row — it was never empty.

**Root cause:** two independent container assumptions, both wrong, and neither one raising.

1. **The field is not JSON.** It is a **Python list repr** — literally starts `['{` and ends
   `}']` — so `json.loads` fails at character 1. The payload is the single string element inside.
2. **Blocks are grouped by type, not flat.** Top-level keys are `PAGE`, `LINE`, `WORD`, each with
   its own list. There is no flat `Blocks` array. The code read `payload["PAGE"]`, which holds
   exactly one page-geometry block carrying no text.

Either mistake alone yields `""`. The parser caught its own exceptions and degraded to `""` by
design, so both failures were invisible.

**Diagnosis:** the corpus builder prints a coverage stat (`with OCR text: N`) as part of its
summary. That line — not an exception, not a test — is what surfaced it. Confirming the shape then
took one probe script: `repr` the first three characters, try `json.loads`, try
`ast.literal_eval`, and print `Counter(BlockType)` per container key.

**Fix:** `ast.literal_eval` → unwrap the single-element list → `json.loads` the inner string →
read `payload["LINE"]`, with a fallback to a flat `Blocks` array. `edgerag/retrieval/corpus.py`.

**Prevention:** two layers, because a silent-empty failure needs more than a unit test.
- `tests/test_corpus.py` builds a fixture through `str([json.dumps(payload)])` — the *shipped*
  container, not a convenient one — and asserts LINE text survives, WORD is dropped, and order is
  preserved.
- `scripts/build_corpus.py` now **fails the build** if OCR coverage is under 80%. Absence of an
  exception proves nothing about a function whose failure mode is a valid return value.

**Cost:** ~25 min, and one wasted 5-minute corpus rebuild.

**The transferable lesson:** a parser that swallows exceptions and returns a falsy default cannot
be validated by "it ran." It needs a *coverage* assertion at the call site. This applies directly
to the Phase 3 allocator, where `free()` on an already-free block is the same shape of bug —
plausible state, no exception, wrong answer.

### B-04 · Chunked vision encoding was never called by any test, and was broken · 2026-08-10 · Phase 4

**Symptom:** `IndexError: The shape of the mask [8, 147456] at index 1 does not match the shape of
the indexed tensor [8, 729] at index 1`, raised on a Colab T4 **after a 9 GB model download**, on
the first real invocation of a function written two phases earlier.

**Root cause:** `encode_images_chunked` passed a **pixel-resolution** mask (384×384 = 147,456)
where the vision tower expects a **patch-resolution** one (27×27 = 729). HuggingFace's own
`get_image_features` builds the pixel mask and then *unfolds* it by `patch_size` to get the patch
grid; I built the pixel mask and passed it straight through, skipping the unfold.

**Why it survived 267 tests:** **nothing in the suite ever called it.** The local tests build
synthetic token ids and never pass a real image through the vision path, so the entire function
was dead code as far as CI was concerned. It was written in Phase 2 to implement `CONTEXT.md` D12
and first executed in Phase 4, on the user's GPU quota.

**Fix:** mirror HF's `unfold(dim=1, size=p, step=p).unfold(dim=2, size=p, step=p).sum(...) > 0`.
Using the same construction rather than `H // p` also keeps the two in step when the image
dimension is not an exact multiple of the patch size — 384/14 is 27.4, and both forms happen to
give 27 here, but only one of them is guaranteed to keep matching.

**Prevention — three tests that should have existed since Phase 2:**
- our chunked encoding equals HF's `get_image_features` **bit-exactly**;
- chunk size does not change the result (this is D12's whole claim, previously unverified);
- the number of image features equals the number of `<image>` token slots, which is the cheap
  invariant that also catches `P-11` and `P-26`.

**It also corrected a claim — twice, which is the instructive part.** D12 said chunking was
*"exactly equivalent"*. The first test run, on a large image, appeared to confirm bit-exactness
for chunk sizes ≥3, and I wrote that down as a property. A smaller image then produced a trailing
**batch-1 chunk**, which takes a different GEMM path, and the claim collapsed at ~5e-06.

The correct statement is: chunking is *mathematically* equivalent and agrees to ~5e-06. Any input
whose sub-image count is not a multiple of the chunk size ends with a partial chunk, so
bit-exactness was never a property — it was a coincidence of one input that happened to divide
evenly. See also `P-28`, which is why the same test then failed only under system load.

**Cost:** one wasted T4 session and a 9 GB download — the user's scarce free-tier quota, which is
the expensive part.

**The transferable lesson, and it is the sharpest one so far:** a decision log entry asserting
*"chunking is exactly equivalent"* is worthless if nothing executes the code it describes. Test
count is not coverage. Before this, three separate documents (D12, the module docstring, the
commit message) all confidently described the behaviour of a function that had never once run.
**Any code path whose only caller lives in a script rather than a test is untested**, however
carefully it is documented.

---

### B-03 · Forked sequences overwrote each other's KV — refcounts were right, data was not · 2026-08-10 · Phase 3c

**Symptom:** none visible in the test suite. 53 paged-cache tests passed, including every fork and
copy-on-write test. The defect was found by deliberately probing the data path that the
bookkeeping tests did not touch.

**Wrong theory:** that Phase 3b was complete because `fork`, `unshare`, and `writable_block_for`
were implemented and tested. They were — and `PagedKVCache._write` never called any of them. It
indexed `self.table.blocks` directly, so a forked sequence appended straight into blocks its
sibling was still reading.

**Root cause:** two of them, and the second only appears once the first is fixed.

1. **The write path bypassed copy-on-write entirely.** The CoW machinery existed at the
   `BlockTable` layer; the tensor writes went around it.
2. **`unshare` repoints the mapping but does not copy the block's contents.** Fixing (1) alone
   gives the forked sequence a private block full of **zeros** — a valid-looking mapping to data
   that was never written. The answer degrades; nothing crashes.

**Why the existing tests missed it:** they asserted on refcounts. Refcounting was correct
throughout. `test_fork_shares_blocks_without_copying` and `test_writing_to_a_forked_block_splits_it`
both passed against a cache that corrupted real KV, because neither looked at a tensor.

**Diagnosis, and the part worth remembering:** the first reproduction attempt *failed*. Forking,
then appending only from the child, left the parent intact — because `gather` slices to the
parent's own `num_tokens`, so the child's write landed in slack the parent never reads. It looked
like CoW was working. The giveaway was `copy_on_writes == 0`: the parent survived by luck, not by
correctness. **Corruption needs both siblings to append** — the second writer overwrites the
first's tokens, and the first then reads the second's data. Asymmetric bugs like this are why
"I could not reproduce it" is weak evidence.

**Fix:** `PagedKVCache._unshare_write_range` walks every logical block the write will touch, calls
`unshare`, and on a split copies that block's KV across **all layers at once**. Done at layer 0 —
splitting per layer would leave layers disagreeing about which block holds the sequence.
`edgerag/cache/paged.py`.

**Prevention:** `tests/test_cow.py` asserts on **tensor contents**, not refcounts: both siblings
append, and each must see its own tokens and none of the other's. Plus the exact-multiple case,
where no partial tail exists and CoW must *not* fire.

**Cost:** ~30 min, all of it in finding a reproduction that actually failed.

**The transferable lesson:** correct bookkeeping is not correct behaviour. Every allocator test in
this project passes against a cache that serves the wrong data, because refcounts and tensors are
different subsystems. Any test suite for a memory manager needs at least one assertion on the
bytes.

---

### B-02 · `rope_theta` silently defaulting to 10000 instead of 100000/130000 · 2026-08-10 · Phase 2

**Symptom:** none yet — caught by probing the config before writing RoPE, not by debugging output.
Had it shipped, the symptom would have been a model that loads, runs, and emits fluent, confident,
wrong text.

**Wrong theory (the one this narrowly avoided):** that `rope_theta` behaves like every other field
and can be read with `getattr(text_config, "rope_theta", 10000.0)`. It returns `None`, so the
default applies.

**Root cause:** in transformers v5 the value moved into a nested dict —
`text_config.rope_parameters["rope_theta"]`. The actual values are **100000** for SmolVLM2-256M and
**130000** for SmolVLM2-2.2B. The library default of 10000 is therefore wrong by 10–13×, *and the
two checkpoints disagree with each other*, so hardcoding a constant would have been wrong too.

**Why it would have been expensive:** a wrong RoPE base does not crash, does not NaN, and does not
fail a shape check. It changes the frequency at which positions rotate, so the model attends to
subtly wrong relative positions. The output is grammatical. The natural diagnosis is "my RoPE
implementation is wrong" (`BUGS.md` P-09) and the search goes to `rotate_half`, the `unsqueeze_dim`,
the position offset — anywhere except the config reader, which "obviously works" because every
other field it reads is correct.

**Diagnosis:** printing every architectural constant needed for the reimplementation *before*
writing any of it, and reading the values rather than scanning for `None`.

**Fix:** `_read_rope_theta` in `edgerag/core/spec.py` reads `rope_parameters` first, falls back to
the direct attribute, and **raises rather than defaulting** if neither exists.

**Prevention:** no silent default. Pinned values for both tiers in `tests/test_spec.py`, so a
checkpoint changing shape under us fails a test. Same treatment applied to `rms_norm_eps` — Llama's
library default is 1e-6 but every SmolVLM checkpoint ships **1e-5**, which is the identical trap one
constant over.

**Cost:** ~15 min, entirely in prevention. Zero debugging.

**The transferable lesson — this is the second instance of one pattern.** L-01 (`pad_token_id`) and
B-02 (`rope_theta`) are the same bug: *the composite config and the nested config disagree, and the
obvious read returns the wrong one.* For a value that will be silently wrong rather than loudly
absent, `getattr(cfg, name, default)` is not a safe idiom — the default is the failure.

---

## Defused landmines

Found by inspection before they fired. Recorded because "why did you check that?" is a better
interview answer than "it crashed and I fixed it," and because the mitigation is load-bearing.

### L-01 · `config.pad_token_id` is out of range for the vocabulary · 2026-08-09 · Phase 1

**Found:** probing checkpoint configs before writing the batching code.

**The defect, in the shipped checkpoints:**

| field | SmolVLM2-256M | SmolVLM2-2.2B | valid? |
|---|---|---|---|
| `config.pad_token_id` | **128002** | **128002** | ❌ `vocab_size` is 49280 |
| `config.text_config.pad_token_id` | 2 | 2 | ✅ |
| `tokenizer.pad_token_id` | 2 (`<\|im_end\|>`) | 2 | ✅ |

128002 is a Llama-3 vocabulary leftover on the *composite* VLM config. `transformers` prints a
warning and continues.

**Why it is dangerous rather than annoying:** `config.pad_token_id` is the obvious field to read,
and it is wrong. Padding a batch with token id 128002 is an out-of-bounds index into a
49280-row embedding table. On CUDA that surfaces as a **device-side assert**, which
(a) is asynchronous, so the traceback points at an unrelated later line, (b) poisons the CUDA
context so every subsequent operation in the process fails, and (c) on Windows/Turing gives an
especially unhelpful message. It would not fire until Phase 5 introduces padded batches, by which
point the scheduler would be the natural suspect and the config the last place anyone looks.

**Mitigation:** never read `pad_token_id` from the composite config. `ModelSpec` takes it from
`text_config`, and a test asserts `pad_token_id < vocab_size` for every checkpoint in the tiering.
Related: `BUGS.md` P-10 and P-11, which are the other two ways padding silently corrupts a batch.

---

## Predicted failure modes

Written **before** building, from prior experience with this class of system. Two purposes:
when a symptom below shows up, the diagnosis takes minutes instead of hours; and any that
*doesn't* occur is evidence the design avoided it.

Move an entry up to **Confirmed** with a full write-up when it actually bites.

### Paged KV cache — where the real pain is

**P-01 · Partial last block attends to uninitialized memory.** ⚠️ *highest-risk line in the project*
The final block of a sequence is only partly filled; the unfilled slots contain whatever the
previous tenant left. If the attention mask doesn't exclude them, you attend to garbage.
*Symptom:* no crash, no NaN — output is fluent and slightly wrong, and logits drift only when
`seq_len % block_size != 0`. *Detection:* the boundary sweep in the Phase 3 gate, which is
specifically designed to catch this. This bug ships to production in real systems.

**P-02 · Off-by-one when `seq_len % block_size == 0`.** `ceil_div` boundary — either a spare empty
block is allocated forever (silent leak) or the next token has nowhere to go (`IndexError`, at
token 16 or 32, reproducibly).

**P-03 · RoPE positions taken from the physical slot instead of the logical position.**
*Symptom:* correct output until prefix sharing is switched on, then coherent-but-wrong output.
The physical block index is meaningless to RoPE; only the logical sequence position is. Appears
*only* with CoW active, which is why the equivalence sweep must run in both modes.

**P-04 · Refcount leak on the abnormal-exit path.** Blocks are freed when a request completes
normally; the cancel / error / timeout paths forget. *Symptom:* everything passes, then the server
OOMs after ~an hour of load. Invisible to unit tests, visible only to the Poisson load test.
*Prevention:* assert pool conservation (allocated + free == total) after every load test.

**P-05 · Double-free on preempt-then-readmit.** Preemption frees blocks; the readmission path
frees them again. Two live sequences then write to the same physical block. *Symptom:*
nondeterministic corruption of *other* requests' output — the worst debugging experience available.
*Prevention:* property test over randomized allocate/free/preempt/readmit sequences.

**P-06 · Block table rebuilt as a Python list → tensor every decode step.** A hidden host-to-device
sync in the hot loop. *Symptom:* **paged is slower than naive**, and you spend a day blaming the
gather (D3) when the real cost is `torch.tensor(list)` per step. *Prevention:* preallocate the
block-table tensor on device and mutate in place; profile before concluding anything about the
gather.

**P-07 · fp16 softmax accumulation drift breaks the equivalence test at long context.**
A tolerance calibrated at seq 32 fails at seq 2048 — and it is *not* a bug, it's fp16.
Wasting hours hunting a nonexistent cache bug here is the classic trap. *Prevention:* upcast
softmax to fp32; set tolerance as a function of sequence length; decide this in Phase 2 (`CONTEXT.md` P2).

### Forward pass

**P-08 · GQA head expansion via `repeat` instead of `repeat_interleave`.** Silently wrong
key/value→query head mapping. Output is grammatical and wrong. Catch it in Phase 2 against HF
logits — which is exactly why the equivalence test comes before the optimization work.

**P-09 · RoPE position offset of 0 during decode instead of `past_len`.** First generated token is
fine, everything after degrades. Very common when the prefill and decode paths are written on
different days.

**P-10 · `-inf` mask in fp16 → NaN.** A fully-masked row (padding) softmaxes to NaN and the NaN
propagates through the whole batch. Use `torch.finfo(dtype).min`, not `float('-inf')`.
*Symptom:* one padded request poisons every concurrent request's output.

**P-11 · Image-token placeholder expansion mismatch.** The `<image>` placeholder must expand to
*exactly* the number of visual embeddings the tower produced. Off by one and every subsequent
embedding is shifted. *Symptom:* total garbage — which is actually good news, it's loud.
Gets worse in Phase 4, where pruning *changes* the count and the expansion logic must follow.

**P-12 · Missing `torch.inference_mode()`.** Autograd graph retained → roughly 2× memory. Under a
4 GB budget this reads as "the model doesn't fit" rather than "I forgot a decorator."

**P-25 · Budget assembled from weights + KV, blown by transient activations.** ⚠️ *measured, see
`CONTEXT.md` D12* — the 256M fixture peaks at 2.7 GiB on a k=5 prompt, of which ~2.0 GiB is
vision-tower activation for 65 sub-images processed in one pass. *Symptom:* the accounting table
sums under budget and the pipeline OOMs anyway; worse, it OOMs *only* on high-`k` or high-page-count
requests, so it looks like a data-dependent bug rather than a design one. *Prevention:* the budget
is defined on `max_memory_allocated` (a peak), not on a sum of resident components, and
`MemoryBudget` wraps the whole prefill rather than just model construction.

### Benchmarking — these destroy credibility, not just numbers

**P-13 · No `torch.cuda.synchronize()` → timing kernel launches, not execution.**
`00_FOUNDATIONS.md` §4 rule 2. Produces impossibly good numbers. An interviewer spots this in
ten seconds and discounts everything else on the CV.

**P-14 · `max_memory_allocated` not reset between runs.** Every run reports the peak of the *worst*
prior run. Memory ablations become meaningless and, worse, monotonic-looking.

**P-15 · Colab thermal/clock variation read as a regression.** Same code, 15% slower, an evening
lost. *Prevention:* log `nvidia-smi` into every results JSON; re-run the baseline in the same
session as the comparison, never across sessions.

**P-16 · Baseline accidentally run with `use_cache=False`.** Produces a fake 5–10× win that
collapses under one question. *Prevention:* assert the baseline config in `bench.py` and print it
into the results JSON. **The baseline gets more scrutiny than the optimization — a wrong baseline
is worse than no benchmark.**

### Scheduler & serving

**P-17 · GPU work executed on the asyncio event loop.** Every request's p99 explodes; it looks
like a model problem and is a threading problem. *Prevention:* scheduler owns the GPU on a
dedicated thread; asyncio touches it only through queues.

**P-18 · Head-of-line blocking from a long prefill.** One 4000-token retrieved context stalls all
in-flight decodes. p99 TTFT looks catastrophic and inexplicable. *Fix:* chunked prefill (Phase 5).
This is also the direct answer to interview question #6.

**P-19 · Starvation of long requests under continuous batching.** Short requests keep getting
admitted; a long one never finishes. Needs aging or a fairness term in admission.

### Quantization

**P-20 · Group-wise scales computed along the wrong axis.** Catastrophic quality loss that reads as
"INT4 just doesn't work on VLMs" — a wrong and embarrassing conclusion to put in a README.
*Prevention:* unit-test quantize→dequantize round-trip error per layer *before* running any
end-to-end quality eval.

**P-21 · Quantizing layers that must stay fp16** (embeddings, LayerNorm/RMSNorm, lm_head, the
vision tower's patch embedding). Instant cliff. Skip-list them explicitly and *state the list* —
the skip list is itself an interview answer.

**P-22 · fp16 overflow on outlier channels during dequant.** Activation outliers are the entire
reason SmoothQuant/AWQ exist. Expect it; if you see it, you've independently rediscovered a real
result — say so.

**P-27 · The test suite is slow and holds four model copies.** Three modules each keep a
module-scoped model fixture (`test_equivalence` holds fp32-CPU *and* fp16-CUDA; `test_paged` and
`test_fastv` one each). One full-suite run died with a fatal interpreter error and retries passed.
Separately, the vision tests pushed the suite past ten minutes until they were cut to one small
image and two chunk sizes. *If it recurs:* consolidate into one session-scoped bundle rather than
raising the machine's limits. A suite slow enough to skip is a suite that stops catching things —
the same argument D4 makes for using the 256M fixture at all.

**P-28 · fp32 CPU results are not bit-reproducible across machine load.** ⚠️ *diagnosed 2026-08-10*
Five vision tests passed when their module ran alone and failed in the full suite, at a 1e-6
tolerance, by ~5e-06. The tempting reading was memory pressure (P-27); it was not.

CPU GEMM reduction order depends on how many threads the kernel is dispatched across, and that
varies with system load. A comparison whose two sides use the **same** tensor shapes is unaffected
— both run under whatever configuration is current — which is why the Phase 2 prefill tests hold
at 1e-6 indefinitely. A comparison across **different** shapes (chunked vs unchunked encoding,
many small GEMMs vs one large one) picks different kernels on each side, and the gap between them
moves with load.

*The rule this yields:* `EXACT_ATOL` (1e-6) is valid only when both sides of a comparison have
identical tensor shapes. Anything crossing a shape boundary — cached decode vs full prefill,
chunked vs unchunked, paged vs naive — belongs at `CACHE_ATOL` (1e-4). That was already the rule
for the KV cache; the vision path simply had not been classified yet.

*Why it presented as flakiness:* a tolerance that is marginally too tight passes on an idle
machine and fails on a busy one, which looks like nondeterminism and invites a retry rather than a
diagnosis.

**P-26 · An all-black page is indistinguishable from a padding sub-image.** `get_image_features`
identifies padding as "every pixel is 0.0" and drops those images before the tower. A genuinely
black scanned page — not rare in a document corpus — is silently dropped, so its `<image>` tokens
receive embeddings belonging to a *different* image and every subsequent image in the sample
shifts by one. *Symptom:* one document in the corpus produces confidently wrong answers while
everything else works. *Note:* normalization runs before this check, so "all zeros" means all
values equal to the dataset mean, not literally black — which makes it rarer but harder to reason
about. *Detection:* assert that the number of image embeddings returned equals the number of
`<image>` token groups in `input_ids`, which is a cheap invariant and also catches P-11.

### Retrieval

**P-23 · Embedding normalization mismatch between index build and query time.** L2-normalized at
build, unnormalized at query → recall collapses and looks like a model quality problem.

**P-24 · Image preprocessing mismatch between index build and serve.** Different resize
interpolation or normalization constants in the two paths. Silent, and it degrades recall
by a few points — just enough to be mistaken for a modelling issue.
