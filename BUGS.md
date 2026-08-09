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

### Retrieval

**P-23 · Embedding normalization mismatch between index build and query time.** L2-normalized at
build, unnormalized at query → recall collapses and looks like a model quality problem.

**P-24 · Image preprocessing mismatch between index build and serve.** Different resize
interpolation or normalization constants in the two paths. Silent, and it degrades recall
by a few points — just enough to be mistaken for a modelling issue.
