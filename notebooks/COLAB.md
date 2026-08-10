# Running the baseline on Colab

Paste these cells into a fresh notebook. Total GPU time: **~25–35 minutes**, most of it the
weight download.

## Before you start

**Switch to a GPU runtime.** Runtime → Change runtime type → **T4 GPU** → Save. A CPU runtime
shows only "System RAM" and "Disk" in the resources panel; a T4 session adds a **GPU RAM** bar and
the backend reads `Python 3 Google Compute Engine backend (GPU)`.

Free tier grants T4 access without compute units — "zero compute units" is the *paid* balance and
does not block you. What it does mean: no guarantee a T4 is available, ~90-minute idle
disconnects, and a dynamic quota that can lock you out after heavy use. **The runner assumes it
will be killed mid-run** and resumes from a manifest, so a disconnect costs one measurement rather
than the session.

---

### Cell 1 — check what you actually got

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

If this says anything other than **Tesla T4**, stop. The runner will refuse anyway
(`CONTEXT.md` D4), because a number measured on a K80 or an L4 is not comparable to the rest of
the results and must not enter `results/`.

### Cell 2 — mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/edgerag
```

Results are fsync'd here after every measurement. The runtime is disposable; these files are not.

### Cell 3 — clone and install

```python
!git clone https://github.com/Tejas544/edgerag.git /content/edgerag
%cd /content/edgerag
!pip install -q -e . 2>&1 | tail -3
```

Colab ships a CUDA torch already, so `-e .` picks it up rather than pulling a fresh 2.5 GB wheel.

### Cell 4 — rebuild the corpus (CPU only, ~6 min)

```python
!python -m scripts.build_corpus --infographic 250 --docvqa 400
!python -m scripts.build_trace --k 5
```

Corpus images are ~1 GB and not in git, so they are rebuilt from the same deterministic stream.
**The trace fingerprint printed here must read `94b148a0b9f5006e`.** If it differs, the workload
is not the one measured locally and no comparison is valid — stop and report it rather than
continuing.

This cell spends no GPU time. Doing it before the weights load means a disconnect during setup
costs no quota.

### Cell 5 — the baseline

```python
!python -m scripts.colab_baseline --drive /content/drive/MyDrive/edgerag
```

Measures, in order:

| cell | what |
|---|---|
| `hf_generate_b1/b2/b4` | HF `generate()`, `use_cache=True`, replaying the frozen trace |
| `hf_generate_nocache_b1` | `use_cache=False`, 16 tokens — the Phase 2 "why the cache matters" comparison |
| `oom_probe` | largest batch a naive pipeline sustains before OOM |

Re-running after a disconnect skips whatever already completed.

### Cell 6 — bring the results home

```python
%cd /content/edgerag
!mkdir -p results
!cp /content/drive/MyDrive/edgerag/baseline.jsonl /content/drive/MyDrive/edgerag/oom_probe.json results/
!cat results/oom_probe.json
!python -c "import json; from bench.bench import records_to_markdown; rows=[json.loads(l) for l in open('results/baseline.jsonl') if l.strip()]; print(records_to_markdown([r for r in rows if 'nocache' not in r['name']], ignore=['batch_size']))"
```

`mkdir -p results` matters: git does not track empty directories, so a fresh clone has no
`results/` and `cp` fails with *"cannot create regular file … Not a directory"*. The failure is
quiet — the table then renders `_no records_` rather than erroring. The `nocache` row is filtered
out because it generates 16 tokens against the others' 64, so the harness correctly refuses to
tabulate it alongside them.

Download `baseline.jsonl` from Drive and commit it to `results/`. Those numbers are the
denominator for every claim in the README.

---

## What to expect, and what would be surprising

From `CONTEXT.md` D11, one k=5 request costs **1,267 MiB** of KV cache and the 2.2B weights are
~4.4 GiB in fp16. On a 16 GB T4 that predicts:

| batch | weights + KV | verdict |
|---|---|---|
| 1 | ~5.7 GiB | fits easily |
| 2 | ~6.9 GiB | fits |
| 4 | ~9.5 GiB | fits, getting tight |
| 8 | ~14.5 GiB | **OOM likely** |

So expect `oom_probe` to land around **4–8**. If it OOMs at 2, something is wrong with prompt
assembly — check the prefill token count against the ~6,758 median. If it sails past 8, the KV
arithmetic in `tests/test_spec.py` is wrong and that matters far more than the baseline.

**Do not** compare the `nocache` row's throughput to the cached rows as a headline "KV cache
speedup" — it generates 16 tokens against the others' 64, and without a cache the cost is
quadratic in prefix length. It is a shape comparison for the Phase 2 plot, not a ratio to quote.
The harness will refuse to put them in one table anyway; `max_new_tokens` differs.
