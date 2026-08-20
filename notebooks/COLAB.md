# Fresh session, start to finish

Use this when starting on a **new Google account** or an entirely new runtime — nothing from a
previous session carries over, including the Drive folder.

Ordering is deliberate: **cheapest and most valuable first**, so a disconnect costs the least.
The gather measurement needs neither the corpus nor the model weights, so it runs in two minutes
before anything is downloaded. The smoke test costs one extra minute and has caught a failure
that would otherwise have wasted twenty-five.

**What does not belong in a T4 session:** anything exact. The Phase 6 memory column — every
{fp16, int8, int4} × {LM, LM+ViT, ViT} cell — is arithmetic over tensor shapes and runs locally in
two seconds with no weights (`python -m scripts.measure_memory_ledger`, `CONTEXT.md` D21). Spend
GPU quota only on numbers that need a GPU: throughput and quality.

### 0 · Switch to a GPU runtime — before anything else

Runtime → Change runtime type → **T4 GPU** → Save.

A CPU runtime shows only "System RAM" and "Disk"; a T4 session adds a **GPU RAM** bar. Every
script below refuses to record a number on anything but a T4, so getting this wrong fails fast
rather than producing unpublishable results.

### 1 · Confirm what you got

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Must say **Tesla T4**. If it says anything else, change the runtime and try again.

### 2 · Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/edgerag
```

A new account will prompt for authorisation. Results are fsync'd here after every measurement, so
a disconnect costs one measurement rather than the session.

### 3 · Clone and install

```python
!git clone https://github.com/Tejas544/edgerag.git /content/edgerag
%cd /content/edgerag
!pip install -q -e ".[serve]" 2>&1 | tail -3
```

**`[serve]` matters**, and only for step 9: `fastapi`, `uvicorn` and `sse-starlette` are optional
extras, not base dependencies, so a plain `pip install -e .` gets you through every measurement
cell and then fails at the server with `No module named 'fastapi'`.

If `/content/edgerag` already exists, `git clone` silently does nothing — run
`!cd /content/edgerag && git pull --ff-only` instead, or the session stays pinned to an old commit
and new scripts fail with `No module named scripts.<x>`.

### 4 · Gather overhead — 2 min, no weights, no corpus

```python
!python -m scripts.colab_gather_overhead --drive /content/drive/MyDrive/edgerag
```

Runs from the model config alone with synthetic KV, because gather cost depends on tensor shape
and memory layout, not on values. Answers `CONTEXT.md` D3 and prices the head-major pool change.

> ✅ **Measured 2026-08-17 — `CONTEXT.md` D19.** The head-major layout took the gather down
> ~20% in absolute time, but the share only moved **77.3% → 72.7%** at the median request length.
> Still ~3× D3's 25% threshold, so the fused kernel is now required rather than optional. Paged
> attention itself costs **+0.55%** over contiguous — the indirection is free and the copy is the
> whole bill. Re-running this cell is now a regression check, not an open question.

### 5 · Corpus and frozen trace — ~6 min, CPU only

```python
!python -m scripts.build_corpus --infographic 250 --docvqa 400
!python -m scripts.build_trace --k 5
```

Corpus images are ~1 GB and not in git, so they are rebuilt from the same deterministic stream.
**The fingerprint must read `94b148a0b9f5006e`.** If it does not, the workload is not the one every
other number was measured against — stop and report it rather than continuing.

No GPU time is spent here, so a disconnect during setup costs no quota.

### 6 · Smoke test the quality run — ~1 min after the download

```python
!python -m scripts.colab_pruning_quality --drive /content/drive/MyDrive/edgerag \
    --n-queries 2 --keep-ratios 1.0 --strategies attention
```

Two requests at `keep_ratio=1.0`, which is deliberately the **worst case**: nothing is pruned, so
both halves of the cache hold the whole sequence and block demand peaks. If this passes, the full
sweep will.

Most of the minute is the 9 GB weight download, which the session then caches — so step 7 starts
immediately.

**What good looks like:** `after load:` around **4.4 GiB reserved of 14.56**, a `block pool:` line
whose "worst case needs" is below the pool size, then a result line with **`n=2`**. If you see
`ABORTING` or `n=0`, stop and send the output.

### 7 · The full quality curve — ~20 min

```python
!python -m scripts.colab_pruning_quality --drive /content/drive/MyDrive/edgerag --n-queries 40
```

The half of Phase 4 that cannot be computed locally. One result line per configuration, roughly
every two minutes.

> ✅ **Measured 2026-08-17 — `CONTEXT.md` D20.** Both questions are answered, and neither the way
> the plan hoped. The curve **has no knee**: ANLS falls 0.438 → 0.203 at keep=0.5, so the
> "50–75% removed while quality holds" target is not reachable on document RAG. And `attention`
> beats `uniform` only at keep=0.75; below half, uniform wins — coverage beats salience once the
> budget is small. Note n=40 gives a standard error near 0.06, so gaps under ~0.12 in that table
> are ties. If this is re-run, spend the time on **more queries, not more ratios**.

### 8 · The quantization ablation — ~30 min

```python
!python -m scripts.colab_quant_ablation --drive /content/drive/MyDrive/edgerag
```

The two Phase 6 columns that need a GPU. The third is already done: the memory column is exact
arithmetic and ran locally (`CONTEXT.md` D21), so this script does not recompute it — **it checks
it**, printing the delta between the bytes each arm actually holds and the bytes the ledger
predicted. A delta over 1 MiB is a finding about the ledger, and the run continues so you keep the
measurement that produced it.

Smoke test first if quota is tight — one arm, two queries, about four minutes:

```python
!python -m scripts.colab_quant_ablation --drive /content/drive/MyDrive/edgerag \
    --arms fp16 --n-queries 2 --trials 1
```

**Read three things, in this order:**

- **`ledger agrees`** on every arm. This is the cheapest possible check that the model on the card
  is the model the README describes.
- **the `tok/s` column, expecting INT4 to be SLOWER.** `QuantLinear` dequantizes then calls a
  normal matmul, and a T4 has no INT4 tensor cores, so the packed bytes are expanded before they
  reach the multiplier. D7 predicted this in advance and chose it as the documented cut line: the
  memory win is real, the speed win needs a fused kernel. A number below 1.00x here is the
  expected result, not a regression.
- **the `ViT` rows' `tok/s`, which must match fp16.** The tower runs once in prefill and is not in
  the decode loop, so quantizing it cannot move decode throughput. If it does, distrust the run.

Resumable per arm: a completed arm is appended to the file and skipped on the next invocation, so
a disconnect costs one arm rather than the session.

### 8b · The single-session latency sweep — ~18 min

`CONTEXT.md` D24 finding 3 left one caveat open: the `tok/s` column was assembled across three
Colab sessions, and D14 finding 5b measured 12–14% clock variance between them. Two sessions
agreed to 1.4% while a third sat 15% low, which is visible in the `ViT` arms — they cannot affect
decode speed at all and still differ.

**Only latency needs re-measuring.** Quality and weights are session-independent, and that is
evidence rather than assumption: `fp16`, `LM@int8` and D20's independent pruning run all scored
`0.4378445165945166` — the same float, three sessions, two scripts — and `ledger_delta_bytes` is
0 on every arm every time. Re-running the 40-query quality loop would spend 36 of 54 minutes
reproducing numbers that provably cannot move, and would triple the disconnect risk on the one
measurement that actually needs a controlled session.

So: all eight arms, one session, two queries each, and the time saved buys **5 trials instead of
3** on the thing being measured.

```python
!python -m scripts.colab_quant_ablation --drive /content/drive/MyDrive/edgerag     --out-name quant_latency.jsonl --n-queries 2 --trials 5
```

It writes its own file rather than appending to the quality table: `summarise()` keeps the last
row per arm, so a 2-query ANLS would quietly replace the 40-query numbers.

**If the runtime is reclaimed part-way, that is expected and not a failure.** Eight sequential
loads of a 4.5 GiB checkpoint is the longest-running cell in this runbook, and free Colab has
reclaimed it twice. Every arm is fsync'd as it completes, so nothing is lost, and each now prints
elapsed time plus free host/GPU memory — if a future run dies, that line distinguishes an
environment quota from a leak instead of leaving it to guesswork.

To finish an interrupted sweep, run the missing arms **with an `fp16` anchor in the same session**,
into their own file. Ratios are then computed within each file rather than across them:

```python
!python -m scripts.colab_quant_ablation --drive /content/drive/MyDrive/edgerag     --out-name quant_latency_b.jsonl --arms fp16 ViT LM8+ViT4 --bits 8 --n-queries 2 --trials 5
```

**The last line is the point.** It must read `SINGLE SESSION (<id>)`. Every record now carries a
`session_id` generated once per process, so "all eight arms in one session" is a property the file
can be checked for rather than a claim about how you ran it. If it says `2 SESSIONS`, the run was
interrupted and resumed — the numbers are still usable, but the `vs fp16` column is only
trustworthy between rows sharing an id.

Two checks while it runs:

- **`ViT@int8` and `ViT@int4` should now land within a couple of percent of `fp16`.** That is the
  built-in control: the vision tower is not in the decode loop, so quantizing it cannot change
  decode speed. In the multi-session table they differed by 13.4%, which was the confound showing
  itself.
- **INT4 should still be ~4× slower than fp16.** That is D7's prediction and it survives the noise;
  what the clean session buys is the second decimal place, not the headline.

### 8bb · Close D24's cross-session latency gap — ~15 min

D24 finding 3 is still open in one respect: the throughput column has never been measured with all
eight arms in a single session, and cross-session `tok/s` carries ~7.5% clock variance — wider than
several of the gaps the table reports. Two earlier attempts were reclaimed part-way, and the
resolution had to be reconstructed by hand from console output.

**Ask the files what is missing rather than guessing.** This needs no GPU and runs anywhere,
including before you start a runtime:

```python
!python -m scripts.latency_coverage
```

It groups every `results/quant_latency*.jsonl` record by `session_id`, reports which session has
the widest coverage and whether it carries an `fp16` anchor (without one the rows are absolute
numbers with nothing to divide by), and **prints the exact command that would close the gap** —
including a note about which extra arms the `--arms × --bits` cross product will also re-measure.

Run whatever it prints. At the time of writing that is:

```python
!python -m scripts.colab_quant_ablation --drive /content/drive/MyDrive/edgerag \
    --out-name quant_latency_finish.jsonl \
    --arms fp16 LM LM+ViT ViT LM8+ViT4 --bits 8 4 --n-queries 2 --trials 5
```

**The last line of that run is the check.** It must read `SINGLE SESSION (<id>)`. Then re-run
`latency_coverage` — it should print `COMPLETE`, at which point D24's cross-session caveat can be
retired from `CONTEXT.md` and the README rather than carried forward another phase.

Quality is *not* re-measured here and does not need to be: `fp16`, `LM@int8` and D20's independent
pruning run all scored the same float to sixteen digits across three sessions, and every arm's
`ledger_delta_bytes` is 0 every time. Only latency is session-sensitive.

### 8c · Phase 5e: the serving layer under load — ~18 min

The gate that was cut when the schedule ran out, and the only one still missing. Phases 5a–5d
proved the scheduler *correct* — admission, chunked prefill, preemption, pool conservation, all
property-tested without a GPU — and then nothing ever put load on it. Everything downstream is
still blank: the throughput-vs-concurrency plot, the p99 TTFT figure, and the two empty slots in
`01_EDGERAG.md` §8's CV bullet.

**Predictions, recorded here before the run so the result is allowed to disagree:**

1. **Aggregate throughput rises far less than concurrency does.** D14 measured 1.32× aggregate for
   4× batch on this checkpoint — it is MHA, decode is bandwidth-bound on KV reads at 192 KiB/token,
   and a scheduler moves work around rather than moving bytes faster. Near-linear scaling here
   should be distrusted before it is believed.
2. **Chunked prefill wins p95 TTFT and loses a little throughput.** Fourteen 512-token chunks cost
   more in launch overhead than one 6,758-token pass; what they buy is that a decoding request
   waits one chunk instead of one whole prefill.
3. **p95 TTFT degrades superlinearly past the service rate.** That is queueing theory, not a
   property of this code — seeing it is a check that the driver really is open-loop.

**Two things about the setup that are deliberate and will otherwise look like mistakes.**

*The pool is bigger than the 4 GiB budget allows.* 1792 blocks × 16 tokens is ~5.25 GiB and admits
4 concurrent requests. The shipped 640-block pool admits **exactly one** — a 6,758-token request
needs ~425 blocks — so a concurrency sweep at the ship configuration would be a flat line
describing the pool rather than the scheduler. The run prints both numbers. *The budget, not the
scheduler, is what caps concurrency at the ship configuration* is the finding, and it only reads
as a finding if the scheduler was measured somewhere it had room.

*Prompts are built before the timed window.* Retrieval and the vision tower are once-per-request
work on the caller's thread (`edgerag/serve/pipeline.py` says so and says why), and they do not
vary with concurrency. Inside the window they would add a near-constant to every TTFT and shrink
the relative size of the queueing being measured. The constant is recorded as `prompt_build_s`
rather than dropped.

This cell does not depend on step 8 — but running it after step 8 means the 4.5 GiB checkpoint is
already cached, so it starts in seconds instead of minutes.

**Smoke test first — ~4 min after the weights are cached.** Two requests at one load factor, no
chunk comparison. It exercises prompt building, calibration, the driver, the summary and the file
write, which is every part that can fail for a reason unrelated to load:

```python
!python -m scripts.colab_poisson --drive /content/drive/MyDrive/edgerag \
    --out-name poisson_smoke.jsonl --n-requests 2 --n-prompts 2 \
    --load-factors 1.0 --skip-chunk-comparison
```

**What good looks like:** a `pool:` block reporting **4 concurrent** at 1792 blocks and **1** at
the shipped 640, a calibration line printing a service time in the 6–12 s range, then one cell
line with `completed 2/2` and `Pool conservation held`. If the pool line says fewer than 2
concurrent the run aborts and tells you what `--num-blocks` to pass instead.

**The full sweep:**

```python
!python -m scripts.colab_poisson --drive /content/drive/MyDrive/edgerag
```

Four offered loads (0.5×, 1×, 2×, 4× the measured single-request service rate) plus one unchunked
control at 2×. Offered load is expressed as a multiple of *measured* service rate rather than in
requests/second, so the same numbers mean the same thing on fp16 and on INT4 — which fixed rates
would not.

**Read four things, in this order:**

- **`Pool conservation held on every cell`.** Blocks free before == blocks free after. A leak does
  not fail a cell; it silently shrinks every *later* cell's pool, so the sweep would report a
  declining curve that is an artifact of its own earlier rows.
- **the `in-flight` column against `admission blocked`.** If admission never blocked and in-flight
  sat at 2, the binding constraint was `max_prefills_per_step=1`, not the block pool — one prefill
  per iteration serialises 14-iteration prefills long before blocks run short. The summary says so
  when it detects it. Pass `--max-prefills-per-step 2` to separate the two effects.
- **the chunked-prefill table.** Same offered load, same arrival seed, one variable. Unchunked p95
  TTFT should be the larger number; that ratio is what D18 and `BUGS.md` P-18 were written to
  predict and has never been priced.
- **the `*` on every p99.** At 12 requests per cell the p99 is the single worst request wearing a
  percentile's name. Quote **p95 over n=12** in the README, not a p99 the sample cannot support.
  A real p99 needs `--n-requests 100`, which is roughly an hour per cell — worth one cell at 2×
  load if quota allows, not worth all five.

Resumable per cell, fsync'd per cell. A disconnect costs one cell.

> ✅ **Measured 2026-08-19 — `CONTEXT.md` D25.** The knee is at **2× offered load**: throughput
> rises 2.29× for 3.53× the mean in-flight depth and then goes flat while p95 TTFT keeps climbing.
> Predictions 1 and 3 hold. **Prediction 2 is falsified** — chunked prefill is 1.51× *worse* on
> p95 TTFT — and the decomposition is what rescues it: once admitted, a chunked request decodes at
> 5.8 tok/s against an unchunked one's 1.4, so chunking buys **4.2× on the decode phase** and
> 1.23× end to end. The TTFT cost is a different queue entirely — `max_prefills_per_step=1` means
> a 15-chunk prefill holds the only prefill slot for 15 iterations and nothing new is admitted.
> **The controlled test for that is one cell and has not been run:**
>
> ```python
> !python -m scripts.colab_poisson --drive /content/drive/MyDrive/edgerag \
>     --out-name poisson_prefill_slots.jsonl --load-factors 2.0 \
>     --max-prefills-per-step 4 --skip-chunk-comparison
> ```
>
> If the p95 TTFT penalty disappears, the mechanism is confirmed. If it does not, D25 finding 3 is
> wrong and the cause is somewhere else. ~3 min.

**If you have quota for one more thing, make it this** — a real tail at the interesting load:

```python
!python -m scripts.colab_poisson --drive /content/drive/MyDrive/edgerag \
    --out-name poisson_p99.jsonl --load-factors 2.0 --n-requests 100 --skip-chunk-comparison
```

**Bring it home and render the plot:**

```python
%cd /content/edgerag
!mkdir -p results
!cp /content/drive/MyDrive/edgerag/poisson_sweep.jsonl results/
!python -m scripts.make_plots
```

`make_plots` skips `serving_tradeoff.png` with a pointer when the file is absent, so this is the
cell that finally draws it. The figure names the device it was told about rather than assuming a
T4, and stamps **UNTRUSTED DEVICE** across the provenance line if the record says so.

---

### 9 · Boot the server and ask it a question — ~3 min

Phase 7's gate end to end: retrieval, the quantized model, the paged cache, and streaming HTTP.
Starts in the background because the notebook needs its cell back.

```python
import importlib.util, json, pathlib, subprocess, time, urllib.request

REPO = '/content/edgerag'

# Preflight. Both of these otherwise surface as a confusing death five seconds later, with the
# real cause buried in a log the traceback does not point at.
problems = []
if not pathlib.Path(REPO, 'scripts/serve_rag.py').exists():
    problems.append('scripts/serve_rag.py is missing -- this clone predates it. Run:'
                    '\n    !cd /content/edgerag && git pull --ff-only')
if importlib.util.find_spec('fastapi') is None:
    problems.append('the serving extras are not installed. Run:'
                    '\n    !pip install -q -e "/content/edgerag[serve]"')

if problems:
    print('cannot start:\n\n' + '\n\n'.join(problems))
else:
    log = open('/content/server.log', 'w')
    server = subprocess.Popen(
        ['python', '-m', 'scripts.serve_rag', '--port', '8000'],
        stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
    )
    # The weights are already cached by step 8, so this is load time, not download time.
    for attempt in range(60):
        time.sleep(5)
        if server.poll() is not None:          # checked BEFORE connecting: a dead process is
            print(open('/content/server.log').read()[-3000:])   # the answer, not a refused port
            print('\n^ server exited during startup -- the cause is above')
            break
        try:
            health = json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))
            print(f'ready after {(attempt + 1) * 5}s:', health)
            break
        except Exception:
            continue
    else:
        print('did not come up within 5 minutes. Tail of the log:')
        print(open('/content/server.log').read()[-2000:])
```

Defaults to **`LM8+ViT4`**, the configuration D24 measured: 2.296 GiB against fp16's 4.185, at a
quality difference of 0.31 sigma. Pass `--arm fp16` to serve unquantized for comparison.

Now ask it something. The corpus is DocVQA and InfographicVQA pages, so ask what a document says:

```python
!curl -s http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"What percentage of users are female?"}],"max_tokens":32}'   | python -m json.tool
```

**Read `retrieved` before `content`.** It lists the page keys the answer was built from — an
answer whose sources are wrong is a *retrieval* result, not a generation one, and the two get
misdiagnosed as each other constantly.

Streaming, which is the actual gate wording:

```python
!curl -N -s http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"What is the total?"}],"max_tokens":32,"stream":true}'
```

Tokens should arrive one `data:` line at a time and finish with `data: [DONE]`. If they all appear
at once, the stream is being buffered somewhere — the server sets `X-Accel-Buffering: no` for
exactly that reason, but `curl` without `-N` will do it too.

When you are done:

```python
server.terminate(); server.wait(); log.close()
```

### 9b · A public URL you can put in front of an interviewer — ~1 min

Step 9's server listens on `127.0.0.1` inside a Colab VM, which nobody else can reach. A
Cloudflare quick tunnel puts it on a public hostname with no account and no config:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

import re, subprocess, time
tunnel = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000', '--no-autoupdate'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in tunnel.stdout:                      # the URL is printed once, then never again
    found = re.search(r'https://[-\w]+\.trycloudflare\.com', line)
    if found:
        print('\npublic:', found.group(0))
        print(f"try:    curl -N {found.group(0)}/v1/chat/completions \\")
        print("          -H 'Content-Type: application/json' \\")
        print('          -d \'{"messages":[{"role":"user","content":"What is the total?"}],'
              '"max_tokens":32,"stream":true}\'')
        break
```

**Read this before you paste the URL anywhere.** A quick tunnel is *public and unauthenticated* —
anyone with the link can send requests to your GPU for as long as it is up, and the corpus pages
come back in the `retrieved` field of every response. It dies with the tunnel process and with the
runtime, so treat it as something you start for a call and stop afterwards, not as a deployment.
`tunnel.terminate()` when you are done.

The health endpoint is the cheapest thing to check first:

```python
!curl -s <the-url-above>/health | python -m json.tool
```

**If you want the demo asset rather than a live link**, `scripts/make_demo.py` records a real
request and renders it to an animated SVG. Re-recording on the T4 with the 2.2B replaces the
GTX 1650 stamp with a Tesla T4 one, and is the version worth committing if you have the quota:

```python
!python -m scripts.make_demo --port 8000 --max-tokens 24 --speed 4
```

### 10 · Bring it home

```python
%cd /content/edgerag
!mkdir -p results
!cp /content/drive/MyDrive/edgerag/*.json /content/drive/MyDrive/edgerag/*.jsonl results/
!ls -la results/
```

Then download `results/` (or commit it) and say so — the numbers get folded into `CONTEXT.md` and
the README.

---

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

---

# Session 2 — Phase 4 quality, refined OOM probe, gather overhead

Cells 1–4 above are identical (check the GPU, mount Drive, clone, rebuild corpus). **The trace
fingerprint must still read `94b148a0b9f5006e`** — if it does not, the workload changed and
nothing here is comparable to session 1.

### Cell 4b — pull, if the runtime survived from an earlier session

```python
%cd /content/edgerag
!git pull --ff-only
```

`git clone` in cell 3 is a no-op when `/content/edgerag` already exists, so a runtime that has
been alive since an earlier session is pinned to whatever commit it cloned. The symptom is
`No module named scripts.<something>` for a script added since — which reads like a broken command
rather than a stale checkout. Run this before the cells below.

Then run these three. Budget ~35 min total; the ordering puts the cheapest and most valuable
first, so a disconnect costs the least.

### Cell 5a — gather overhead (~2 min, no weights)

```python
!python -m scripts.colab_gather_overhead --drive /content/drive/MyDrive/edgerag
```

Loads **no model weights** — gather cost depends on tensor shape and layout, not values — so this
is a two-minute cell rather than a 9 GB download. Run it first.

It answers `CONTEXT.md` D3's promise: *"measure the gather overhead as a fraction of decode time
and publish it."* Without this number, "why didn't you write a fused kernel?" is answered with a
shrug.

**Watch the final line.** Local (untrusted) numbers put gather near 60% of the paged attention
path, well above D3's 25% revisit threshold. If the T4 agrees, `CONTEXT.md` P6 is the cheap fix
(one pool-layout change removes a redundant copy) and a fused kernel is the expensive one.

### Cell 5b — refined OOM probe (~8 min)

```python
!python -m scripts.colab_baseline --drive /content/drive/MyDrive/edgerag --only oom_probe
```

`--only` re-measures just that cell and suppresses the rest, so this does not repeat session 1's
baseline. It is needed because the original probe only doubled (1, 2, 4, 8) and so proved
*"at least 4, fewer than 8"*, not 4. Publishing 4 as the denominator would inflate any later
"N× more concurrent sequences" claim by up to 75%. The probe now walks the gap linearly.

Expect `max_ok` between 4 and 7, and `"refined": true`.

### Cell 5c — Phase 4 quality curve (~25 min, needs the 2.2B)

```python
!python -m scripts.colab_pruning_quality --drive /content/drive/MyDrive/edgerag --n-queries 60
```

The half of Phase 4 that cannot be computed locally. `scripts/measure_pruning_memory.py` already
gives MiB reclaimed exactly; **publishing that without this would be the flattering half.**

Scores held-out questions with ANLS at each keep ratio, for both the attention-based selector and
a uniform-stride control. Two things to read:

- **Where the curve falls off.** The plan targets 50–75% of visual tokens removed at ≥95% quality
  retention. `keep=0.5` reclaims 443 MiB; whether it holds quality is exactly what this measures.
- **Whether `attention` beats `uniform` at the same ratio.** If it does not, the honest finding is
  *"visual tokens are redundant on this workload"*, not *"FastV works"* — a real result either
  way, but a different one.

The `keep=1.0` row is proven bit-identical to running no compressor at all (`atol=0, rtol=0` in
`tests/test_fastv.py`), so it is the true baseline rather than a near-miss.

### Cell 6 — bring it all home

```python
%cd /content/edgerag
!mkdir -p results
!cp /content/drive/MyDrive/edgerag/*.json /content/drive/MyDrive/edgerag/*.jsonl results/
!ls -la results/
```

Then commit `results/` and tell me — I will fold the numbers into `CONTEXT.md` and the README.

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
