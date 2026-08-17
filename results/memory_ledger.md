# Memory ledger — HuggingFaceTB/SmolVLM2-2.2B-Instruct

Computed exactly from the checkpoint config at group size 128; no weights are loaded and no GPU is involved (`python -m scripts.measure_memory_ledger`).

## Weights, by ablation arm (GiB)

| arm | bits | language | vision | connector | total | vs fp16 |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 16 | 3.376 | 0.769 | 0.040 | 4.185 | 1.00x |
| LM | 8 | 1.900 | 0.769 | 0.040 | 2.708 | 1.55x |
| LM+ViT | 8 | 1.900 | 0.515 | 0.020 | 2.435 | 1.72x |
| ViT | 8 | 3.376 | 0.515 | 0.020 | 3.911 | 1.07x |
| LM | 4 | 1.150 | 0.769 | 0.040 | 1.958 | 2.14x |
| LM+ViT | 4 | 1.150 | 0.386 | 0.010 | 1.546 | 2.71x |
| ViT | 4 | 3.376 | 0.386 | 0.010 | 3.772 | 1.11x |

The fp16 row is over a 4 GiB budget on weights alone, before one KV block is allocated.

## The budget, arm LM+ViT at int4, one request in flight

| component | dtype | bytes | GiB | note |
|---|:--:|---:|---:|---|
| KV cache, 1 request | float16 | 1,341,259,776 | 1.2491 | 6758 median prompt tokens + 64 generated, at 192 KiB/token (MHA -- D10) |
| language | int4 | 1,234,374,720 | 1.1496 | quantized at 4.125 bits/weight; still fp16: 192 MiB skip list, 193 MiB embeddings/norms |
| vision | int4 | 414,507,456 | 0.3860 | quantized at 4.139 bits/weight; still fp16: 255 MiB ragged groups, 3 MiB embeddings/norms |
| activation + workspace | float16 | 353,548,512 | 0.3293 | inferred: measured T4 peak 5.763 GiB minus fp16 weights and the KV of a median 6758-token request; +/-0.17 GiB for the p10-p90 length spread (5921-7795 tokens) |
| connector | int4 | 10,948,608 | 0.0102 | quantized at 4.125 bits/weight |
| **total** | | **3,354,639,072** | **3.1243** | budget 4.00 GiB -- within budget |

Computed exactly from tensor shapes, not measured -- the budget itself is defined on `torch.cuda.max_memory_allocated()` (`CONTEXT.md` P3), and this table is the prediction that measurement will be checked against. **Excludes the CUDA context** (**not measured on this device** -- 300-600 MiB on Turing, per CONTEXT.md P3), which is reported separately because it is a driver cost, not a pipeline cost.

Headroom 0.88 GiB = **1.7 concurrent requests** at 1.25 GiB of KV each. The fp16 arm supports none.
