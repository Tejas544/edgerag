"""One request, end to end, through the code that ships.

Extracted from ``scripts/colab_pruning_quality.py`` when Phase 6 needed the same path. Two
experiments that share a pipeline must share the *implementation* of it: a copied decode loop
diverges on the first bug fix, and then the pruning curve and the quantization table are measuring
two subtly different systems while both claim to hold everything else constant
(``00_FOUNDATIONS.md`` §4 rule 5).

Our decoder, our greedy loop, our cache. ``01_EDGERAG.md`` §2 forbids ``generate()``, and the
point of the rule is that what gets benchmarked should be what gets shipped.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image

from bench.metrics import sync
from edgerag.compress.fastv import FastVCompressor, FastVConfig, build_visual_mask
from edgerag.core.model import encode_images_chunked, merge_image_features
from edgerag.retrieval.trace import build_prompt_messages

REPO_ROOT = Path(__file__).resolve().parents[1]


class KVCacheLike(Protocol):
    """What :func:`generate` needs from a cache, which is deliberately almost nothing.

    Phase 4 passes a ``CompressedKVCache`` (two pools, one either side of the pruning cut); Phase 6
    passes a plain ``PagedKVCache``, because quantization is measured with pruning *off* and
    allocating a second pool to leave it empty would distort the memory column being measured.
    Typing the parameter as a protocol rather than a concrete class is what lets the same function
    serve both without either experiment inheriting the other's cache.
    """

    def reset(self) -> None: ...


@torch.inference_mode()
def generate(
    decoder: torch.nn.Module,
    hf_model: torch.nn.Module,
    processor: Any,
    spec: Any,
    entry: Any,
    docs_by_key: dict,
    config: FastVConfig,
    strategy: str,
    max_new_tokens: int,
    device: torch.device,
    cache: KVCacheLike,
) -> tuple[str, dict[str, Any]]:
    """Greedy decode one trace request. Returns the answer text and its timings.

    ``cache`` is supplied by the caller and **reused across requests**. Constructing it here --
    which is what this function did originally -- allocates a fresh block pool per request:
    1024 blocks x 16 tokens x 192 KiB/token is **3.2 GiB**, on top of 4.18 GiB of weights, and it
    OOM'd on every single call. See ``BUGS.md`` B-05.

    Prefill and decode are timed separately, and both are reported. They are different regimes --
    prefill is compute-bound and decode is memory-bandwidth-bound (D14) -- so a single end-to-end
    number averages two things that respond to opposite optimisations.
    """
    docs = [docs_by_key[k] for k in entry.retrieved_doc_keys if k in docs_by_key]
    images = [Image.open(REPO_ROOT / d.image_path).convert("RGB") for d in docs]
    prompt = processor.apply_chat_template(
        build_prompt_messages(docs, entry.question), add_generation_prompt=True
    )
    batch = processor(text=prompt, images=images, return_tensors="pt")
    for image in images:
        image.close()

    input_ids = batch["input_ids"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    features = encode_images_chunked(hf_model, pixel_values)
    embeds = decoder.embed_tokens(input_ids)
    embeds = merge_image_features(input_ids, embeds, features, spec.image_token_id)
    visual_mask = build_visual_mask(input_ids, spec.image_token_id)

    cache.reset()
    compressor = FastVCompressor(config, strategy=strategy) if config.enabled else None

    t0 = time.perf_counter()
    logits = decoder(
        inputs_embeds=embeds, cache=cache, compressor=compressor, visual_mask=visual_mask
    )
    sync()  # not torch.cuda.synchronize(): this must be runnable on CPU so a test can cover it
    ttft = time.perf_counter() - t0

    tokens: list[int] = []
    next_id = int(logits[0, -1].argmax())
    eos = processor.tokenizer.eos_token_id
    decode_start = time.perf_counter()
    for _ in range(max_new_tokens):
        if next_id == eos:
            break
        tokens.append(next_id)
        step = decoder(input_ids=torch.tensor([[next_id]], device=device), cache=cache)
        next_id = int(step[0, -1].argmax())
    sync()
    decode_s = time.perf_counter() - decode_start

    # A cache that does not compress has no savings to report, and inventing a zero would put a
    # meaningless "0 MiB reclaimed" column in the quantization results.
    savings = cache.savings() if hasattr(cache, "savings") else {}

    return processor.tokenizer.decode(tokens, skip_special_tokens=True), {
        "ttft_s": ttft,
        "decode_s": decode_s,
        "generated_tokens": len(tokens),
        # Recorded because CONTEXT.md D21 could not: the baseline harness stored `prompt_tokens:
        # -1`, so the activation term in the memory ledger had to be inferred from a median
        # request length instead of the length that was actually run.
        "prefill_tokens": int(input_ids.shape[1]),
        **savings,
    }


def free_duplicate_hf_decoder(hf_model: torch.nn.Module) -> None:
    """Drop HuggingFace's text decoder once its weights have been copied into ours.

    Deliberately surgical: ``vision_model`` and ``connector`` must survive, because
    ``encode_images_chunked`` still runs them (``CONTEXT.md`` D2). Everything else in the HF tree
    is now a second copy of weights we own.
    """
    inner = getattr(hf_model, "model", hf_model)
    for owner, name in ((inner, "text_model"), (hf_model, "lm_head")):
        if hasattr(owner, name):
            delattr(owner, name)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
