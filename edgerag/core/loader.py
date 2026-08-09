"""Model and processor loading.

The HF boundary declared in ``CONTEXT.md`` D2: weights, config, tokenizer, image processor, and
the vision tower forward come from ``transformers``. Everything downstream of the decoder input
embeddings is ours.

This module is the *only* place ``transformers`` is imported for model construction. Keeping the
boundary in one file means "what did you actually write?" has a one-line answer backed by
``git grep``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from edgerag.core.spec import ModelSpec

#: Headline model -- all published numbers come from this (CONTEXT.md D10).
HEADLINE_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

#: Local fixture -- fast enough that the equivalence suite runs every commit (CONTEXT.md D4).
FIXTURE_MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

#: Middle tier for local integration runs that need more than 256M of model.
INTEGRATION_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


@dataclass
class LoadedModel:
    """A loaded checkpoint plus the introspected spec that describes its memory behaviour."""

    model: torch.nn.Module
    processor: Any
    spec: ModelSpec
    device: torch.device
    dtype: torch.dtype

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def component_params(self) -> dict[str, int]:
        """Parameter counts split by component.

        The vision/decoder split is what makes the Phase 6 quantization ablation interpretable:
        "the vision tower is 12% of parameters but carries the quality cliff" is only a sentence
        you can say if you counted.
        """
        counts: dict[str, int] = {}
        for name, module in self.model.named_children():
            counts[name] = sum(p.numel() for p in module.parameters())
        return counts


def load_config(model_id: str) -> Any:
    return AutoConfig.from_pretrained(model_id)


def load_spec(model_id: str) -> ModelSpec:
    """Introspect a checkpoint's memory-relevant shape without downloading weights."""
    return ModelSpec.from_hf_config(model_id, load_config(model_id))


def load_model(
    model_id: str = FIXTURE_MODEL,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
) -> LoadedModel:
    """Load weights, processor, and spec.

    ``float16`` and never ``bfloat16``: both tiers are Turing (SM 7.5), which has no bf16 support
    at all. ``00_FOUNDATIONS.md`` §3 -- bf16 silently falls back or errors.
    """
    if dtype is torch.bfloat16:
        raise ValueError(
            "bfloat16 is unsupported on Turing (SM 7.5); both the GTX 1650 and the T4 require "
            "float16. See 00_FOUNDATIONS.md §3."
        )

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")

    config = load_config(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        processor=processor,
        spec=ModelSpec.from_hf_config(model_id, config),
        device=device,
        dtype=dtype,
    )
