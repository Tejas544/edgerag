"""The linear-layer seam that quantization drops into.

Built in Phase 2 and filled in Phase 6. The point of introducing it now, while there is only one
implementation, is that quantization then becomes a **config flag rather than a refactor**: by
Phase 6 the model, the paged cache, and the scheduler all exist, and threading a new layer type
through them at that point would touch every file and invalidate every equivalence test at exactly
the wrong moment.

Only ``LinearBase`` is used before Phase 6. It is a thin wrapper over ``nn.Linear`` and costs
nothing.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from edgerag.core.quant import (
    QuantConfig,
    dequantize_groupwise,
    pack_int4,
    quantize_groupwise,
    unpack_int4,
)

#: Anything that builds a linear layer from ``(in_features, out_features, bias=...)``.
#:
#: A class satisfies it, and so does ``partial(QuantLinear, config=cfg)`` -- which is how the
#: quantization flag reaches every projection in the stack without threading a config object
#: through four constructors. This is the seam doing its job: ``DecoderLayer`` never learns that
#: quantization exists.
LinearFactory = Callable[..., "LinearBase"]

#: fp16 is the only float dtype this project ships (``loader.load_model`` refuses bf16 on Turing;
#: fp32 is test-only), so the "dense" side of every memory comparison is 2 bytes per weight.
FP16_BYTES = 2


class LinearBase(nn.Module):
    """The interface every linear implementation satisfies.

    Weight-only quantization means activations stay fp16 and only ``weight`` changes
    representation, so the signature never changes -- which is exactly why this seam is cheap.
    """

    in_features: int
    out_features: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def load_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        """Take a dense fp16/fp32 weight from the checkpoint.

        Implementations that quantize do it here, once at load time -- never in ``forward``.
        """

    @abstractmethod
    def weight_bytes(self) -> int:
        """Actual bytes occupied, for the memory ledger.

        Declared on the interface so ``BudgetLedger`` never has to ask what kind of layer it is
        holding. An INT4 layer reports its packed size, not its logical parameter count.
        """


class FP16Linear(LinearBase):
    """Unquantized baseline. The reference every quantized implementation is compared against."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def load_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(weight.shape)} does not match layer "
                f"({self.out_features}, {self.in_features})"
            )
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            if bias is not None:
                if self.linear.bias is None:
                    raise ValueError("bias supplied to a layer constructed without one")
                self.linear.bias.copy_(bias)
            elif self.linear.bias is not None:
                raise ValueError("layer has a bias but none was supplied")

    def weight_bytes(self) -> int:
        total = self.linear.weight.numel() * self.linear.weight.element_size()
        if self.linear.bias is not None:
            total += self.linear.bias.numel() * self.linear.bias.element_size()
        return total


class QuantLinear(LinearBase):
    """Weight-only INT4/INT8 linear layer. Phase 6.

    Quantization happens once, in :meth:`load_weight`, never in ``forward`` -- a layer that
    re-quantizes per call would pay the cost on every token and defeat the purpose.

    ``forward`` currently dequantizes to fp16 and calls a normal matmul. **That is a memory win and
    a speed loss**, and saying so plainly is the honest position: on a T4 there are no INT4 tensor
    cores, so the packed bytes have to be expanded before they reach the multiplier and the
    expansion costs more than the narrower load saves. The speed win requires fusing the
    dequantize into the matmul so 4x fewer bytes actually cross the bus, which is a Triton kernel
    and Colab-only (``CONTEXT.md`` D7).
    """

    packed: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()

        if in_features % self.config.group_size:
            raise ValueError(
                f"in_features={in_features} is not divisible by group_size="
                f"{self.config.group_size}, so the final group would be ragged. Refusing to "
                "construct rather than quietly falling back to a group size that divides: the "
                "fallback changes how many bytes this layer occupies, and the memory ledger would "
                "stop describing the model that is actually running. SmolVLM2-2.2B's vision MLP "
                "is the real case -- in_features=4304 = 16 x 269, which no power-of-two group "
                "above 16 divides (CONTEXT.md D21)."
            )

        n_groups = in_features // self.config.group_size
        packed_width = in_features // 2 if self.config.bits == 4 else in_features
        # Buffers, not parameters: these are not trained, and registering them as parameters would
        # put integer tensors in the optimiser's path.
        self.register_buffer(
            "packed", torch.zeros((out_features, packed_width), dtype=torch.uint8)
        )
        self.register_buffer(
            "scales", torch.zeros((out_features, n_groups), dtype=torch.float16)
        )
        # The Llama decoder's projections carry no bias and SigLIP's all do, so a quantized layer
        # has to be able to hold one or the vision arm of the Phase 6 ablation cannot be built at
        # all. It stays fp16: one value per output channel is 0.05% of the layer at these shapes,
        # and quantizing it would add a second scale to reason about for nothing.
        self.register_buffer(
            "bias", torch.zeros(out_features, dtype=torch.float16) if bias else None
        )

    def load_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        if (bias is None) != (self.bias is None):
            raise ValueError(
                f"bias mismatch: layer was constructed with bias={self.bias is not None} but "
                f"load_weight was given bias={bias is not None}"
            )
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(weight.shape)} does not match layer "
                f"({self.out_features}, {self.in_features})"
            )

        codes, scales = quantize_groupwise(weight, self.config)
        packed = pack_int4(codes) if self.config.bits == 4 else codes.to(torch.uint8)
        with torch.no_grad():
            self.packed.copy_(packed.to(self.packed.device))
            self.scales.copy_(scales.to(self.scales.device))
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.to(self.bias.device))

    def dequantized(self) -> torch.Tensor:
        codes = (
            unpack_int4(self.packed, self.in_features)
            if self.config.bits == 4
            else self.packed.to(torch.int8)
        )
        return dequantize_groupwise(codes, self.scales, self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, self.dequantized().to(x.dtype), bias)

    def weight_bytes(self) -> int:
        """Actual bytes held -- packed weights **plus scales, plus any bias**.

        Counting only the packed weights would overstate the saving. At group size 128 the scales
        add one fp16 per 128 values, so real INT4 is 4.125 bits per weight, not 4.

        The scales follow the module's dtype rather than pinning themselves to fp16, because
        ``Module.to(dtype)`` casts every floating-point buffer and fighting that would be a
        surprise elsewhere. It matters only for the fp32 CPU test path; everything that ships is
        fp16, which is the dtype the ledger reports.
        """
        total = (
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
        )
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total


#: Layers that must stay fp16, and why. ``BUGS.md`` P-21: quantizing any of these produces an
#: immediate quality cliff that looks like "INT4 does not work on VLMs" and is really "INT4 does
#: not work on *these five layers*". Stating the skip list is itself part of the result.
#:
#: * ``embed_tokens`` -- a lookup table, not a matmul; quantizing it perturbs every token's
#:   starting point and there is no dequantize-fusion win to be had, because nothing multiplies.
#: * ``lm_head`` -- produces the logits that argmax reads directly, so its error is not attenuated
#:   by any downstream layer.
#: * ``*norm*`` -- per-channel gains near 1.0 with tiny variance; a 4-bit grid over that range is
#:   almost all rounding error, and they are a negligible share of parameters anyway.
#: * the vision tower's patch embedding -- the single point every pixel passes through.
QUANTIZATION_SKIP_LIST: tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    "norm",
    "layernorm",
    "patch_embedding",
)


def should_quantize(module_name: str) -> bool:
    """Whether a layer may be quantized, by name."""
    lowered = module_name.lower()
    return not any(skip in lowered for skip in QUANTIZATION_SKIP_LIST)


#: Left dense because ``QUANTIZATION_SKIP_LIST`` says so -- ``BUGS.md`` P-21.
SKIP_SENSITIVE = "skip-list"

#: Left dense because ``in_features`` is not divisible by the group size. Not a policy choice: the
#: layer cannot be constructed at this group size at all, and silently shrinking the group for one
#: layer would make the ledger describe a model nobody ran.
SKIP_RAGGED = "ragged-groups"


@dataclass(frozen=True)
class LayerPlan:
    """What one linear layer would cost quantized, computed from shapes alone.

    Shapes alone is the point: this is exact and needs neither a GPU nor a downloaded checkpoint,
    so the whole memory ablation runs locally in seconds against a ``meta``-device model
    (``CONTEXT.md`` D4). ``planned_bytes`` is not re-derived arithmetic -- it is
    :meth:`QuantLinear.weight_bytes` on a meta instance of the very layer that would be built, so
    the ledger cannot drift from the implementation.
    """

    name: str
    in_features: int
    out_features: int
    has_bias: bool
    dense_bytes: int
    planned_bytes: int
    skipped: str | None = None

    @property
    def n_weights(self) -> int:
        return self.in_features * self.out_features

    @property
    def saved_bytes(self) -> int:
        return self.dense_bytes - self.planned_bytes

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.planned_bytes / self.n_weights


@dataclass(frozen=True)
class QuantizationPlan:
    """Every linear layer in a module tree, classified. The Phase 6 memory column comes from this.

    Separating the plan from the act mirrors ``pack_int4`` being separate from
    ``quantize_groupwise``, and for the same reason: the accounting is the deliverable, and it
    must be inspectable without mutating a 4 GiB model to get it.
    """

    config: QuantConfig
    layers: tuple[LayerPlan, ...]

    @property
    def quantized(self) -> tuple[LayerPlan, ...]:
        return tuple(layer for layer in self.layers if layer.skipped is None)

    @property
    def skipped(self) -> tuple[LayerPlan, ...]:
        return tuple(layer for layer in self.layers if layer.skipped is not None)

    def skipped_for(self, reason: str) -> tuple[LayerPlan, ...]:
        return tuple(layer for layer in self.layers if layer.skipped == reason)

    @property
    def dense_bytes(self) -> int:
        """Bytes these layers occupy at fp16 -- the baseline every saving is measured against."""
        return sum(layer.dense_bytes for layer in self.layers)

    @property
    def planned_bytes(self) -> int:
        return sum(layer.planned_bytes for layer in self.layers)

    @property
    def saved_bytes(self) -> int:
        return self.dense_bytes - self.planned_bytes

    @property
    def bits_per_weight(self) -> float:
        """Effective width over the layers actually quantized, scales and all.

        Reported separately from the headline ratio because they answer different questions: this
        one says whether the quantizer is doing what it claims (4.125 at group 128), while the
        ratio over the *whole* model says what the skip list left on the table.
        """
        weights = sum(layer.n_weights for layer in self.quantized)
        if not weights:
            return 0.0
        return 8.0 * sum(layer.planned_bytes for layer in self.quantized) / weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "n_layers": len(self.layers),
            "n_quantized": len(self.quantized),
            "n_skipped": len(self.skipped),
            "dense_bytes": self.dense_bytes,
            "planned_bytes": self.planned_bytes,
            "saved_bytes": self.saved_bytes,
            "bits_per_weight": round(self.bits_per_weight, 4),
            "skipped": [
                {
                    "name": layer.name,
                    "reason": layer.skipped,
                    "shape": [layer.out_features, layer.in_features],
                    "dense_bytes": layer.dense_bytes,
                }
                for layer in self.skipped
            ],
        }


def _quantized_bytes(
    in_features: int, out_features: int, has_bias: bool, config: QuantConfig
) -> int:
    """Bytes a ``QuantLinear`` of this shape would hold, by building one on ``meta``.

    A meta tensor has shape and dtype but no storage, so this costs nothing and cannot disagree
    with the real layer -- which a second copy of the packing arithmetic eventually would.
    """
    with torch.device("meta"):
        return QuantLinear(in_features, out_features, bias=has_bias, config=config).weight_bytes()


def plan_quantization(
    root: nn.Module, config: QuantConfig | None = None, *, prefix: str = ""
) -> QuantizationPlan:
    """Classify every ``nn.Linear`` under ``root`` as quantizable or not, and price both outcomes.

    Works on a ``meta``-device model with no weights on disk, which is what makes the whole
    {fp16, int8, int4} x {LM, LM+ViT, ViT} memory table a local, exact, two-second computation
    instead of a T4 session.

    ``nn.Embedding`` and ``nn.Conv2d`` are not candidates at all -- the vision tower's patch
    embedding is a Conv2d, so the skip list catches it twice over, by type and by name.
    """
    config = config or QuantConfig()
    layers: list[LayerPlan] = []

    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        qualified = f"{prefix}{name}"
        has_bias = module.bias is not None
        dense = (module.in_features * module.out_features + (module.out_features if has_bias else 0)
                 ) * FP16_BYTES

        reason: str | None = None
        if not should_quantize(qualified):
            reason = SKIP_SENSITIVE
        elif module.in_features % config.group_size:
            reason = SKIP_RAGGED

        planned = (
            dense
            if reason is not None
            else _quantized_bytes(module.in_features, module.out_features, has_bias, config)
        )
        layers.append(
            LayerPlan(
                name=qualified,
                in_features=module.in_features,
                out_features=module.out_features,
                has_bias=has_bias,
                dense_bytes=dense,
                planned_bytes=planned,
                skipped=reason,
            )
        )

    return QuantizationPlan(config=config, layers=tuple(layers))


def quantize_module_(
    root: nn.Module, config: QuantConfig | None = None, *, prefix: str = ""
) -> QuantizationPlan:
    """Replace eligible ``nn.Linear`` layers under ``root`` with :class:`QuantLinear`, in place.

    This is how the *vision tower* gets quantized: it is HuggingFace's module tree, not ours
    (``CONTEXT.md`` D2), so there is no constructor to pass a flag to. The language decoder does
    not come through here -- it is built quantized from the start via
    ``load_from_hf(..., quant_config=...)``, because building it dense and replacing afterwards
    would allocate the full fp16 stack first and defeat the purpose on a 4 GiB device.

    Returns the same plan :func:`plan_quantization` would have returned, so what ran and what was
    budgeted are the same object.
    """
    config = config or QuantConfig()
    plan = plan_quantization(root, config, prefix=prefix)

    for layer in plan.quantized:
        parent_path, _, attr = layer.name[len(prefix):].rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        old = getattr(parent, attr)
        if old.weight.is_meta:
            raise ValueError(
                f"{layer.name} is on the meta device and has no weights to quantize. Use "
                "plan_quantization() for accounting; quantize_module_() needs real tensors."
            )
        # Build the replacement where the original lives, so a CUDA model never round-trips its
        # packed buffers through host memory.
        with torch.device(old.weight.device):
            new = QuantLinear(
                old.in_features, old.out_features, bias=old.bias is not None, config=config
            )
        new.load_weight(old.weight.data, None if old.bias is None else old.bias.data)
        setattr(parent, attr, new)

    return plan
