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

import torch
from torch import nn

from edgerag.core.quant import (
    QuantConfig,
    dequantize_groupwise,
    pack_int4,
    quantize_groupwise,
    unpack_int4,
)


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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        if bias:
            raise NotImplementedError("no quantized layer in this model has a bias")
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()

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

    def load_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        if bias is not None:
            raise ValueError("bias is not supported on a quantized layer")
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

    def dequantized(self) -> torch.Tensor:
        codes = (
            unpack_int4(self.packed, self.in_features)
            if self.config.bits == 4
            else self.packed.to(torch.int8)
        )
        return dequantize_groupwise(codes, self.scales, self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dequantized().to(x.dtype))

    def weight_bytes(self) -> int:
        """Actual bytes held -- packed weights **plus scales**.

        Counting only the packed weights would overstate the saving. At group size 128 the scales
        add one fp16 per 128 values, so real INT4 is 4.125 bits per weight, not 4.
        """
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


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
