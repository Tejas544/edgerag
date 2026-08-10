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
