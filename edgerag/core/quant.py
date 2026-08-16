"""Group-wise INT4 weight quantization.

Phase 6, filling the ``QuantLinear`` seam that Phase 2 put in place so this would be a config flag
rather than a refactor (``CONTEXT.md`` D7).

**Weight-only.** Activations stay fp16; only the stored weights change representation. That keeps
the layer signature identical, which is the whole reason the seam was cheap — and it is the right
choice for the actual bottleneck: D14 established that decode is memory-bandwidth-bound, so the
win comes from moving 4x fewer bytes across the bus, not from faster arithmetic.

**Group-wise, not per-tensor.** One scale for an entire weight matrix has to span every outlier in
it, which stretches the quantization grid until ordinary values land on the same few levels.
Splitting each row into groups of 128 gives each group its own scale, so an outlier ruins 128
values rather than 4 million. This is the single decision that separates INT4 that works from INT4
that produces confident nonsense (``BUGS.md`` P-20).

**Symmetric.** Zero maps exactly to zero, so no zero-point has to be stored or added. Asymmetric
packing buys a little accuracy on skewed distributions and costs an extra tensor plus an add in
the hot path; weight distributions here are near-symmetric, so it is not worth it.

**On a T4 there are no INT4 tensor cores** (``00_FOUNDATIONS.md`` §3). The memory win is real and
immediate; a *speed* win needs the dequantize fused into the matmul so the 4x-smaller bytes
actually reach the multiplier. Until that kernel exists, expect INT4 to be slower than fp16 and
report it that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Values per scale. 128 is the standard choice: small enough that a single outlier is contained,
#: large enough that the scales themselves stay a rounding error (one fp16 per 128 weights is
#: 0.4% overhead, against 12.5% at group size 4).
DEFAULT_GROUP_SIZE = 128

#: Signed 4-bit range. Symmetric quantization uses -7..7 rather than the full -8..7, so that
#: +max and -max map to mirror-image levels; using -8 would make the grid lopsided and bias every
#: dequantized value slightly negative.
INT4_MAX = 7


@dataclass(frozen=True)
class QuantConfig:
    group_size: int = DEFAULT_GROUP_SIZE
    bits: int = 4

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError(f"only 4- and 8-bit are supported, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

    @property
    def qmax(self) -> int:
        return INT4_MAX if self.bits == 4 else 127


def quantize_groupwise(
    weight: torch.Tensor, config: QuantConfig | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(out_features, in_features)`` to integer levels plus per-group scales.

    Returns ``(codes, scales)`` where ``codes`` matches the weight shape as ``int8`` and ``scales``
    is ``(out_features, n_groups)`` in fp16.

    **The axis is the part that goes wrong.** Groups run along the *input* dimension, so each group
    covers 128 weights that all multiply into the same output. Grouping along the output dimension
    instead produces a tensor of exactly the right shape, dequantizes without error, and destroys
    the model -- `BUGS.md` P-20, which is why the round-trip test asserts on error magnitude and
    not merely on shapes.
    """
    config = config or QuantConfig()
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    if in_features % config.group_size:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={config.group_size}; "
            "a ragged final group would need its own scale and is not supported"
        )

    n_groups = in_features // config.group_size
    grouped = weight.detach().float().reshape(out_features, n_groups, config.group_size)

    # Symmetric: the scale is set by the largest magnitude in the group.
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    # A group of exact zeros would divide by zero. Its codes are all zero regardless, so any
    # non-zero scale reconstructs it correctly -- 1.0 keeps the stored scale finite.
    scales = torch.where(amax > 0, amax / config.qmax, torch.ones_like(amax))

    codes = torch.round(grouped / scales).clamp(-config.qmax, config.qmax).to(torch.int8)
    return codes.reshape(out_features, in_features), scales.squeeze(-1).half()


def dequantize_groupwise(
    codes: torch.Tensor, scales: torch.Tensor, config: QuantConfig | None = None
) -> torch.Tensor:
    """Reconstruct an fp16 weight from codes and per-group scales."""
    config = config or QuantConfig()
    out_features, in_features = codes.shape
    n_groups = in_features // config.group_size

    grouped = codes.reshape(out_features, n_groups, config.group_size).to(torch.float16)
    return (grouped * scales.unsqueeze(-1)).reshape(out_features, in_features)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit codes two-per-byte along the input dimension.

    Without this step "INT4" still occupies a full byte per value and saves nothing -- the codes
    tensor is ``int8``. Packing is what turns the representation into an actual 4x reduction, and
    it is the only reason the memory claim is true.

    Values are stored biased by +8 into the range 0..15 so each nibble is unsigned and the low
    nibble cannot borrow the sign of the high one.
    """
    if codes.shape[-1] % 2:
        raise ValueError(f"input dimension {codes.shape[-1]} must be even to pack two per byte")

    biased = (codes.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    low, high = biased[..., 0::2], biased[..., 1::2]
    return (low | (high << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`, restoring signed codes as ``int8``."""
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8

    out = torch.empty(
        (*packed.shape[:-1], in_features), dtype=torch.int8, device=packed.device
    )
    out[..., 0::2] = low.to(torch.int8)
    out[..., 1::2] = high.to(torch.int8)
    return out


def quantization_error(weight: torch.Tensor, config: QuantConfig | None = None) -> dict[str, float]:
    """Round-trip error for one weight matrix.

    Run per layer *before* any end-to-end quality evaluation. If the axis is wrong or the group
    size is nonsense, this shows it in milliseconds -- whereas an end-to-end eval shows only that
    quality collapsed, with no clue as to which of a hundred layers did it (``BUGS.md`` P-20).
    """
    config = config or QuantConfig()
    codes, scales = quantize_groupwise(weight, config)
    restored = dequantize_groupwise(codes, scales, config).float()
    original = weight.detach().float()

    error = (restored - original).abs()
    denom = original.abs().mean().clamp(min=1e-12)

    # Per-output-row error, and specifically the WORST row. Tensor-wide averages are dominated by
    # large-magnitude weights, and the damage a wrong grouping axis does lands on the *small* ones
    # -- a group spanning rows of wildly different magnitude crushes the quiet rows to zero while
    # the loud ones quantize fine. Measured: a deliberately flipped axis moves the tensor-wide
    # figure by 19% and the worst row by orders of magnitude. The aggregate is nearly blind to the
    # failure it is supposed to catch (``BUGS.md`` P-20).
    row_denom = original.abs().mean(dim=1).clamp(min=1e-12)
    row_relative = error.mean(dim=1) / row_denom

    return {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "relative_error": float(error.mean() / denom),
        "max_row_relative_error": float(row_relative.max()),
        "signal_to_noise_db": float(
            10
            * torch.log10(
                original.pow(2).mean() / (restored - original).pow(2).mean().clamp(min=1e-20)
            )
        ),
    }
