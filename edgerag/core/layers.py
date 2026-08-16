"""Decoder building blocks, written from scratch.

These are ours (``CONTEXT.md`` D2). They are deliberately numerically faithful to HuggingFace's
*eager* Llama path, because ``tests/test_equivalence.py`` compares against it and a tolerance wide
enough to absorb an implementation difference is also wide enough to hide a bug.

Three places where the obvious implementation is subtly wrong, each matched deliberately:

* **RMSNorm computes variance in fp32**, then casts back *before* the weight multiply. Doing the
  whole thing in fp16 drifts, and the drift grows with hidden size.
* **RoPE computes cos/sin in fp32** regardless of model dtype, then casts. fp16 sin/cos at
  position 6000 loses enough precision to move logits.
* **Softmax runs in fp32** and casts back to the query dtype. This is `BUGS.md` P-07: doing it in
  fp16 makes the equivalence test fail at long context for reasons that are not a bug, and burns
  an evening.
"""

from __future__ import annotations

import torch
from torch import nn

from edgerag.core.spec import ModelSpec


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    No mean subtraction and no bias -- that is the whole difference from LayerNorm, and it is why
    it is cheaper. The fp32 upcast is not optional: ``x.pow(2).mean(-1)`` over 2048 fp16 values
    accumulates visible error.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # Cast back *first*, then apply the weight -- matching HF. Multiplying in fp32 and casting
        # afterwards gives a different rounding and shows up as drift in the equivalence test.
        return self.weight * x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings.

    ``theta`` must come from the checkpoint, never from a default -- see ``BUGS.md`` B-02. The
    SmolVLM checkpoints use 100000 and 130000; the library default of 10000 produces a model that
    runs and is wrong.
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``(batch, seq, head_dim)``.

        Computed in fp32 and cast at the end. ``position_ids`` carries *logical* positions, which
        matters once paged attention and prefix sharing arrive: the physical block slot a token
        lives in is meaningless to RoPE (``BUGS.md`` P-03).
        """
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq.to(x.device) @ positions.to(x.device)).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first, negated.

    Note this is the *halves* convention (``[-x2, x1]``), not the interleaved-pairs convention
    used by some other implementations. Mixing the two produces a model that runs and is wrong --
    the same failure class as B-02.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key.

    ``q``/``k`` are ``(batch, heads, seq, head_dim)`` and ``cos``/``sin`` are ``(batch, seq,
    head_dim)``, so both gain a head axis at dim 1 to broadcast across heads.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads to match query heads under GQA.

    ``(batch, kv_heads, seq, head_dim) -> (batch, kv_heads * n_rep, seq, head_dim)``, where each
    kv head is repeated ``n_rep`` times *consecutively*.

    The consecutive ordering is the point. ``torch.repeat`` tiles the whole axis
    (``h0 h1 h2 h0 h1 h2``) while ``repeat_interleave`` repeats each element
    (``h0 h0 h0 h1 h1 h1``). Only the latter matches how query heads are grouped; the former gives
    a model that runs and produces grammatical, wrong text -- ``BUGS.md`` P-08.

    ``n_rep == 1`` (full MHA, which the 2.2B headline model uses) returns the input untouched.
    """
    if n_rep == 1:
        return x
    batch, kv_heads, seq, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq, head_dim)
    return x.reshape(batch, kv_heads * n_rep, seq, head_dim)


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scaling: float,
    n_rep: int,
) -> torch.Tensor:
    """Memory-efficient attention. The production path.

    :func:`eager_attention` materialises the full ``(batch, heads, queries, keys)`` score matrix.
    That is fine at test sizes and **catastrophic at real ones**: a 6,800-token prefill with 32
    heads needs 2.96 GiB for the fp16 scores and another 5.92 GiB for the fp32 softmax copy, so a
    single layer peaks near 9 GiB. It is why the Phase 4 quality run OOM'd on a 14.6 GiB T4
    (``BUGS.md`` B-05) while HuggingFace's own baseline, which uses SDPA, ran fine.

    ``F.scaled_dot_product_attention`` never forms that matrix. Eager is kept because the
    equivalence tests compare against HF's eager path and because it is readable, but nothing
    ships on it.
    """
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)
    if attn_mask is not None:
        # Trim to the actual key count, exactly as the eager path does. Layers above FastV's cut
        # hold fewer tokens than the mask was built for (CONTEXT.md D5), and SDPA -- unlike a
        # broadcast add -- rejects the mismatch rather than absorbing it.
        attn_mask = attn_mask[..., : key.shape[-2]]
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, scale=scaling
    )
    return out.transpose(1, 2).contiguous()


def last_row_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scaling: float,
    n_rep: int,
) -> torch.Tensor:
    """Attention paid by the **final** query only, as ``(batch, heads, 1, keys)``.

    This is all FastV's default ``last_row`` scoring needs (``CONTEXT.md`` D17), and it costs
    ``heads x keys`` instead of ``heads x queries x keys`` -- about 870 KB at 6,800 tokens versus
    9 GiB. Computing the whole matrix to read one row of it is what made visual-token pruning
    impossible to measure at realistic prompt lengths.
    """
    key = repeat_kv(key, n_rep)
    last_query = query[:, :, -1:, :]
    scores = torch.matmul(last_query, key.transpose(2, 3)) * scaling
    if attn_mask is not None:
        scores = scores + attn_mask[..., -1:, : key.shape[-2]]
    return torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)


def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scaling: float,
    n_rep: int,
    need_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scaled dot-product attention, written out.

    Deliberately eager rather than ``F.scaled_dot_product_attention``: this is the reference the
    paged implementation is validated against in Phase 3, and a reference whose internals are
    opaque cannot be reasoned about when the paged version disagrees with it by 1e-3.

    The mask is *additive* (0 to keep, large negative to drop), not boolean. It must use
    ``finfo(dtype).min`` rather than ``-inf``: a fully-masked row -- which a padded sequence
    produces -- softmaxes ``-inf`` to NaN, and that NaN then propagates across the whole batch
    (``BUGS.md`` P-10).
    """
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask[..., : key.shape[-2]]

    # fp32 softmax, then back. BUGS.md P-07.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    # Returned only when asked: the weights are (B, H, Q, K) and materialising them for every
    # layer would cost more memory than the KV cache they are used to shrink.
    return out, (attn_weights if need_weights else None)


def build_causal_mask(
    seq_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Additive causal mask of shape ``(batch, 1, seq_len, past_len + seq_len)``.

    Query at logical position ``past_len + i`` may attend to keys ``0 .. past_len + i``. During
    decode ``seq_len == 1`` and the row is all-visible, which is why a mask bug can pass every
    single-token test and only appear during prefill.

    ``padding_mask`` is ``(batch, past_len + seq_len)`` with 1 for real tokens. Left-padded rows
    therefore mask their leading positions for every query.
    """
    min_value = torch.finfo(dtype).min
    total = past_len + seq_len

    q_pos = torch.arange(past_len, total, device=device).unsqueeze(1)
    k_pos = torch.arange(total, device=device).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, 0.0, min_value).to(dtype)
    mask = mask[None, None, :, :]

    if padding_mask is not None:
        pad = torch.where(padding_mask.bool(), 0.0, min_value).to(dtype)
        mask = mask + pad[:, None, None, :]
    return mask


class Attention(nn.Module):
    """Grouped-query attention over a KV cache."""

    def __init__(
        self,
        spec: ModelSpec,
        layer_idx: int,
        linear_cls: type[nn.Module],
        use_eager: bool = False,
    ) -> None:
        super().__init__()
        #: Eager materialises the full score matrix -- O(queries x keys) memory. Default off; the
        #: equivalence tests turn it on to compare against HuggingFace's eager path.
        self.use_eager = use_eager
        self.layer_idx = layer_idx
        self.n_q_heads = spec.n_q_heads
        self.n_kv_heads = spec.n_kv_heads
        self.head_dim = spec.head_dim
        self.n_rep = spec.n_rep
        self.scaling = spec.head_dim**-0.5

        q_out = spec.n_q_heads * spec.head_dim
        kv_out = spec.n_kv_heads * spec.head_dim
        self.q_proj = linear_cls(spec.hidden_size, q_out, bias=False)
        self.k_proj = linear_cls(spec.hidden_size, kv_out, bias=False)
        self.v_proj = linear_cls(spec.hidden_size, kv_out, bias=False)
        self.o_proj = linear_cls(q_out, spec.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cache: object | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, _ = hidden_states.shape
        shape = (batch, seq_len, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        # RoPE is applied *before* the cache write, so the cache holds rotated keys. Rotating on
        # read instead would mean re-rotating the whole prefix at every decode step.
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)

        if self.use_eager:
            # Test-only path: bit-comparable with HuggingFace's eager implementation, and the
            # reference the SDPA path is validated against.
            out, weights = eager_attention(
                q, k, v, attn_mask, self.scaling, self.n_rep, need_weights=need_weights
            )
        else:
            out = sdpa_attention(q, k, v, attn_mask, self.scaling, self.n_rep)
            # Only the last row is ever read, so only the last row is computed.
            weights = (
                last_row_scores(q, k, attn_mask, self.scaling, self.n_rep)
                if need_weights
                else None
            )
        return self.o_proj(out.reshape(batch, seq_len, -1)), weights


class SwiGLUMLP(nn.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``.

    Three projections rather than two. ``gate`` and ``up`` are separate matrices of identical
    shape, and swapping them is a silent error -- both directions produce valid shapes.
    """

    def __init__(self, spec: ModelSpec, linear_cls: type[nn.Module]) -> None:
        super().__init__()
        if spec.hidden_act != "silu":
            raise ValueError(f"unsupported activation {spec.hidden_act!r}; expected silu")
        self.gate_proj = linear_cls(spec.hidden_size, spec.intermediate_size, bias=False)
        self.up_proj = linear_cls(spec.hidden_size, spec.intermediate_size, bias=False)
        self.down_proj = linear_cls(spec.intermediate_size, spec.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: norm -> attn -> residual, norm -> mlp -> residual."""

    def __init__(
        self,
        spec: ModelSpec,
        layer_idx: int,
        linear_cls: type[nn.Module],
        use_eager: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(spec, layer_idx, linear_cls, use_eager=use_eager)
        self.mlp = SwiGLUMLP(spec, linear_cls)
        self.input_layernorm = RMSNorm(spec.hidden_size, spec.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(spec.hidden_size, spec.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cache: object | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.self_attn(
            self.input_layernorm(hidden_states), cos, sin, attn_mask, cache, need_weights
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, weights
