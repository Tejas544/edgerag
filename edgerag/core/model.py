"""Our decoder stack, and the boundary where HuggingFace weights become ours.

``CONTEXT.md`` D2 draws the line: everything from the decoder's input embeddings onward is written
here. The vision tower forward stays with HuggingFace, because it is a one-shot prefill cost with
no KV state and therefore not part of the memory-management surface this project is about.

Token interleaving (:func:`merge_image_features`) is ours despite being small -- it is the
multimodal boundary, and it is where an off-by-one silently shifts every embedding after the first
image (``BUGS.md`` P-11).
"""

from __future__ import annotations

import torch
from torch import nn

from edgerag.core.layers import DecoderLayer, RMSNorm, RotaryEmbedding, build_causal_mask
from edgerag.core.linear import FP16Linear, LinearBase
from edgerag.core.spec import ModelSpec

#: D12 -- sub-images per vision-tower forward pass. A k=5 document prompt splits into ~65
#: sub-images, and pushing all of them through at once peaked at ~2 GiB of transient activation on
#: the 256M fixture. Sub-images do not attend to each other inside the tower, so chunking is
#: exactly equivalent rather than an approximation.
DEFAULT_VISION_CHUNK = 8


class EdgeRagDecoder(nn.Module):
    """The language decoder: embeddings -> N transformer blocks -> norm -> lm_head."""

    def __init__(self, spec: ModelSpec, linear_cls: type[LinearBase] = FP16Linear) -> None:
        super().__init__()
        self.spec = spec
        self.embed_tokens = nn.Embedding(spec.vocab_size, spec.hidden_size, spec.pad_token_id)
        self.layers = nn.ModuleList(
            [DecoderLayer(spec, i, linear_cls) for i in range(spec.n_layers)]
        )
        self.norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps)
        self.rotary = RotaryEmbedding(spec.head_dim, spec.rope_theta)
        self.lm_head = linear_cls(spec.hidden_size, spec.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        cache: object | None = None,
    ) -> torch.Tensor:
        """Return logits of shape ``(batch, seq_len, vocab_size)``.

        ``inputs_embeds`` is how multimodal input arrives: image features are merged into the
        embedding stream before this is called, so the decoder itself never knows about images.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")

        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, seq_len, _ = hidden.shape

        past_len = getattr(cache, "seq_len", 0) if cache is not None else 0
        if position_ids is None:
            # Logical positions continue from the cache, never restart at 0. Restarting is
            # BUGS.md P-09: the first generated token is fine and everything after degrades.
            position_ids = (
                torch.arange(past_len, past_len + seq_len, device=hidden.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )

        cos, sin = self.rotary(hidden, position_ids)
        attn_mask = build_causal_mask(
            seq_len, past_len, hidden.dtype, hidden.device, padding_mask
        )

        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attn_mask, cache)

        return self.lm_head(self.norm(hidden))


def merge_image_features(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_features: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Scatter vision features into the positions marked by ``<image>`` tokens.

    ``image_features`` is ``(n_subimages, tokens_per_subimage, hidden)`` and is written into the
    embedding stream in row-major order: sub-image 0's tokens fill the first run of image-token
    slots, and so on.

    The count check is the whole point. If the processor emitted a different number of image
    tokens than the tower produced features -- because an image was dropped as padding
    (``BUGS.md`` P-26), or the placeholder expansion disagrees with the config
    (``BUGS.md`` P-11) -- then every embedding after the mismatch shifts by one and the model
    produces confident nonsense. Torch would not complain: the scatter shapes still broadcast.
    """
    image_mask = input_ids == image_token_id
    n_slots = int(image_mask.sum())
    n_features = image_features.shape[0] * image_features.shape[1]

    if n_slots != n_features:
        raise ValueError(
            f"{n_slots} <image> token slots but {n_features} image features "
            f"({image_features.shape[0]} sub-images x {image_features.shape[1]} tokens). "
            "A mismatch shifts every subsequent embedding -- see BUGS.md P-11 and P-26."
        )
    if n_slots == 0:
        return inputs_embeds

    flat = image_features.reshape(-1, image_features.shape[-1]).to(inputs_embeds.dtype)
    merged = inputs_embeds.clone()
    merged[image_mask] = flat
    return merged


@torch.inference_mode()
def encode_images_chunked(
    hf_model: nn.Module,
    pixel_values: torch.Tensor,
    chunk_size: int = DEFAULT_VISION_CHUNK,
) -> torch.Tensor:
    """Run the HF vision tower and connector in chunks of sub-images (``CONTEXT.md`` D12).

    ``pixel_values`` is ``(batch, n_subimages, channels, h, w)``; returns
    ``(n_real_subimages, tokens_per_subimage, text_hidden)``.

    Chunking bounds the transient activation peak, which on a k=5 document prompt is the single
    largest term in the memory budget -- larger than the weights and the KV cache combined on the
    fixture. It is exactly equivalent rather than approximate, because no attention crosses
    sub-image boundaries inside the tower.

    All-zero padding sub-images are dropped first, matching HF's own ``get_image_features``.
    """
    inner = hf_model.model if hasattr(hf_model, "model") else hf_model
    vision_model = inner.vision_model
    connector = inner.connector

    batch, n_sub = pixel_values.shape[0], pixel_values.shape[1]
    flat = pixel_values.reshape(batch * n_sub, *pixel_values.shape[2:]).to(
        dtype=next(vision_model.parameters()).dtype
    )

    per_image = flat.shape[1:].numel()
    real = (flat == 0.0).sum(dim=(-1, -2, -3)) != per_image
    real[0] |= ~torch.any(real)
    flat = flat[real].contiguous()

    outputs = []
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        patch_mask = torch.ones(
            (chunk.shape[0], chunk.shape[2], chunk.shape[3]),
            dtype=torch.bool,
            device=chunk.device,
        )
        hidden = vision_model(chunk, patch_attention_mask=patch_mask).last_hidden_state
        outputs.append(connector(hidden))

    return torch.cat(outputs, dim=0)


def load_from_hf(
    spec: ModelSpec,
    hf_model: nn.Module,
    linear_cls: type[LinearBase] = FP16Linear,
) -> EdgeRagDecoder:
    """Copy checkpoint weights out of the HF module tree into our decoder.

    Deliberately explicit rather than a ``load_state_dict`` with a name-mapping dict. A key that
    fails to map silently leaves a randomly-initialised layer in place, and the model then produces
    plausible-looking output -- ``strict=True`` catches missing keys but not *mis-mapped* ones.
    Reaching for each tensor by attribute means a rename upstream raises ``AttributeError`` here
    rather than degrading quality somewhere unmeasurable.
    """
    inner = hf_model.model if hasattr(hf_model, "model") else hf_model
    text = inner.text_model

    ours = EdgeRagDecoder(spec, linear_cls)
    ours = ours.to(dtype=next(text.parameters()).dtype, device=next(text.parameters()).device)

    with torch.no_grad():
        ours.embed_tokens.weight.copy_(text.embed_tokens.weight)
        ours.norm.weight.copy_(text.norm.weight)
        ours.lm_head.load_weight(hf_model.lm_head.weight)

        for i, (dst, src) in enumerate(zip(ours.layers, text.layers, strict=True)):
            dst.self_attn.q_proj.load_weight(src.self_attn.q_proj.weight)
            dst.self_attn.k_proj.load_weight(src.self_attn.k_proj.weight)
            dst.self_attn.v_proj.load_weight(src.self_attn.v_proj.weight)
            dst.self_attn.o_proj.load_weight(src.self_attn.o_proj.weight)
            dst.mlp.gate_proj.load_weight(src.mlp.gate_proj.weight)
            dst.mlp.up_proj.load_weight(src.mlp.up_proj.weight)
            dst.mlp.down_proj.load_weight(src.mlp.down_proj.weight)
            dst.input_layernorm.weight.copy_(src.input_layernorm.weight)
            dst.post_attention_layernorm.weight.copy_(src.post_attention_layernorm.weight)
            assert dst.self_attn.layer_idx == i

    ours.eval()
    return ours
