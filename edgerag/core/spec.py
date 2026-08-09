"""Model introspection and KV-cache arithmetic.

``01_EDGERAG.md`` §7 question 1 is *"Compute the KV cache size for your model at seq_len 2048,
batch 8. Show the arithmetic."* That arithmetic lives here rather than in a notebook, so it is
version-controlled, unit-tested, and cannot drift from the model actually being served.

The formula (fp16, one sequence)::

    kv_bytes = 2 * L * H_kv * d_head * S * dtype_bytes
               ^   ^   ^      ^         ^
               |   |   |      |         sequence length
               |   |   |      head dim
               |   |   number of *key/value* heads (not query heads -- this is the GQA saving)
               |   layers
               K and V

The leading 2 is K-and-V, not a fudge factor. The distinction that matters is ``H_kv`` versus
``H_q``: under GQA the cache scales with key/value heads only, which is where the memory saving
comes from and why the ratio is worth reporting explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIB = 1024**2
GIB = 1024**3

DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 1,  # packed two-per-byte; see effective_weight_bytes
}


@dataclass(frozen=True)
class ModelSpec:
    """The subset of a VLM config that governs memory.

    Deliberately a flat, plain-data view rather than a wrapper around the HF config object: the
    paged allocator and the budget ledger should depend on five integers, not on whichever config
    class a given checkpoint happens to ship.
    """

    model_id: str
    model_type: str

    # Text decoder -- this is what the KV cache is made of.
    n_layers: int
    hidden_size: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float

    # Vision tower -- prefill-time compute, no KV state (CONTEXT.md D2).
    vision_layers: int
    vision_hidden: int
    vision_image_size: int
    vision_patch_size: int

    # Pixel-shuffle compression factor applied between tower and decoder.
    scale_factor: int
    image_token_id: int

    #: Read from ``text_config``, never from the composite config -- see ``BUGS.md`` L-01. The
    #: composite config ships 128002 (a Llama-3 leftover), which is out of range for a 49280-token
    #: vocabulary and would index off the end of the embedding table during padded batching.
    pad_token_id: int = 2

    # --- attention shape ---------------------------------------------------------------------

    def __post_init__(self) -> None:
        # BUGS.md L-01. A pad id outside the vocabulary is an out-of-bounds embedding lookup,
        # which on CUDA is an async device-side assert that poisons the context and reports a
        # traceback pointing somewhere unrelated. Fail here instead, at construction, in Python.
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError(
                f"{self.model_id}: pad_token_id={self.pad_token_id} is outside vocab_size="
                f"{self.vocab_size}. Read it from config.text_config, not the composite config "
                "(see BUGS.md L-01)."
            )

    @property
    def uses_gqa(self) -> bool:
        return self.n_kv_heads < self.n_q_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head. 1 means full MHA."""
        if self.n_q_heads % self.n_kv_heads:
            raise ValueError(
                f"{self.model_id}: {self.n_q_heads} query heads is not divisible by "
                f"{self.n_kv_heads} kv heads"
            )
        return self.n_q_heads // self.n_kv_heads

    # --- KV cache arithmetic -----------------------------------------------------------------

    def kv_bytes_per_token(self, dtype: str = "float16") -> int:
        """Bytes of KV cache one token occupies across the whole decoder stack."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * DTYPE_BYTES[dtype]

    def kv_bytes(self, seq_len: int, batch: int = 1, dtype: str = "float16") -> int:
        return self.kv_bytes_per_token(dtype) * seq_len * batch

    def kv_report(self, seq_len: int = 2048, batch: int = 8, dtype: str = "float16") -> str:
        """The arithmetic, written out. This is the whiteboard answer to §7 question 1."""
        per_tok = self.kv_bytes_per_token(dtype)
        total = self.kv_bytes(seq_len, batch, dtype)
        attention = f"GQA {self.n_rep}:1" if self.uses_gqa else "MHA (no GQA saving)"
        return (
            f"{self.model_id} @ seq_len={seq_len}, batch={batch}, {dtype}\n"
            f"  per token = 2 (K,V) x {self.n_layers} layers x {self.n_kv_heads} kv-heads "
            f"x {self.head_dim} head-dim x {DTYPE_BYTES[dtype]} bytes\n"
            f"            = {per_tok:,} bytes ({per_tok / 1024:.1f} KiB/token)\n"
            f"  x {seq_len} tokens x {batch} sequences\n"
            f"            = {total:,} bytes ({total / GIB:.3f} GiB)\n"
            f"  attention  = {attention}"
        )

    # --- visual token arithmetic -------------------------------------------------------------

    @property
    def patches_per_side(self) -> int:
        return self.vision_image_size // self.vision_patch_size

    @property
    def visual_tokens_per_subimage(self) -> int:
        """Tokens the decoder sees per sub-image, after pixel shuffle.

        The tower emits ``patches_per_side ** 2`` patches; pixel shuffle folds a
        ``scale_factor x scale_factor`` neighbourhood into one token, dividing by
        ``scale_factor ** 2``.
        """
        return (self.patches_per_side**2) // (self.scale_factor**2)

    def visual_tokens_for(self, n_subimages: int, include_global: bool = True) -> int:
        """Total visual tokens for one image split into ``n_subimages`` tiles.

        SmolVLM prepends a downscaled view of the whole image alongside the tiles, so a 2x2 split
        costs five sub-images, not four.
        """
        return self.visual_tokens_per_subimage * (n_subimages + (1 if include_global else 0))

    # --- weights -----------------------------------------------------------------------------

    @staticmethod
    def effective_weight_bytes(n_params: int, dtype: str) -> int:
        """Bytes for ``n_params`` weights. INT4 packs two values per byte."""
        if dtype == "int4":
            return (n_params + 1) // 2
        return n_params * DTYPE_BYTES[dtype]

    # --- construction ------------------------------------------------------------------------

    @classmethod
    def from_hf_config(cls, model_id: str, config: Any) -> ModelSpec:
        """Build a spec from a HuggingFace config object.

        Reads through ``text_config`` / ``vision_config`` and falls back for fields that some
        checkpoints omit. ``rope_theta`` in particular is absent from several SmolVLM configs and
        silently defaults to 10000.0 in the modelling code -- reading it here rather than
        hardcoding it downstream means our RoPE cannot quietly disagree with HF's
        (``BUGS.md`` P-09).
        """
        txt = config.text_config
        vis = config.vision_config

        n_q = txt.num_attention_heads
        head_dim = getattr(txt, "head_dim", None) or txt.hidden_size // n_q

        return cls(
            model_id=model_id,
            model_type=config.model_type,
            n_layers=txt.num_hidden_layers,
            hidden_size=txt.hidden_size,
            n_q_heads=n_q,
            n_kv_heads=getattr(txt, "num_key_value_heads", None) or n_q,
            head_dim=head_dim,
            vocab_size=txt.vocab_size,
            max_position_embeddings=getattr(txt, "max_position_embeddings", 8192),
            rope_theta=float(getattr(txt, "rope_theta", None) or 10000.0),
            vision_layers=vis.num_hidden_layers,
            vision_hidden=vis.hidden_size,
            vision_image_size=vis.image_size,
            vision_patch_size=vis.patch_size,
            scale_factor=getattr(config, "scale_factor", 1),
            image_token_id=config.image_token_id,
            # text_config, deliberately -- BUGS.md L-01.
            pad_token_id=int(getattr(txt, "pad_token_id", None) or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["uses_gqa"] = self.uses_gqa
        payload["n_rep"] = self.n_rep
        payload["kv_bytes_per_token_fp16"] = self.kv_bytes_per_token("float16")
        payload["visual_tokens_per_subimage"] = self.visual_tokens_per_subimage
        return payload
