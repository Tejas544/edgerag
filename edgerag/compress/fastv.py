"""FastV-style visual token pruning.

Phase 4. ``CONTEXT.md`` D5 chose FastV over ToMe because ToMe merges tokens *inside the ViT* --
it cuts vision-encoder compute, not KV cache -- and this project's binding constraint is KV bytes.
D11 and D15 then upgraded this from "differentiator" to the highest-leverage optimisation
available: **78% of KV bytes on this workload are visual**, and D14 showed batching cannot recover
what KV bandwidth costs.

The mechanism, from *"An Image is Worth 1/2 Tokens After Layer 2"* (arXiv:2403.06764): visual
tokens receive steeply decreasing attention after the first couple of decoder layers, so most can
be discarded with little effect on the answer. We score each visual token by the attention it
receives, keep the top fraction, and drop the rest for every layer beyond the scoring layer.

Three implementation choices that are not in the paper and matter here:

* **Pruning happens once, at the end of prefill** (D5), not per layer. Per-layer pruning makes the
  resident token set layer-dependent and therefore the block tables layer-dependent -- real
  complexity for no additional result, since decode length dominates total KV residency.
* **Text tokens are never pruned.** They are a small minority here (25% of prefill) and carry the
  question. Pruning them trades the thing being asked for memory that visual tokens can supply
  more cheaply.
* **Kept tokens retain their original positions.** RoPE is applied before the cache write, so a
  surviving token keeps the rotation it was computed with. The pruned sequence is
  position-*sparse*, not renumbered -- renumbering would silently change every relative distance
  the model was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: FastV's published choice. Layers 0-1 attend broadly; the collapse in visual attention starts
#: after layer 2, which is why scoring earlier prunes on noise.
DEFAULT_SCORE_LAYER = 2


@dataclass(frozen=True)
class FastVConfig:
    """Pruning configuration.

    ``keep_ratio`` is the fraction of *visual* tokens retained. 1.0 disables pruning entirely,
    which is the ablation baseline and must be exactly a no-op rather than an approximation.
    """

    keep_ratio: float = 0.5
    score_layer: int = DEFAULT_SCORE_LAYER
    protect_last_n: int = 1
    #: "last_row" (FastV paper: predicts what decode will attend to) or "mean" (more signal, but
    #: scores what the prompt attended to). Swept in the Phase 4 ablation.
    score_mode: str = "last_row"

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.score_layer < 0:
            raise ValueError(f"score_layer must be non-negative, got {self.score_layer}")

    @property
    def enabled(self) -> bool:
        return self.keep_ratio < 1.0


def visual_token_scores(
    attn_weights: torch.Tensor,
    visual_mask: torch.Tensor,
    mode: str = "last_row",
) -> torch.Tensor:
    """Attention received by each token. ``(batch, heads, queries, keys)`` -> ``(batch, seq)``.

    Non-visual positions score ``-inf`` so a top-k over all positions can never select one.

    **Two defensible scoring modes, and the choice is not obvious:**

    ``last_row`` (default, and what the FastV paper uses)
        Score by the attention the *final* prompt token pays to each key. The argument is that
        every token generated during decode is a query resembling that one, so the last row is the
        best available predictor of what decode will actually attend to. Pruning is a bet about
        the future, and this bets using the most similar observation available.

    ``mean``
        Average over every attending query. Uses far more signal and is less sensitive to one
        unusual final token, but it scores tokens by how much the *prompt* attended to them, which
        is a different question from what *generation* will need.

    **The normalisation in ``mean`` is the subtle part.** Under causal masking token *j* is only
    visible to queries *j..S-1*, so a plain column mean over all S queries systematically
    under-scores late tokens -- and in a RAG prompt the late tokens are the most recently retrieved
    document. Dividing by the number of queries that could attend removes that artifact. It does
    *not* flatten the scores completely, and should not: early keys legitimately receive more
    attention per query because they compete with fewer rivals in early rows.
    """
    if attn_weights.dim() != 4:
        raise ValueError(f"expected (batch, heads, queries, keys), got {tuple(attn_weights.shape)}")
    if mode not in ("last_row", "mean"):
        raise ValueError(f"unknown scoring mode {mode!r}; expected 'last_row' or 'mean'")

    if mode == "last_row":
        scores = attn_weights[:, :, -1, :].float().mean(dim=1)
    else:
        seq_len = attn_weights.shape[-1]
        received = attn_weights.float().sum(dim=1).sum(dim=1)
        n_heads = attn_weights.shape[1]
        n_queries = attn_weights.shape[2]
        if n_queries == seq_len:
            eligible = torch.arange(
                seq_len, 0, -1, device=attn_weights.device, dtype=torch.float32
            )
        else:
            # Decode or chunked prefill: every key was visible to every query present.
            eligible = torch.full((seq_len,), float(n_queries), device=attn_weights.device)
        scores = received / (eligible * n_heads)

    return scores.masked_fill(~visual_mask.bool(), float("-inf"))


def select_kept_indices(
    scores: torch.Tensor,
    visual_mask: torch.Tensor,
    config: FastVConfig,
) -> torch.Tensor:
    """Sorted indices of the tokens that survive pruning, for a single sequence.

    Every non-visual token survives, plus the top ``keep_ratio`` of visual tokens by score, plus
    the final ``protect_last_n`` positions unconditionally.

    **The last position is protected for a mechanical reason, not a heuristic one:** generation
    reads logits from it. If the final token were pruned the model would have no position to
    generate from. It is nearly always text, but a prompt ending in an image would break without
    this.

    Returns ascending indices, so the pruned sequence keeps its original order and a standard
    causal mask remains valid over it.
    """
    if scores.dim() != 1:
        raise ValueError(f"expected scores for one sequence, got {tuple(scores.shape)}")

    seq_len = scores.shape[0]
    visual = visual_mask.bool()
    n_visual = int(visual.sum())

    keep = ~visual  # all text
    if config.protect_last_n > 0:
        keep[max(0, seq_len - config.protect_last_n) :] = True

    if n_visual and config.enabled:
        n_keep_visual = max(1, round(n_visual * config.keep_ratio))
        top = torch.topk(scores, k=min(n_keep_visual, n_visual)).indices
        keep[top] = True
    elif n_visual:
        keep |= visual  # keep_ratio == 1.0 must be an exact no-op

    return torch.nonzero(keep, as_tuple=False).flatten()


def uniform_stride_indices(
    visual_mask: torch.Tensor, config: FastVConfig
) -> torch.Tensor:
    """Attention-free baseline: keep every n-th visual token.

    The honest control for the Phase 4 sweep. If attention-based selection does not beat evenly
    spaced selection at the same ratio, then the *scoring* contributes nothing and the result is
    "visual tokens are redundant", not "FastV works" -- a distinction the published curve should
    not leave ambiguous.
    """
    seq_len = visual_mask.shape[0]
    visual = visual_mask.bool()
    keep = ~visual
    if config.protect_last_n > 0:
        keep[max(0, seq_len - config.protect_last_n) :] = True

    visual_positions = torch.nonzero(visual, as_tuple=False).flatten()
    if visual_positions.numel():
        if config.enabled:
            n_keep = max(1, round(visual_positions.numel() * config.keep_ratio))
            picks = torch.linspace(
                0, visual_positions.numel() - 1, steps=n_keep, device=visual_mask.device
            ).round().long()
            keep[visual_positions[picks]] = True
        else:
            keep |= visual

    return torch.nonzero(keep, as_tuple=False).flatten()


def build_visual_mask(
    input_ids: torch.Tensor, image_token_id: int
) -> torch.Tensor:
    """``True`` where a position holds an image token."""
    return input_ids == image_token_id


class FastVCompressor:
    """Selection strategy handed to :meth:`EdgeRagDecoder.forward`.

    Thin by design: the decoder owns *when* pruning happens and the cache owns *where* the
    survivors are stored, so this owns only *which* tokens survive. That split is what makes the
    uniform-stride control a one-line substitution rather than a parallel implementation.
    """

    def __init__(self, config: FastVConfig | None = None, strategy: str = "attention") -> None:
        self.config = config or FastVConfig()
        if strategy not in ("attention", "uniform"):
            raise ValueError(f"unknown strategy {strategy!r}; expected 'attention' or 'uniform'")
        self.strategy = strategy
        self.last_kept: torch.Tensor | None = None

    def select(self, attn_weights: torch.Tensor, visual_mask: torch.Tensor) -> torch.Tensor:
        """Indices surviving the cut, for one sequence.

        ``attn_weights`` is ``(heads, queries, keys)`` -- one sequence's weights, batch dimension
        already removed by the caller.
        """
        if self.strategy == "uniform":
            kept = uniform_stride_indices(visual_mask, self.config)
        else:
            scores = visual_token_scores(
                attn_weights.unsqueeze(0), visual_mask.unsqueeze(0), self.config.score_mode
            )[0]
            kept = select_kept_indices(scores, visual_mask, self.config)
        self.last_kept = kept
        return kept
