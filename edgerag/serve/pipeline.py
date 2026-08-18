"""Question in, retrieved-and-merged prompt out. The RAG half of Phase 7's serving path.

This is the seam between :mod:`edgerag.retrieval` and :mod:`edgerag.serve`: it retrieves pages for
a question, renders them into the checkpoint's own chat template, runs the vision tower, and
merges the visual features into the embedding stream -- producing the ``prompt_embeds`` a
:class:`~edgerag.sched.request.Request` carries and the executor slices.

**Why the prompt is built here and not in the executor.** Retrieval and the vision tower are
*per-request, once* work; the executor runs *per-iteration, many times*. Putting the tower inside
the executor would re-encode the same pages on every chunk of a chunked prefill -- 14 times for a
7,000-token prompt on D18's default chunk size. Doing it once, here, at admission time, is the
only arrangement where chunked prefill and multimodal prompts are both correct.

**What this costs, honestly:** the vision tower runs on the caller's thread, not the engine's
worker thread. That is a deliberate trade and it is the one place P-17's rule is bent: encoding is
a bounded one-shot cost per request rather than the unbounded per-token stream the rule exists to
protect, and moving it to the worker would serialise every arriving request behind the currently
decoding one. If tower time ever shows up in p99 TTFT, the fix is a second thread for encoding,
not moving it into the decode loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from edgerag.core.model import encode_images_chunked, merge_image_features
from edgerag.core.spec import ModelSpec
from edgerag.retrieval.corpus import CorpusDoc
from edgerag.retrieval.index import FlatIndex
from edgerag.retrieval.trace import build_prompt_messages

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Retrieved pages per question. Matches the frozen trace (``build_trace --k 5``), so a served
#: request has the same shape as every benchmarked one.
DEFAULT_K = 5


@dataclass
class RetrievedPrompt:
    """Everything the scheduler and executor need for one retrieved, multimodal prompt."""

    token_ids: list[int]
    embeds: torch.Tensor
    doc_keys: list[str]

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


class RagPipeline:
    """Retrieval plus prompt assembly. Holds no request state; safe to call concurrently."""

    def __init__(
        self,
        index: FlatIndex,
        docs_by_key: dict[str, CorpusDoc],
        hf_model: torch.nn.Module,
        decoder: torch.nn.Module,
        processor: Any,
        spec: ModelSpec,
        device: torch.device,
        k: int = DEFAULT_K,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self.index = index
        self.docs_by_key = docs_by_key
        self.hf_model = hf_model
        self.decoder = decoder
        self.processor = processor
        self.spec = spec
        self.device = device
        self.k = k
        self.repo_root = repo_root

    def retrieve(self, question: str) -> list[CorpusDoc]:
        """Top-``k`` pages for a question.

        Text-only scoring by default (``FlatIndex``'s ``alpha``), because D22 measured the
        image-space query signal as noise -- 0.14x the text side's spread, with a mean per-query
        maximum indistinguishable from zero. Passing ``None`` for the image vector is therefore
        not a shortcut; it is the measured configuration.
        """
        keys = self.index.search(None, question, self.k)
        return [self.docs_by_key[key] for key in keys if key in self.docs_by_key]

    @torch.inference_mode()
    def build(self, question: str) -> RetrievedPrompt:
        """Retrieve, template, encode the pages, and merge them into the embedding stream."""
        docs = self.retrieve(question)
        prompt = self.processor.apply_chat_template(
            build_prompt_messages(docs, question), add_generation_prompt=True
        )

        images = [Image.open(self.repo_root / d.image_path).convert("RGB") for d in docs]
        try:
            batch = self.processor(text=prompt, images=images or None, return_tensors="pt")
        finally:
            for image in images:
                image.close()

        input_ids = batch["input_ids"].to(self.device)
        embeds = self.decoder.embed_tokens(input_ids)

        if "pixel_values" in batch:
            features = encode_images_chunked(self.hf_model, batch["pixel_values"].to(self.device))
            embeds = merge_image_features(
                input_ids, embeds, features, self.spec.image_token_id
            )

        return RetrievedPrompt(
            token_ids=input_ids[0].tolist(),
            embeds=embeds,
            doc_keys=[d.doc_key for d in docs],
        )
