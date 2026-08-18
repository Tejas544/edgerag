"""Document and query embeddings for retrieval. Phase 7, and a decision worth stating explicitly
(``CONTEXT.md`` D22) rather than let PLAN.md's sketch stand in for it.

**Image embeddings reuse the VLM's own vision tower** (D6): a separate SigLIP-SO400M would cost
~800 MB fp16 against a 4 GB budget, for a capability already resident once the decoder is loaded.
:func:`embed_image` calls the same :func:`~edgerag.core.model.encode_images_chunked` path the RAG
prompt path uses, then mean-pools every sub-image and every token into one fixed-size vector.

**There is no query image**, and this matters more than it looks like it should. DocVQA-style
retrieval is a text question against a corpus of document *pages* -- the question never carries an
image of its own. PLAN.md's original sketch ("image embeddings ... text embeddings ... hybrid
scoring") does not resolve how a text-only query is supposed to be compared against an image
vector at all, and the naive answer -- score a query against a document's own image and call
whatever comes back "image similarity" -- is circular for a self-retrieval test and meaningless
for a real one, since nothing ties the *question's* meaning to that image except through the
document it happens to be a page of. Building that and shipping it untested would have looked
correct and measured nothing.

**The fix: project the query's text into the same space the connector projects images into, via
the decoder's own ``embed_tokens``.** ``embed_tokens`` and the connector output are both
``hidden_size``-dimensional because both feed the same residual stream during generation -- the
model has *some* reason to relate them, even though neither was trained with a contrastive
retrieval objective the way CLIP's image/text towers are. Whether that weak relationship is worth
anything for ranking is exactly what :func:`edgerag.retrieval.index.recall_at_k` measures, against
a text-only control, rather than assumes. Zero new cost either way: ``embed_tokens`` is already
resident as part of the decoder that generates the answer.

**Text scoring is sparse (TF-IDF over the OCR field), not a second dense encoder.** A from-scratch
TF-IDF over ``CorpusDoc.text`` needs no new dependency and no second resident model. It also
degrades honestly: 112 of 362 corpus documents ship no OCR text at all (DocVQA scans with no
Textract sidecar), and a TF-IDF vector of an empty string is correctly the zero vector -- those
documents are found by the image-space score alone, or not at all, and the recall harness reports
which.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from edgerag.core.model import EdgeRagDecoder, encode_images_chunked

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run tokens. No stemming -- a corpus this size doesn't need it, and
    a stemmer is exactly the kind of dependency this module exists to avoid."""
    return _TOKEN_RE.findall(text.lower())


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class TfidfVectorizer:
    """TF-IDF over a fixed vocabulary, fit once at index-build time.

    Smoothed IDF (``sklearn``'s convention: ``log((1+N)/(1+df)) + 1``) so a term appearing in
    every document gets a positive weight rather than exactly zero -- a corpus-wide word should
    contribute a little, not vanish and take the whole query with it if that word dominates.
    """

    vocab: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, documents: list[str]) -> TfidfVectorizer:
        doc_freq: dict[str, int] = {}
        for doc in documents:
            for term in set(_tokenize(doc)):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self.vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        n_docs = len(documents)
        self.idf = np.array(
            [math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0 for term in self.vocab],
            dtype=np.float32,
        )
        return self

    def transform(self, text: str) -> np.ndarray:
        """L2-normalized TF-IDF vector. An empty or out-of-vocabulary text is the zero vector --
        not an error, and not a NaN from dividing by a zero norm."""
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens or not self.vocab:
            return vec

        counts: dict[int, int] = {}
        for term in tokens:
            idx = self.vocab.get(term)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1

        for idx, count in counts.items():
            vec[idx] = count * self.idf[idx]

        return _l2_normalize(vec)


@torch.inference_mode()
def embed_image(
    hf_model: torch.nn.Module, processor: Any, image: Image.Image, device: torch.device
) -> np.ndarray:
    """One L2-normalized vector per page, in the decoder's own hidden space.

    The processor's own image-splitting policy runs unmodified -- an infographic-sized page
    becomes several sub-images plus a global thumbnail, exactly as it does when this same image
    heads into a RAG prompt (``CONTEXT.md`` D12). Mean-pooling over sub-images and tokens is what
    makes the output shape independent of how many tiles a given page happened to split into; two
    documents of different sizes still produce directly comparable vectors.
    """
    batch = processor(text="<image>", images=[image], return_tensors="pt")
    pixel_values = batch["pixel_values"].to(device)

    features = encode_images_chunked(hf_model, pixel_values)  # (n_sub, tokens, hidden)
    pooled = features.float().mean(dim=(0, 1))
    return _l2_normalize(pooled.cpu().numpy())


@torch.inference_mode()
def embed_query_for_image_space(
    decoder: EdgeRagDecoder, processor: Any, text: str, device: torch.device
) -> np.ndarray:
    """Project query text into the space :func:`embed_image` produces vectors in.

    A bag-of-embeddings mean over ``embed_tokens``, including whatever BOS/EOS the tokenizer adds
    -- they are the same tokens on every query and mostly cancel out in the mean, and stripping
    them is not worth the extra surface for a first pass. This needs only an embedding-table
    lookup, no transformer layers, so it costs nothing meaningful even at query volume.
    """
    ids = processor.tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        pooled = decoder.embed_tokens(ids).float().mean(dim=(0, 1))
    return _l2_normalize(pooled.cpu().numpy())
