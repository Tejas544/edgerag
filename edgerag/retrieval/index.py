"""Retrieval index. Phase 7 (``PLAN.md``): an interface with a flat implementation, deliberately --
**this is the seam VecCore drops into later** (``PLAN.md`` repo layout note), the same relationship
``LinearBase``/``QuantLinear`` had before Phase 6 (``CONTEXT.md`` D7): build the interface while
there is only one implementation, so a second one is a config flag rather than a refactor.

362 documents does not need an approximate index -- exhaustive cosine similarity over 362 vectors
is sub-millisecond, and introducing ANN machinery (product quantization, HNSW) to solve a problem
this corpus does not have would be exactly the kind of premature abstraction
``00_FOUNDATIONS.md`` warns against. ``FlatIndex`` is not a placeholder for a "real" index; at
this corpus size it is the right index, and stays the right one until the corpus size argues
otherwise.

**Hybrid scoring, weighted linearly:** ``score = alpha * image_similarity + (1 - alpha) *
text_similarity``. Not learned, not softmax-combined -- a corpus of 362 documents and 650 queries
is not enough to fit a combiner without overfitting the eval split, and a linear blend is legible
enough to reason about directly (D8: retrieval quality is tuned *late*, against a measurement, not
designed in from a paper).

**The image side is not a validated signal, and this module does not pretend otherwise.** See
``edgerag/retrieval/embed.py`` for why: the query's "image-space" vector comes from
``embed_tokens``, a layer trained for next-token prediction, not retrieval, projected into the
same space the connector emits for images. Whether that carries any real relevance signal is an
empirical question, so :func:`recall_at_k` always runs alongside :meth:`FlatIndex.search_text_only`
and :meth:`FlatIndex.search_image_only` -- the same "compare against a naive baseline" discipline
every other ablation in this project uses (FastV's ``attention`` vs ``uniform``, D20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from edgerag.retrieval.corpus import CorpusDoc
from edgerag.retrieval.embed import TfidfVectorizer

#: D8's default, unchallenged until the recall harness argues otherwise: weight image and text
#: evenly. 112 of 362 documents have no OCR text at all, so alpha=1.0 (text-only) would blind the
#: index to a third of the corpus, and alpha=0.0 (image-only) throws away exact keyword matches
#: InfographicVQA questions frequently quote verbatim.
DEFAULT_ALPHA = 0.5


class RetrievalIndex(Protocol):
    """What a retriever must do. ``VecCore`` implements this later without touching a caller."""

    def search(
        self, image_space_query: np.ndarray | None, query_text: str, k: int
    ) -> list[str]: ...


@dataclass
class FlatIndex:
    """Exhaustive cosine-similarity search over every corpus document.

    Built once via :meth:`build`, then queried repeatedly -- the TF-IDF vectorizer is fit on the
    corpus text at build time, exactly as it would be for a real offline index, so query-time text
    never influences the vocabulary a query is scored against.
    """

    doc_keys: list[str]
    image_vectors: np.ndarray  # (n_docs, hidden), L2-normalized, zero row if unavailable
    text_vectors: np.ndarray  # (n_docs, vocab_size), L2-normalized
    vectorizer: TfidfVectorizer
    alpha: float = DEFAULT_ALPHA

    @classmethod
    def build(
        cls,
        docs: list[CorpusDoc],
        image_embeddings: dict[str, np.ndarray],
        alpha: float = DEFAULT_ALPHA,
    ) -> FlatIndex:
        """Assemble the index from pre-computed image embeddings and the corpus's own OCR text.

        ``image_embeddings`` is computed separately (:func:`~edgerag.retrieval.embed.embed_image`
        needs a loaded vision tower; this classmethod does not), so building -- and testing -- the
        *text* half of the index never needs a model at all.
        """
        vectorizer = TfidfVectorizer().fit([d.text for d in docs])

        hidden = next(iter(image_embeddings.values())).shape[0] if image_embeddings else 0
        doc_keys = [d.doc_key for d in docs]
        image_vectors = np.zeros((len(docs), hidden), dtype=np.float32)
        text_vectors = np.zeros((len(docs), vectorizer.vocab_size), dtype=np.float32)

        for i, doc in enumerate(docs):
            if doc.doc_key in image_embeddings:
                image_vectors[i] = image_embeddings[doc.doc_key]
            text_vectors[i] = vectorizer.transform(doc.text)

        return cls(
            doc_keys=doc_keys,
            image_vectors=image_vectors,
            text_vectors=text_vectors,
            vectorizer=vectorizer,
            alpha=alpha,
        )

    def search(
        self, image_space_query: np.ndarray | None, query_text: str, k: int
    ) -> list[str]:
        """Top-``k`` document keys, ranked by the hybrid score.

        Vectors on both sides are L2-normalized, so a plain dot product against the matrix *is*
        cosine similarity -- no per-query renormalization needed. ``image_space_query=None``
        degrades to the text score alone rather than raising, the same "a partial signal is still
        a signal" reasoning :func:`~edgerag.retrieval.embed.embed_image` applies to documents with
        no OCR text.
        """
        text_sim = self.text_vectors @ self.vectorizer.transform(query_text)

        if image_space_query is not None and self.image_vectors.shape[1]:
            image_sim = self.image_vectors @ image_space_query
            score = self.alpha * image_sim + (1.0 - self.alpha) * text_sim
        else:
            score = text_sim

        return self._top_k(score, k)

    def search_image_only(self, image_space_query: np.ndarray, k: int) -> list[str]:
        """One half of the control :func:`recall_at_k` checks the hybrid score against."""
        return self._top_k(self.image_vectors @ image_space_query, k)

    def search_text_only(self, query_text: str, k: int) -> list[str]:
        """The other half of the same control."""
        return self._top_k(self.text_vectors @ self.vectorizer.transform(query_text), k)

    def _top_k(self, score: np.ndarray, k: int) -> list[str]:
        order = np.argsort(-score, kind="stable")[:k]
        return [self.doc_keys[i] for i in order]


def recall_at_k(
    index: FlatIndex, queries: list[Any], query_vectors: dict[str, np.ndarray], k: int
) -> dict[str, float]:
    """Recall@k for the hybrid score **and** both single-signal controls, in one pass.

    Returns all three so a hybrid number is never read alone: if ``hybrid`` does not beat
    ``max(text_only, image_only)``, the honest finding is "the blend adds nothing over its better
    half," not "hybrid retrieval works" (the same shape of caution D20 applied to FastV's
    ``attention`` vs ``uniform`` comparison).

    ``query_vectors`` maps ``query_id -> image-space query vector``
    (:func:`~edgerag.retrieval.embed.embed_query_for_image_space`), computed by the caller so this
    function stays pure numpy and needs no model to test.
    """
    hits = {"hybrid": 0, "text_only": 0, "image_only": 0}
    for query in queries:
        vec = query_vectors.get(query.query_id)
        if query.gold_doc_key in index.search(vec, query.question, k):
            hits["hybrid"] += 1
        if query.gold_doc_key in index.search_text_only(query.question, k):
            hits["text_only"] += 1
        if vec is not None and query.gold_doc_key in index.search_image_only(vec, k):
            hits["image_only"] += 1

    n = len(queries) or 1
    return {name: count / n for name, count in hits.items()}
