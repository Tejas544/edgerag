"""Phase 7: hybrid retrieval, and the honest question it exists to answer.

Pure-numpy tests first -- ``TfidfVectorizer`` and ``FlatIndex`` need no model and run in
milliseconds, the same discipline ``tests/test_quant.py`` uses for the same reason. The one
``slow`` test at the bottom is the real point of the file: it runs :func:`recall_at_k`'s three-way
comparison (hybrid vs text-only vs image-only) on the fixture, because ``embed.py``'s central claim
-- that projecting a query through ``embed_tokens`` gives *any* usable relevance signal against
image vectors -- is an empirical one and does not get to go untested.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgerag.retrieval.corpus import CorpusDoc
from edgerag.retrieval.embed import TfidfVectorizer
from edgerag.retrieval.index import FlatIndex, recall_at_k

# --- TfidfVectorizer -----------------------------------------------------------------------------


def test_identical_text_scores_maximum_similarity() -> None:
    vec = TfidfVectorizer().fit(["revenue grew by twelve percent", "unrelated filler document"])
    a = vec.transform("revenue grew by twelve percent")
    assert np.dot(a, a) == pytest.approx(1.0)


def test_disjoint_vocabulary_scores_zero() -> None:
    vec = TfidfVectorizer().fit(["apples oranges bananas", "trucks bridges tunnels"])
    a = vec.transform("apples oranges")
    b = vec.transform("trucks bridges")
    assert np.dot(a, b) == pytest.approx(0.0)


def test_empty_text_is_the_zero_vector_not_a_nan() -> None:
    vec = TfidfVectorizer().fit(["some document text", "another one"])
    result = vec.transform("")
    assert np.all(result == 0.0)
    assert not np.any(np.isnan(result))


def test_out_of_vocabulary_query_is_the_zero_vector() -> None:
    vec = TfidfVectorizer().fit(["alpha beta gamma"])
    result = vec.transform("zzz yyy xxx")
    assert np.all(result == 0.0)


def test_a_term_in_every_document_still_gets_positive_weight() -> None:
    """Smoothed IDF: a corpus-wide word contributes a little, it does not vanish to zero."""
    vec = TfidfVectorizer().fit(["common word alpha", "common word beta", "common word gamma"])
    assert vec.idf[vec.vocab["common"]] > 0.0


def test_case_and_punctuation_are_normalised() -> None:
    vec = TfidfVectorizer().fit(["Revenue Grew!", "filler"])
    a = vec.transform("REVENUE grew.")
    b = vec.transform("revenue grew")
    np.testing.assert_allclose(a, b)


def test_transform_before_fit_does_not_raise() -> None:
    """An unfit vectorizer has an empty vocabulary; this is a degenerate input, not a bug."""
    result = TfidfVectorizer().transform("anything")
    assert result.shape == (0,)


# --- FlatIndex ------------------------------------------------------------------------------------


def _doc(key: str, text: str = "") -> CorpusDoc:
    return CorpusDoc(
        doc_key=key, source="t", doc_id=key, page_no=0, image_path="", width=1, height=1,
        text=text, n_text_chars=len(text),
    )


def test_text_only_search_ranks_the_matching_document_first() -> None:
    docs = [_doc("a", "quarterly revenue report"), _doc("b", "unrelated hiking trail guide")]
    index = FlatIndex.build(docs, image_embeddings={})
    assert index.search_text_only("revenue report", k=1) == ["a"]


def test_documents_with_no_ocr_text_are_findable_by_image_alone() -> None:
    """112 of 362 real corpus docs have this shape (BUGS.md-adjacent: DocVQA ships no OCR)."""
    docs = [_doc("scan", text=""), _doc("infographic", text="marketing budget breakdown")]
    image_embeddings = {
        "scan": np.array([1.0, 0.0], dtype=np.float32),
        "infographic": np.array([0.0, 1.0], dtype=np.float32),
    }
    index = FlatIndex.build(docs, image_embeddings, alpha=1.0)
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), "", k=1) == ["scan"]


def test_search_with_no_image_query_degrades_to_text_only() -> None:
    docs = [_doc("a", "budget report"), _doc("b", "trail map")]
    index = FlatIndex.build(
        docs, image_embeddings={"a": np.array([1.0], dtype=np.float32)}
    )
    assert index.search(None, "budget report", k=1) == index.search_text_only(
        "budget report", k=1
    )


def test_alpha_zero_is_pure_text_and_alpha_one_is_pure_image() -> None:
    docs = [_doc("a", "matches the text query"), _doc("b", "matches nothing")]
    image_embeddings = {
        "a": np.array([0.0, 1.0], dtype=np.float32),  # doesn't match the image query
        "b": np.array([1.0, 0.0], dtype=np.float32),  # does
    }
    query_image = np.array([1.0, 0.0], dtype=np.float32)

    text_index = FlatIndex.build(docs, image_embeddings, alpha=0.0)
    assert text_index.search(query_image, "matches the text query", k=1) == ["a"]

    image_index = FlatIndex.build(docs, image_embeddings, alpha=1.0)
    assert image_index.search(query_image, "matches the text query", k=1) == ["b"]


def test_build_with_no_image_embeddings_at_all_is_pure_text() -> None:
    """A corpus with zero image vectors computed yet must not crash -- it degrades to text."""
    docs = [_doc("a", "findable by text"), _doc("b", "also present")]
    index = FlatIndex.build(docs, image_embeddings={})
    assert index.image_vectors.shape == (2, 0)
    assert index.search(np.array([1.0]), "findable by text", k=1) == ["a"]


# --- recall_at_k ------------------------------------------------------------------------------


class _Query:
    def __init__(self, query_id: str, question: str, gold_doc_key: str) -> None:
        self.query_id = query_id
        self.question = question
        self.gold_doc_key = gold_doc_key


def test_recall_at_k_reports_all_three_signals() -> None:
    docs = [_doc("a", "revenue report"), _doc("b", "trail map")]
    image_embeddings = {"a": np.array([1.0, 0.0], dtype=np.float32)}
    index = FlatIndex.build(docs, image_embeddings, alpha=0.5)
    queries = [_Query("q1", "revenue report", "a")]
    query_vectors = {"q1": np.array([1.0, 0.0], dtype=np.float32)}

    result = recall_at_k(index, queries, query_vectors, k=1)
    assert set(result) == {"hybrid", "text_only", "image_only"}
    assert result["hybrid"] == 1.0
    assert result["text_only"] == 1.0
    assert result["image_only"] == 1.0


def test_recall_at_k_handles_a_query_with_no_image_vector() -> None:
    """A query the caller never embedded (e.g. embedding failed) must not crash image_only."""
    docs = [_doc("a", "revenue report")]
    index = FlatIndex.build(docs, image_embeddings={})
    queries = [_Query("q1", "revenue report", "a")]

    result = recall_at_k(index, queries, query_vectors={}, k=1)
    assert result["image_only"] == 0.0  # nothing to score, so no image-side hit
    assert result["text_only"] == 1.0


def test_recall_at_k_on_an_empty_query_list_does_not_divide_by_zero() -> None:
    index = FlatIndex.build([_doc("a")], image_embeddings={})
    result = recall_at_k(index, [], {}, k=1)
    assert result == {"hybrid": 0.0, "text_only": 0.0, "image_only": 0.0}


# --- the real question: does the image-space query signal do anything at all ------------------


@pytest.mark.slow
def test_hybrid_recall_against_text_and_image_only_controls_on_the_fixture() -> None:
    """Runs the real embedding pipeline on the 256M fixture and reports where each signal lands.

    This does not assert that hybrid beats both controls -- ``embed.py`` says plainly that the
    image-space query signal (query text through ``embed_tokens``, compared against connector
    output) has no contrastive training behind it and might contribute nothing. What it asserts is
    narrower and non-negotiable: recall must be well above chance for *something* (the corpus has
    real lexical signal InfographicVQA questions quote directly), and every score must be a valid
    probability. If every control reads near 1/n_docs, the retrieval stack is broken, not merely
    unhelpful.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    import torch
    from PIL import Image

    from edgerag.core.loader import FIXTURE_MODEL
    from edgerag.core.model import load_from_hf
    from edgerag.core.spec import ModelSpec
    from edgerag.retrieval.embed import embed_image, embed_query_for_image_space

    config = transformers.AutoConfig.from_pretrained(FIXTURE_MODEL)
    hf_model = transformers.AutoModelForImageTextToText.from_pretrained(
        FIXTURE_MODEL, config=config, dtype=torch.float32
    )
    hf_model.eval()
    processor = transformers.AutoProcessor.from_pretrained(FIXTURE_MODEL)
    spec = ModelSpec.from_hf_config(FIXTURE_MODEL, config)
    decoder = load_from_hf(spec, hf_model)
    device = torch.device("cpu")

    # Three synthetic pages with genuinely different colours, so the vision tower has something
    # to distinguish, and genuinely different text, so TF-IDF does too.
    colors = {"red": (200, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 200)}
    texts = {
        "red": "total revenue increased twelve percent this quarter",
        "green": "hiking trail elevation map and distance chart",
        "blue": "server uptime and latency dashboard metrics",
    }
    docs = [_doc(name, text=texts[name]) for name in colors]

    image_embeddings = {}
    for name, rgb in colors.items():
        image = Image.new("RGB", (64, 64), color=rgb)
        image_embeddings[name] = embed_image(hf_model, processor, image, device)

    index = FlatIndex.build(docs, image_embeddings, alpha=0.5)

    queries = [
        _Query("q_red", "what was the revenue increase this quarter", "red"),
        _Query("q_green", "how long is the hiking trail", "green"),
        _Query("q_blue", "what is the server uptime", "blue"),
    ]
    query_vectors = {
        q.query_id: embed_query_for_image_space(decoder, processor, q.question, device)
        for q in queries
    }

    result = recall_at_k(index, queries, query_vectors, k=1)
    for name, score in result.items():
        assert 0.0 <= score <= 1.0, f"{name} recall {score} is not a valid probability"

    # The lexical signal is real and unambiguous on this synthetic set (each question quotes its
    # document's vocabulary and no other document's), so text-only recall is the one number this
    # test can require rather than merely report.
    assert result["text_only"] == 1.0, (
        f"text-only recall was {result['text_only']}, expected 1.0 on an unambiguous synthetic "
        "set -- TF-IDF scoring itself is broken, not just the image signal"
    )
