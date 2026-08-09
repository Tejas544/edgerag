"""Tests for corpus construction.

The OCR parser is the interesting part: it fails by returning ``""``, never by raising, so it is
exactly the kind of code that breaks silently and is discovered three phases later. See
``BUGS.md`` B-01.
"""

from __future__ import annotations

import json

from edgerag.retrieval.corpus import QueryItem, assign_heldout, extract_textract_lines


def _textract_payload() -> dict:
    """The real shape: blocks grouped by type, not one flat array."""
    return {
        "PAGE": [{"BlockType": "PAGE", "Geometry": {}, "Id": "page-1"}],
        "LINE": [
            {"BlockType": "LINE", "Text": "Social Media Demographics", "Id": "l1"},
            {"BlockType": "LINE", "Text": "Pinterest: 71% female", "Id": "l2"},
            {"BlockType": "LINE", "Text": "", "Id": "l3"},  # empty text must be dropped
        ],
        "WORD": [{"BlockType": "WORD", "Text": "Social"}, {"BlockType": "WORD", "Text": "Media"}],
    }


def _as_shipped(payload: dict) -> str:
    """Reproduce the container exactly as the dataset ships it: a Python list repr."""
    return str([json.dumps(payload)])


# --- BUGS.md B-01 -------------------------------------------------------------------------


def test_parses_the_python_list_repr_container() -> None:
    """The field starts with ``['{`` -- json.loads fails at character 1."""
    raw = _as_shipped(_textract_payload())
    assert raw.startswith("['{")
    assert raw.endswith("}']")

    text = extract_textract_lines(raw)
    assert "Social Media Demographics" in text
    assert "Pinterest: 71% female" in text


def test_reads_line_blocks_not_page_blocks() -> None:
    """The original bug: PAGE holds one geometry block and no text, so the result was always ''."""
    text = extract_textract_lines(_as_shipped(_textract_payload()))
    assert text.count("\n") == 1  # two lines kept, the empty one dropped
    assert "Social" in text


def test_word_blocks_are_dropped() -> None:
    """WORD duplicates LINE content at ~5x the size; keeping both would bloat the index."""
    text = extract_textract_lines(_as_shipped(_textract_payload()))
    assert text.split("\n") == ["Social Media Demographics", "Pinterest: 71% female"]


def test_block_order_is_preserved_not_geometry_sorted() -> None:
    """Sorting by top-then-left mangles the multi-column layouts infographics are full of."""
    payload = {
        "LINE": [
            {"BlockType": "LINE", "Text": "second", "Geometry": {"BoundingBox": {"Top": 0.9}}},
            {"BlockType": "LINE", "Text": "first", "Geometry": {"BoundingBox": {"Top": 0.1}}},
        ]
    }
    assert extract_textract_lines(_as_shipped(payload)) == "second\nfirst"


def test_accepts_plain_json_container_too() -> None:
    """Not every export is wrapped in a list repr; both containers must work."""
    assert "hello" in extract_textract_lines(
        json.dumps({"LINE": [{"BlockType": "LINE", "Text": "hello"}]})
    )


def test_falls_back_to_flat_blocks_array() -> None:
    raw = json.dumps({"Blocks": [{"BlockType": "LINE", "Text": "flat form"}]})
    assert extract_textract_lines(raw) == "flat form"


def test_degrades_to_empty_string_never_raises() -> None:
    """One malformed blob must cost one document's text, not the whole corpus build."""
    for junk in ["", "not json at all", "[", "{}", "[]", '["not-json-inside"]', "null"]:
        assert extract_textract_lines(junk) == ""


# --- held-out split -----------------------------------------------------------------------


def _queries() -> list[QueryItem]:
    return [
        QueryItem(
            query_id=f"q{i}",
            source="docvqa",
            question="?",
            answers=["a"],
            gold_doc_key=f"docvqa:{i // 3}:0",  # three questions per document
        )
        for i in range(30)
    ]


def test_heldout_split_never_splits_a_document() -> None:
    """Two questions about one page on opposite sides of the split would contaminate every
    quality number with the prefix reuse that Phase 3 optimises."""
    queries = _queries()
    assign_heldout(queries, fraction=0.25, seed=7)

    by_doc: dict[str, set[str]] = {}
    for q in queries:
        by_doc.setdefault(q.gold_doc_key, set()).add(q.split)
    assert all(len(splits) == 1 for splits in by_doc.values())


def test_heldout_split_is_deterministic() -> None:
    a, b = _queries(), _queries()
    assign_heldout(a, seed=7)
    assign_heldout(b, seed=7)
    assert [q.split for q in a] == [q.split for q in b]


def test_heldout_fraction_is_approximately_honoured() -> None:
    queries = _queries()
    assign_heldout(queries, fraction=0.25, seed=7)
    heldout_docs = {q.gold_doc_key for q in queries if q.split == "heldout"}
    assert len(heldout_docs) == 2  # floor(10 docs * 0.25) == 2
