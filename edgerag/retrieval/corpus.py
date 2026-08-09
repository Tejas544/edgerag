"""Corpus construction from DocVQA / InfographicVQA.

The corpus is built once and frozen. Everything measured from Phase 1 onward replays the same
inputs, which is ``00_FOUNDATIONS.md`` §4 rule 5 ("fix everything you're not measuring") applied at
the project level rather than the function level.

Two sources, chosen for complementary properties:

* **InfographicVQA** ships Amazon Textract OCR alongside each image, so these are genuinely
  *multimodal* documents -- image and text -- without taking on an OCR dependency.
* **DocVQA** is dense scanned text with **multiple questions per document**. That grouping is not
  incidental: it is a naturally-occurring prefix-sharing workload, which is precisely what the
  copy-on-write block sharing in Phase 3 exists to exploit. Synthesising that pattern would have
  been guesswork; here it comes from the data.

Documents are deduplicated on ``(source, doc_id, page_no)`` because several questions map to the
same page, and a corpus that stored one copy per question would inflate both the index and the
memory accounting.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_IMAGE_DIR = DATA_DIR / "corpus"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"

HF_REPO = "lmms-lab/DocVQA"


@dataclass
class CorpusDoc:
    """One retrievable page: an image, and whatever text we have for it."""

    doc_key: str  # stable primary key, "<source>:<doc_id>:<page_no>"
    source: str
    doc_id: str
    page_no: int
    image_path: str
    width: int
    height: int
    text: str  # "" when the source ships no OCR -- this is expected, not a defect
    n_text_chars: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class QueryItem:
    """One question with its gold answers and the document that answers it."""

    query_id: str
    source: str
    question: str
    answers: list[str]
    gold_doc_key: str
    split: str = "eval"  # "eval" or "heldout"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def extract_textract_lines(ocr_raw: str) -> str:
    """Pull readable text out of the Textract blob InfographicVQA ships.

    The container is awkward in two ways that are easy to get wrong silently (see ``BUGS.md``
    B-01):

    1. The field is not JSON. It is a **Python list repr** -- it literally starts ``['{`` and ends
       ``}']`` -- so ``json.loads`` fails on character 1. The payload is the single string element
       inside that list.
    2. The decoded payload groups blocks **by type**: top-level keys are ``PAGE``, ``LINE``, and
       ``WORD``, each mapping to its own list. It is not one flat ``Blocks`` array. Reading
       ``payload["PAGE"]`` returns exactly one page-geometry block and zero text, which yields an
       empty string rather than an error.

    We keep ``LINE`` and drop ``WORD``: lines preserve reading order and cut the text to roughly a
    fifth of the size. Block order is preserved rather than re-sorted by geometry, because sorting
    by top-then-left mangles the multi-column layouts that infographics are full of.

    Returns ``""`` on anything unexpected -- a malformed blob should cost one document's text, not
    the corpus build. Callers must therefore check *coverage*, not just absence of exceptions.
    """
    if not ocr_raw:
        return ""

    payload: Any = None
    try:
        payload = json.loads(ocr_raw)
    except (json.JSONDecodeError, TypeError):
        try:
            payload = ast.literal_eval(ocr_raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return ""

    # Unwrap the single-element list, then decode the JSON string it holds.
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return ""
    if not isinstance(payload, dict):
        return ""

    blocks: list[Any] = []
    if isinstance(payload.get("LINE"), list):
        blocks = payload["LINE"]
    else:
        # Fall back to a flat array under any of the names Textract exports use.
        for key in ("Blocks", "blocks"):
            if isinstance(payload.get(key), list):
                blocks = payload[key]
                break

    lines = [
        b["Text"]
        for b in blocks
        if isinstance(b, dict) and b.get("BlockType") == "LINE" and b.get("Text")
    ]
    return "\n".join(lines)


def _normalise_image(image: Any) -> Any:
    """DocVQA pages arrive as mode ``L``; the vision tower expects three channels."""
    return image if image.mode == "RGB" else image.convert("RGB")


def iter_source_rows(config_name: str, split: str, limit: int) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    stream = load_dataset(HF_REPO, config_name, split=split, streaming=True)
    yield from stream.take(limit)


def build_corpus(
    n_infographic: int = 250,
    n_docvqa: int = 400,
    split: str = "validation",
    image_dir: Path = CORPUS_IMAGE_DIR,
    jpeg_quality: int = 92,
) -> tuple[list[CorpusDoc], list[QueryItem]]:
    """Stream both sources, dedupe pages, and write images to disk.

    Row counts are *question* counts, not page counts -- DocVQA yields roughly one page per three
    questions, which is the property that makes the prefix-sharing measurement possible.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    docs: dict[str, CorpusDoc] = {}
    queries: list[QueryItem] = []

    sources = [
        ("infographicvqa", "InfographicVQA", n_infographic),
        ("docvqa", "DocVQA", n_docvqa),
    ]

    for source, config_name, limit in sources:
        if limit <= 0:
            continue
        for row in iter_source_rows(config_name, split, limit):
            if source == "docvqa":
                doc_id = str(row.get("docId"))
                page_no = int(row.get("ucsf_document_page_no") or 0)
                text = ""  # DocVQA ships no OCR; retrieval leans on the image embedding
            else:
                doc_id = str(row.get("questionId"))
                page_no = 0
                text = extract_textract_lines(row.get("ocr") or "")

            doc_key = f"{source}:{doc_id}:{page_no}"

            if doc_key not in docs:
                image = _normalise_image(row["image"])
                path = image_dir / f"{source}_{doc_id}_{page_no}.jpg"
                image.save(path, "JPEG", quality=jpeg_quality)
                docs[doc_key] = CorpusDoc(
                    doc_key=doc_key,
                    source=source,
                    doc_id=doc_id,
                    page_no=page_no,
                    image_path=str(path.relative_to(DATA_DIR.parent)).replace("\\", "/"),
                    width=image.width,
                    height=image.height,
                    text=text,
                    n_text_chars=len(text),
                )

            answers = list(row.get("answers") or [])
            queries.append(
                QueryItem(
                    query_id=f"{source}:{row.get('questionId')}",
                    source=source,
                    question=str(row.get("question") or ""),
                    answers=answers,
                    gold_doc_key=doc_key,
                )
            )

    return list(docs.values()), queries


def assign_heldout(queries: list[QueryItem], fraction: float = 0.25, seed: int = 1234) -> None:
    """Carve out a held-out split, in place.

    Split by **document**, not by question. Splitting by question would put two questions about the
    same page on opposite sides of the split, and any quality number measured against it would be
    contaminated by prefix reuse -- the exact thing Phase 3 optimises.
    """
    import random

    rng = random.Random(seed)
    doc_keys = sorted({q.gold_doc_key for q in queries})
    rng.shuffle(doc_keys)
    n_heldout = int(len(doc_keys) * fraction)
    heldout = set(doc_keys[:n_heldout])

    for q in queries:
        q.split = "heldout" if q.gold_doc_key in heldout else "eval"


def write_jsonl(path: Path, items: list[CorpusDoc] | list[QueryItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.to_json() + "\n")


def load_corpus(path: Path = CORPUS_PATH) -> list[CorpusDoc]:
    with path.open("r", encoding="utf-8") as fh:
        return [CorpusDoc(**json.loads(line)) for line in fh if line.strip()]


def load_queries(path: Path = QUERIES_PATH) -> list[QueryItem]:
    with path.open("r", encoding="utf-8") as fh:
        return [QueryItem(**json.loads(line)) for line in fh if line.strip()]
