from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class PdfChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class PdfDocument:
    document_id: str
    name: str
    path: Path
    pages: int
    chunks: int


def make_document_id(path: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return stem or "document"


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def extract_pdf_pages(path: Path) -> list[str]:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("pdfplumber is not installed. Run: pip install -r requirements.txt") from exc

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=False, x_tolerance=2, y_tolerance=3) or ""
            cleaned = re.sub(r"[ \t]+", " ", text).strip()
            pages.append(cleaned)
    return pages


def chunk_pages(
    document_id: str,
    document_name: str,
    pages: list[str],
    max_words: int = 420,
    overlap_words: int = 80,
) -> list[PdfChunk]:
    chunks: list[PdfChunk] = []
    buffer: list[tuple[int, str]] = []
    word_count = 0

    def flush() -> None:
        nonlocal buffer, word_count
        if not buffer:
            return
        text = "\n\n".join(part for _, part in buffer).strip()
        if text:
            page_numbers = [page for page, _ in buffer]
            chunks.append(
                PdfChunk(
                    chunk_id=f"{document_id}:{len(chunks) + 1}",
                    document_id=document_id,
                    document_name=document_name,
                    page_start=min(page_numbers),
                    page_end=max(page_numbers),
                    text=text,
                )
            )
        if overlap_words <= 0:
            buffer = []
            word_count = 0
            return
        tail_words = tokenize(text)[-overlap_words:]
        tail_text = " ".join(tail_words)
        last_page = buffer[-1][0]
        buffer = [(last_page, tail_text)] if tail_text else []
        word_count = len(tail_words)

    for page_index, page_text in enumerate(pages, start=1):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|(?<=\.)\s{2,}", page_text) if p.strip()]
        for paragraph in paragraphs or [page_text]:
            paragraph_words = len(tokenize(paragraph))
            if buffer and word_count + paragraph_words > max_words:
                flush()
            buffer.append((page_index, paragraph))
            word_count += paragraph_words
    flush()
    return chunks


class PdfStore:
    def __init__(self, docs_dir: Path) -> None:
        self.docs_dir = docs_dir
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._documents: dict[str, PdfDocument] = {}
        self._chunks: dict[str, list[PdfChunk]] = {}
        self.refresh()

    def refresh(self) -> None:
        documents: dict[str, PdfDocument] = {}
        chunks_by_doc: dict[str, list[PdfChunk]] = {}
        for path in sorted(self.docs_dir.glob("*.pdf")):
            document_id = make_document_id(path)
            pages = extract_pdf_pages(path)
            chunks = chunk_pages(document_id, path.name, pages)
            documents[document_id] = PdfDocument(
                document_id=document_id,
                name=path.name,
                path=path,
                pages=len(pages),
                chunks=len(chunks),
            )
            chunks_by_doc[document_id] = chunks
        self._documents = documents
        self._chunks = chunks_by_doc

    def add_pdf(self, filename: str, content: bytes) -> PdfDocument:
        safe_name = Path(filename).name
        if not safe_name.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")
        target = self.docs_dir / safe_name
        target.write_bytes(content)
        self.refresh()
        return self._documents[make_document_id(target)]

    def list_documents(self) -> list[PdfDocument]:
        return list(self._documents.values())

    def get_document(self, document_id: str) -> PdfDocument | None:
        return self._documents.get(document_id)

    def list_chunks(self, document_id: str, limit: int = 50) -> list[PdfChunk]:
        return self._chunks.get(document_id, [])[:limit]

    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 6,
    ) -> list[PdfChunk]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        candidate_chunks = self._candidate_chunks(document_id)
        if not candidate_chunks:
            return []

        doc_freq = Counter()
        tokenized_chunks: list[tuple[PdfChunk, Counter[str]]] = []
        for chunk in candidate_chunks:
            counts = Counter(tokenize(chunk.text))
            tokenized_chunks.append((chunk, counts))
            doc_freq.update(counts.keys())

        total_docs = len(candidate_chunks)
        scored: list[PdfChunk] = []
        for chunk, counts in tokenized_chunks:
            length_norm = max(sum(counts.values()), 1)
            score = 0.0
            for term, query_weight in query_counts.items():
                if term not in counts:
                    continue
                idf = math.log((total_docs + 1) / (doc_freq[term] + 0.5)) + 1.0
                score += query_weight * counts[term] * idf / math.sqrt(length_norm)
            if score > 0:
                scored.append(
                    PdfChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        text=chunk.text,
                        score=round(score, 4),
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _candidate_chunks(self, document_id: str | None) -> list[PdfChunk]:
        if document_id:
            return self._chunks.get(document_id, [])
        return [chunk for chunks in self._chunks.values() for chunk in chunks]
