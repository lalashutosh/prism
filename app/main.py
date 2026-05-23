from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.document_store import PdfChunk, PdfStore


ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(os.getenv("PDF_DOCS_DIR", ROOT_DIR / "docs")).resolve()
STATIC_DIR = Path(__file__).with_name("static")


def load_dotenv(path: Path = ROOT_DIR / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv()
store: PdfStore | None = None
app = FastAPI(title="PDF Query API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    score: float
    text: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=12)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/query", response_model=QueryResponse)
def query_pdf(payload: QueryRequest) -> QueryResponse:
    chunks = get_store().retrieve(
        query=payload.question,
        limit=payload.top_k,
    )
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant PDF text was found.")
    return QueryResponse(
        answer=answer_from_sources(payload.question, chunks),
        sources=[source_response(chunk) for chunk in chunks],
    )


def get_store() -> PdfStore:
    global store
    if store is None:
        store = PdfStore(DOCS_DIR)
    return store


def answer_from_sources(question: str, chunks: list[PdfChunk]) -> str:
    parts = [
        f'Relevant excerpts for: "{question}"',
        "",
    ]
    for index, chunk in enumerate(chunks[:3], start=1):
        pages = f"{chunk.page_start}-{chunk.page_end}" if chunk.page_start != chunk.page_end else str(chunk.page_start)
        excerpt = " ".join(chunk.text.split())[:700]
        parts.append(f"{index}. {chunk.document_name}, pages {pages} [{chunk.chunk_id}]")
        parts.append(excerpt)
        parts.append("")
    return "\n".join(parts).strip()


def source_response(chunk: PdfChunk) -> SourceResponse:
    return SourceResponse(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        score=chunk.score,
        text=chunk.text[:1200],
    )
