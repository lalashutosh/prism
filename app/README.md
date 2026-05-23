# Web App Architecture

This folder contains the thin web layer for the PDF query experience. The app is intentionally small: it serves the frontend and exposes one user-facing API for asking questions against indexed documents.

## Request Flow

1. The browser loads `static/index.html` from `GET /`.
2. `static/app.js` sends user questions to `POST /api/query`.
3. `main.py` validates the request with Pydantic.
4. `document_store.py` retrieves the most relevant PDF chunks from the local docs directory.
5. `main.py` returns a retrieval answer plus source excerpts.

## Files

- `main.py`: FastAPI entrypoint. Owns route definitions, request/response models, static file mounting, and response shaping.
- `document_store.py`: Local document indexing and retrieval logic. It extracts PDF text, chunks pages, scores chunks, and returns source matches.
- `static/index.html`: Minimal frontend shell.
- `static/app.js`: Browser behavior. It only calls `POST /api/query`.
- `static/styles.css`: Frontend styling.
- `__init__.py`: Marks `app` as a Python package.

## API Boundary

The only user communication API is:

```http
POST /api/query
```

Request body:

```json
{
  "question": "What does the document say about risk?",
  "top_k": 6
}
```

Response body:

```json
{
  "answer": "Relevant excerpts...",
  "sources": []
}
```

Keep upload, document management, health checks, and LLM provider routes out of this layer unless the product scope explicitly changes.

## Starting Locally

From the project root:

```powershell
python -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The server process keeps running in the terminal. Stop it with `Ctrl+C`.

## Runtime Data

`main.py` creates a `PdfStore` lazily on the first query using `PDF_DOCS_DIR` or `docs/` by default. PDFs are read from that directory and indexed in memory. The first query may take longer while the documents are indexed. Restart the server after changing the document set.

## Design Notes

- The frontend is static and server-rendering is not used.
- The backend does retrieval only; no Anthropic, Gemini, or other LLM provider is wired here.
- `document_store.py` can keep document-specific helpers internally, but the public web API should stay query-centered.
