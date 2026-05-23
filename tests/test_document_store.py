from app.document_store import PdfStore, chunk_pages, tokenize


def test_tokenize_normalizes_words():
    assert tokenize("AI-system, Article 5!") == ["ai-system", "article", "5"]


def test_chunk_pages_preserves_page_range():
    chunks = chunk_pages(
        document_id="doc",
        document_name="doc.pdf",
        pages=["alpha beta. " * 40, "gamma delta. " * 40],
        max_words=30,
        overlap_words=5,
    )
    assert chunks
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2


def test_retrieve_scores_relevant_chunk(tmp_path):
    store = PdfStore(tmp_path)
    store._documents = {}
    store._chunks = {
        "demo": chunk_pages(
            "demo",
            "demo.pdf",
            [
                "Biometric identification and high-risk AI systems.",
                "Unrelated information about invoices and payments.",
            ],
            max_words=20,
            overlap_words=0,
        )
    }

    results = store.retrieve("biometric high-risk", document_id="demo", limit=1)

    assert len(results) == 1
    assert "Biometric" in results[0].text
