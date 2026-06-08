from ingest import load_documents, chunk_document
from retriever import get_collection, embed_and_store, retrieve
from generator import generate_response


def _ensure_index():
    collection = get_collection()
    if collection.count() == 0:
        docs = load_documents()
        all_chunks = []
        for doc in docs:
            chunks = chunk_document(doc["text"], doc["source"], doc["doc_type"])
            all_chunks.extend(chunks)
        embed_and_store(all_chunks)
    return collection


def ask(question: str) -> dict:
    """
    End-to-end RAG query.
    Returns {"answer": str, "sources": list[str]}
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    collection = _ensure_index()

    if collection.count() == 0:
        return {
            "answer": "No documents indexed. Add files to documents/ and restart.",
            "sources": [],
        }

    chunks = retrieve(question)

    if not chunks:
        return {
            "answer": "No relevant documents found. Try rephrasing your question.",
            "sources": [],
        }

    answer = generate_response(question, chunks)

    sources = [
        f"{c['source']} (dist: {c['distance']:.3f}) — {c['text'][:120].strip()}..."
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}
