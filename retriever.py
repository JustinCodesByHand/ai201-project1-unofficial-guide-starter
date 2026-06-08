import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL, N_RESULTS

# Embedding function and ChromaDB client are initialized once at module load.
# sentence-transformers downloads the model on first use — this may take
# 30–60 seconds the very first time. Subsequent runs use a local cache.
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},
)


def get_collection():
    """Return the ChromaDB collection. Used by app.py during ingestion."""
    return _collection


def embed_and_store(chunks):
    """Store embedded chunks in ChromaDB. Embedding is handled by the collection's
    SentenceTransformer function — pass text, get vectors automatically."""
    _collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "doc_type": c.get("doc_type", "formal")} for c in chunks],
        ids=[c["chunk_id"] for c in chunks],
    )
    print(f"Stored {_collection.count()} total chunks in the vector database.")


def retrieve(query, n_results=N_RESULTS):
    """
    Semantic search over the housing corpus.

    Returns up to n_results dicts: {"text", "source", "doc_type", "distance"}.
    Distance is cosine distance after formal-source boost (lower = more relevant).
    Two-pass fetch ensures formal docs compete against high-volume Reddit chunks.
    """
    if _collection.count() == 0:
        return []

    FORMAL_BOOST = 0.70
    DISTANCE_THRESHOLD = 0.5

    def _fetch(n, where=None):
        kw = {
            "query_texts": [query],
            "n_results": min(n, _collection.count()),
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kw["where"] = where
        r = _collection.query(**kw)
        return r["documents"][0], r["metadatas"][0], r["distances"][0]

    # Pass 1: top candidates from full collection
    docs, metas, dists = _fetch(n_results * 3)

    # Pass 2: top candidates from formal sources only — prevents Reddit from
    # crowding out stats/guide chunks before the boost is applied
    formal_count = len(_collection.get(
        where={"doc_type": {"$eq": "formal"}}, include=[]
    )["ids"])
    if formal_count > 0:
        fdocs, fmetas, fdists = _fetch(n_results * 3, where={"doc_type": {"$eq": "formal"}})
        docs = list(docs) + list(fdocs)
        metas = list(metas) + list(fmetas)
        dists = list(dists) + list(fdists)

    # Merge, dedupe by text prefix, apply boost
    seen_text_keys = set()
    boosted = []
    for doc, meta, dist in zip(docs, metas, dists):
        key = doc[:80]
        if key in seen_text_keys:
            continue
        seen_text_keys.add(key)
        effective_dist = dist * FORMAL_BOOST if meta.get("doc_type") == "formal" else dist
        boosted.append({
            "text": doc,
            "source": meta["source"],
            "doc_type": meta.get("doc_type", "formal"),
            "distance": effective_dist,
        })

    boosted.sort(key=lambda c: c["distance"])

    # Final pass: filter threshold, dedupe same-post informal chunks, take top n
    seen_q_lines = set()
    formal_source_counts = {}
    MAX_PER_FORMAL_SOURCE = 2
    chunks = []
    for c in boosted:
        if c["distance"] > DISTANCE_THRESHOLD:
            continue
        if c["doc_type"] == "informal":
            q_line = c["text"].split("\n")[0]
            if q_line in seen_q_lines:
                continue
            seen_q_lines.add(q_line)
        else:
            count = formal_source_counts.get(c["source"], 0)
            if count >= MAX_PER_FORMAL_SOURCE:
                continue
            formal_source_counts[c["source"]] = count + 1
        chunks.append(c)
        if len(chunks) >= n_results:
            break

    for chunk in chunks:
        print(f"[{chunk['source']}] (dist: {chunk['distance']:.3f}) {chunk['text'][:80]}...")

    return chunks

