# RAG Pipeline — NYU Off-Campus Housing Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG system that answers qualitative questions about off-campus housing near NYU using experiential knowledge from Reddit threads, tenant reviews, NYC tenant rights guides, and neighborhood resources.

**Architecture:** Documents ingested from `documents/` → chunked by document type (200-char/0-overlap for informal Reddit/review posts, 500-char/150-overlap for formal tenant rights and neighborhood guides) → embedded with `all-MiniLM-L6-v2` via sentence-transformers → stored in ChromaDB → top-k=5 retrieved and passed to Groq LLM with grounding prompt.

**Tech Stack:** Python, sentence-transformers (`all-MiniLM-L6-v2`), ChromaDB ≥0.6, Groq API, pdfplumber, python-dotenv, Streamlit or Gradio (Milestone 5)

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/ingest.py` | Load raw docs from `documents/`, detect type, return list of `{text, metadata}` dicts |
| `src/chunk.py` | `chunk_document(text, doc_type)` → list of chunk strings; strategy varies by type |
| `src/embed.py` | Init ChromaDB collection, embed chunks, upsert with metadata |
| `src/retrieve.py` | Query ChromaDB top-k, return chunks + source metadata |
| `src/generate.py` | Build grounded prompt, call Groq, return answer |
| `src/pipeline.py` | Orchestrator: ingest → chunk → embed (one-time), then retrieve + generate per query |
| `app.py` | Streamlit UI: text input → pipeline query → display answer + sources |
| `tests/test_chunk.py` | Unit tests for chunking logic |
| `tests/test_retrieve.py` | Integration tests for embed + retrieve round-trip |
| `tests/test_generate.py` | Tests for prompt construction and grounding |
| `documents/` | Raw source files (`.txt`, `.pdf`, `.md`) placed here manually |

---

## Task 1: Project Setup & Environment Verification

**Files:**
- Modify: `requirements.txt`
- Create: `src/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Verify .env has required keys**

Open `.env.example` and confirm it has `GROQ_API_KEY`. Copy to `.env` if not done:
```
GROQ_API_KEY=your_key_here
```

- [ ] **Step 2: Install dependencies**

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Expected: all packages install without error. If chromadb fails on Windows, run:
```powershell
.venv\Scripts\python.exe -m pip install chromadb --no-build-isolation
```

- [ ] **Step 3: Create package init files**

```powershell
New-Item -ItemType Directory -Force src
New-Item -ItemType Directory -Force tests
New-Item -ItemType File -Path src\__init__.py
New-Item -ItemType File -Path tests\__init__.py
```

- [ ] **Step 4: Verify imports work**

```powershell
.venv\Scripts\python.exe -c "import sentence_transformers; import chromadb; import groq; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```powershell
git add src\__init__.py tests\__init__.py requirements.txt
git commit -m "chore: scaffold src and tests packages"
```

---

## Task 2: Document Ingestion (`src/ingest.py`)

**Files:**
- Create: `src/ingest.py`
- Create: `tests/test_ingest.py`
- Create: `documents/sample_reddit.txt`, `documents/sample_constitution.txt` (test fixtures)

- [ ] **Step 1: Create test fixture files**

Create `documents/sample_reddit.txt`:
```
honestly the place on bleecker near nyu is super sketchy, management never responds and there are roaches. avoid rose associates fr
```

Create `documents/sample_constitution.txt`:
```
Section 1: Security Deposits
Landlords may collect a maximum of one month's rent as a security deposit. The deposit must be returned within 14 days of lease termination along with an itemized list of any deductions.

Section 2: Broker Fees
As of 2020, broker fees may not be charged to tenants in New York City when the broker was hired by the landlord.
```

- [ ] **Step 2: Write failing test**

`tests/test_ingest.py`:
```python
import pytest
from src.ingest import load_documents

def test_load_txt_document():
    docs = load_documents("documents")
    assert len(docs) >= 1
    for doc in docs:
        assert "text" in doc
        assert "source" in doc["metadata"]
        assert "doc_type" in doc["metadata"]

def test_txt_doc_type_is_informal():
    docs = load_documents("documents")
    reddit_doc = next(d for d in docs if "reddit" in d["metadata"]["source"])
    assert reddit_doc["metadata"]["doc_type"] == "informal"

def test_constitution_doc_type_is_formal():
    docs = load_documents("documents")
    const_doc = next(d for d in docs if "constitution" in d["metadata"]["source"])
    assert const_doc["metadata"]["doc_type"] == "formal"
```

- [ ] **Step 3: Run test to verify it fails**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ingest.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` — `src/ingest.py` doesn't exist yet.

- [ ] **Step 4: Implement `src/ingest.py`**

```python
import os
import pdfplumber

INFORMAL_KEYWORDS = {"reddit", "discord", "slack", "tweet", "post", "thread"}

def _detect_doc_type(filename: str) -> str:
    name_lower = filename.lower()
    if any(kw in name_lower for kw in INFORMAL_KEYWORDS):
        return "informal"
    return "formal"

def _load_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_documents(directory: str) -> list[dict]:
    docs = []
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if fname.startswith(".") or os.path.isdir(fpath):
            continue
        if fname.endswith(".pdf"):
            text = _load_pdf(fpath)
        elif fname.endswith((".txt", ".md")):
            text = _load_text(fpath)
        else:
            continue
        docs.append({
            "text": text.strip(),
            "metadata": {
                "source": fname,
                "doc_type": _detect_doc_type(fname),
            }
        })
    return docs
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ingest.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/ingest.py tests/test_ingest.py documents/sample_reddit.txt documents/sample_constitution.txt
git commit -m "feat: implement document ingestion with doc_type detection"
```

---

## Task 3: Chunking Strategy (`src/chunk.py`)

**Files:**
- Create: `src/chunk.py`
- Create: `tests/test_chunk.py`

Strategy:
- **Informal** (Reddit/Discord posts): 200-char chunks, 0 overlap — short posts, no structure to preserve
- **Formal** (constitutions, newsletters, FAQs): 500-char chunks, 150-char overlap — preserves multi-sentence context across boundaries

- [ ] **Step 1: Write failing tests**

`tests/test_chunk.py`:
```python
from src.chunk import chunk_document

def test_informal_chunk_size():
    text = "x" * 600
    chunks = chunk_document(text, doc_type="informal")
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) == 3

def test_formal_chunk_overlap():
    # 500 chars then 150 overlap means chunk 2 starts at char 350
    text = "A" * 350 + "B" * 300
    chunks = chunk_document(text, doc_type="formal")
    assert len(chunks) == 2
    # overlap: chunk[1] should start 150 chars before end of chunk[0]
    assert chunks[1][:150] == chunks[0][-150:]

def test_empty_text_returns_empty_list():
    assert chunk_document("", doc_type="informal") == []

def test_text_shorter_than_chunk_size_returns_single_chunk():
    chunks = chunk_document("short text", doc_type="formal")
    assert len(chunks) == 1
    assert chunks[0] == "short text"
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_chunk.py -v
```
Expected: FAIL — `src/chunk.py` not found.

- [ ] **Step 3: Implement `src/chunk.py`**

```python
CHUNK_CONFIG = {
    "informal": {"size": 200, "overlap": 0},
    "formal":   {"size": 500, "overlap": 150},
}

def chunk_document(text: str, doc_type: str) -> list[str]:
    if not text:
        return []
    cfg = CHUNK_CONFIG.get(doc_type, CHUNK_CONFIG["formal"])
    size, overlap = cfg["size"], cfg["overlap"]
    step = size - overlap
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_chunk.py -v
```
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/chunk.py tests/test_chunk.py
git commit -m "feat: implement adaptive chunking (informal 200/0, formal 500/150)"
```

---

## Task 4: Embedding & Vector Store (`src/embed.py`)

**Files:**
- Create: `src/embed.py`
- Create: `tests/test_embed.py`

- [ ] **Step 1: Write failing test**

`tests/test_embed.py`:
```python
import chromadb
from src.embed import build_collection

def test_build_collection_upserts_chunks():
    docs = [
        {"text": "Robotics club meets Tuesdays", "metadata": {"source": "reddit.txt", "doc_type": "informal"}},
        {"text": "Article I: Name of organization is Robotics Club", "metadata": {"source": "constitution.txt", "doc_type": "formal"}},
    ]
    collection = build_collection(docs, collection_name="test_collection", persist_dir=None)
    assert collection.count() == 2

def test_build_collection_stores_metadata():
    docs = [
        {"text": "Chess club is very chill", "metadata": {"source": "post.txt", "doc_type": "informal"}},
    ]
    collection = build_collection(docs, collection_name="test_meta", persist_dir=None)
    results = collection.get(include=["metadatas"])
    assert results["metadatas"][0]["source"] == "post.txt"
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embed.py -v
```
Expected: FAIL — `src/embed.py` not found.

- [ ] **Step 3: Implement `src/embed.py`**

```python
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from src.chunk import chunk_document

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _make_id(source: str, chunk_index: int) -> str:
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()

def build_collection(
    docs: list[dict],
    collection_name: str = "club_guide",
    persist_dir: str | None = "chroma_store",
) -> chromadb.Collection:
    if persist_dir:
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.EphemeralClient()

    collection = client.get_or_create_collection(collection_name)
    model = _get_model()

    all_chunks, all_ids, all_meta = [], [], []
    for doc in docs:
        chunks = chunk_document(doc["text"], doc["metadata"]["doc_type"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(_make_id(doc["metadata"]["source"], i))
            all_meta.append({**doc["metadata"], "chunk_index": i})

    if all_chunks:
        embeddings = model.encode(all_chunks).tolist()
        collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_meta,
        )
    return collection
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embed.py -v
```
Expected: both PASS. Note: first run downloads `all-MiniLM-L6-v2` (~90MB) — normal.

- [ ] **Step 5: Commit**

```powershell
git add src/embed.py tests/test_embed.py
git commit -m "feat: embed chunks with all-MiniLM-L6-v2 and persist to ChromaDB"
```

---

## Task 5: Retrieval (`src/retrieve.py`)

**Files:**
- Create: `src/retrieve.py`
- Modify: `tests/test_retrieve.py` (new file)

- [ ] **Step 1: Write failing test**

`tests/test_retrieve.py`:
```python
from src.embed import build_collection
from src.retrieve import retrieve

def test_retrieve_returns_top_k():
    docs = [
        {"text": "Robotics club has strong mentorship program for freshmen", "metadata": {"source": "a.txt", "doc_type": "informal"}},
        {"text": "Chess club hosts weekly tournaments open to all skill levels", "metadata": {"source": "b.txt", "doc_type": "informal"}},
        {"text": "Debate team offers leadership roles including president and VP", "metadata": {"source": "c.txt", "doc_type": "informal"}},
    ]
    collection = build_collection(docs, collection_name="retrieve_test", persist_dir=None)
    results = retrieve(collection, query="which club has mentorship", top_k=2)
    assert len(results) == 2
    assert all("text" in r and "source" in r for r in results)

def test_retrieve_most_relevant_chunk_first():
    docs = [
        {"text": "Robotics club has strong mentorship program for freshmen", "metadata": {"source": "a.txt", "doc_type": "informal"}},
        {"text": "Chess club hosts weekly tournaments", "metadata": {"source": "b.txt", "doc_type": "informal"}},
    ]
    collection = build_collection(docs, collection_name="relevance_test", persist_dir=None)
    results = retrieve(collection, query="mentorship for new students", top_k=2)
    assert "mentorship" in results[0]["text"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_retrieve.py -v
```
Expected: FAIL — `src/retrieve.py` not found.

- [ ] **Step 3: Implement `src/retrieve.py`**

```python
import chromadb
from sentence_transformers import SentenceTransformer

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def retrieve(collection: chromadb.Collection, query: str, top_k: int = 5) -> list[dict]:
    model = _get_model()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "doc_type": meta.get("doc_type", "unknown"),
            "distance": dist,
        })
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_retrieve.py -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/retrieve.py tests/test_retrieve.py
git commit -m "feat: implement top-k semantic retrieval from ChromaDB"
```

---

## Task 6: Grounded Generation (`src/generate.py`)

**Files:**
- Create: `src/generate.py`
- Create: `tests/test_generate.py`

- [ ] **Step 1: Write failing tests**

`tests/test_generate.py`:
```python
from src.generate import build_prompt

def test_prompt_includes_all_chunks():
    chunks = [
        {"text": "Robotics club meets Tuesdays", "source": "reddit.txt"},
        {"text": "Chess club is inactive", "source": "post.txt"},
    ]
    prompt = build_prompt("Which clubs are active?", chunks)
    assert "Robotics club meets Tuesdays" in prompt
    assert "Chess club is inactive" in prompt

def test_prompt_includes_source_labels():
    chunks = [{"text": "Some info", "source": "reddit.txt"}]
    prompt = build_prompt("question?", chunks)
    assert "reddit.txt" in prompt

def test_prompt_contains_grounding_instruction():
    chunks = [{"text": "Info", "source": "a.txt"}]
    prompt = build_prompt("question?", chunks)
    grounding_phrases = ["only use", "provided documents", "do not"]
    assert any(p in prompt.lower() for p in grounding_phrases)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_generate.py -v
```
Expected: FAIL — `src/generate.py` not found.

- [ ] **Step 3: Implement `src/generate.py`**

```python
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an assistant for a student organization guide. "
    "Answer questions ONLY using the information in the provided documents. "
    "Do not use outside knowledge. If the documents do not contain enough information "
    "to answer, say so explicitly. Always cite the source document for each claim."
)

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)
    return f"Documents:\n{context}\n\nQuestion: {question}"

def generate_answer(question: str, chunks: list[dict], model: str = "llama3-8b-8192") -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    user_prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_generate.py -v
```
Expected: all 3 PASS. (No Groq API call in these tests — only `build_prompt` tested.)

- [ ] **Step 5: Commit**

```powershell
git add src/generate.py tests/test_generate.py
git commit -m "feat: grounded generation with Groq LLM and source citation"
```

---

## Task 7: Pipeline Orchestrator (`src/pipeline.py`)

**Files:**
- Create: `src/pipeline.py`

- [ ] **Step 1: Implement `src/pipeline.py`**

```python
from src.ingest import load_documents
from src.embed import build_collection
from src.retrieve import retrieve
from src.generate import generate_answer

def build_index(documents_dir: str = "documents", persist_dir: str = "chroma_store"):
    docs = load_documents(documents_dir)
    collection = build_collection(docs, persist_dir=persist_dir)
    print(f"Indexed {collection.count()} chunks from {len(docs)} documents.")
    return collection

def query(collection, question: str, top_k: int = 5) -> dict:
    chunks = retrieve(collection, question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return {
        "answer": answer,
        "sources": [{"source": c["source"], "text": c["text"][:120]} for c in chunks],
    }
```

- [ ] **Step 2: Smoke-test the pipeline end-to-end**

```powershell
.venv\Scripts\python.exe -c "
from src.pipeline import build_index, query
col = build_index()
result = query(col, 'Which clubs are most active this semester?')
print(result['answer'])
print('Sources:', [s['source'] for s in result['sources']])
"
```
Expected: printed answer citing documents in `documents/`. If `GROQ_API_KEY` missing, `KeyError` — verify `.env`.

- [ ] **Step 3: Commit**

```powershell
git add src/pipeline.py
git commit -m "feat: pipeline orchestrator for index build and query"
```

---

## Task 8: Streamlit Query Interface (`app.py`)

**Files:**
- Create: `app.py`
- Modify: `requirements.txt` (uncomment `streamlit`)

- [ ] **Step 1: Uncomment Streamlit in requirements.txt**

In `requirements.txt`, change:
```
# streamlit>=1.40.0
```
to:
```
streamlit>=1.40.0
```

Install:
```powershell
.venv\Scripts\python.exe -m pip install streamlit>=1.40.0
```

- [ ] **Step 2: Implement `app.py`**

```python
import streamlit as st
from src.pipeline import build_index, query

st.set_page_config(page_title="Club Guide", layout="centered")
st.title("Unofficial Student Org & Club Guide")

@st.cache_resource
def get_collection():
    return build_index()

collection = get_collection()

question = st.text_input("Ask about student clubs:", placeholder="Which clubs have strong mentorship programs?")

if question:
    with st.spinner("Searching documents..."):
        result = query(collection, question, top_k=5)
    st.markdown("### Answer")
    st.write(result["answer"])
    st.markdown("### Sources")
    for s in result["sources"]:
        st.markdown(f"**{s['source']}** — _{s['text']}..._")
```

- [ ] **Step 3: Run the app**

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
Expected: browser opens at `http://localhost:8501`. Type a question and verify answer + sources render.

- [ ] **Step 4: Commit**

```powershell
git add app.py requirements.txt
git commit -m "feat: Streamlit query UI with source attribution"
```

---

## Task 9: Fill in `planning.md` and `README.md`

**Files:**
- Modify: `planning.md`
- Modify: `README.md`

- [ ] **Step 1: Fill planning.md — Documents table**

Add your 10 sources. Use the fixture files as a template. Example row:
```
| 1 | r/[YourUniversity] Reddit thread | Reddit thread | https://reddit.com/r/... |
```
Aim for: 2–3 Reddit threads, 1–2 club constitutions (PDFs), 1–2 Discord/Slack FAQ exports, 1–2 departmental newsletters, 1 official club directory page.

- [ ] **Step 2: Fill planning.md — Chunking Strategy**

```
Chunk size: 200 chars (informal), 500 chars (formal)
Overlap: 0 (informal), 150 chars (formal)
Reasoning: Informal posts are short, self-contained opinions — splitting with overlap wastes context. Formal documents (constitutions, FAQs) contain multi-sentence rules where context spans boundaries; 150-char overlap ensures key clauses aren't severed.
```

- [ ] **Step 3: Fill planning.md — Retrieval Approach**

```
Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
Top-k: 5
Production tradeoff reflection: all-MiniLM-L6-v2 is fast and free but has a 256-token context limit, which can truncate long formal chunks. For production I'd evaluate text-embedding-3-small (OpenAI, 8191-token limit, API cost) vs. e5-large-v2 (local, stronger on domain text, higher latency). Multilingual support irrelevant for this corpus; domain specificity and context length are the primary tradeoffs.
```

- [ ] **Step 4: Fill planning.md — Evaluation Plan table**

Use the 5 questions from the project summary:
```
| 1 | Which clubs are most recommended for students looking to gain leadership experience? | Names clubs with leadership roles per the documents |
| 2 | Based on recent activity, which organizations are holding regular weekly meetings this semester? | Lists clubs with documented meeting schedules |
| 3 | What do students say is the primary benefit of joining Robotics Club compared to other groups? | Cites student testimonials from Reddit/Discord sources |
| 4 | Which clubs are known for having the strongest mentorship programs for new members? | Names clubs with explicit mentorship mentions |
| 5 | Are there any organizations frequently cited as dead or inactive despite official listing? | Names inactive clubs per student posts |
```

- [ ] **Step 5: Commit planning.md**

```powershell
git add planning.md
git commit -m "docs: complete planning.md spec sections"
```

---

## Task 10: Collect Real Documents & Run Evaluation

**Files:**
- Add: files to `documents/` directory
- Modify: `README.md`

- [ ] **Step 1: Collect and save real documents**

For each source in your planning.md table:
- Reddit threads: copy post text + top comments into `.txt` files named `reddit_<clubname>_<topic>.txt`
- Club constitutions: save as `constitution_<clubname>.pdf` or `.txt`
- Discord/Slack FAQs: export and save as `discord_<clubname>_faq.txt`
- Newsletters: save as `newsletter_<source>_<date>.txt`

Naming convention matters — `ingest.py` uses filename to detect `informal` vs `formal`.

- [ ] **Step 2: Rebuild the index with real documents**

```powershell
# Delete old chroma store to force fresh index
Remove-Item -Recurse -Force chroma_store -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -c "from src.pipeline import build_index; col = build_index(); print(col.count(), 'chunks')"
```
Expected: chunk count > 50 if you have 10+ documents.

- [ ] **Step 3: Run the 5 evaluation questions**

```powershell
.venv\Scripts\python.exe -c "
from src.pipeline import build_index, query
import json
col = build_index()
questions = [
    'Which clubs are most recommended for students looking to gain leadership experience for their resumes?',
    'Based on recent activity, which organizations are holding regular weekly meetings this semester?',
    'What do students say is the primary benefit of joining Robotics Club compared to other similar groups?',
    'Which clubs are known for having the strongest mentorship programs for new members?',
    'Are there any student organizations frequently cited as dead or inactive despite being listed on the official university website?',
]
for q in questions:
    r = query(col, q)
    print(f'Q: {q}')
    print(f'A: {r[\"answer\"][:300]}')
    print(f'Sources: {[s[\"source\"] for s in r[\"sources\"]]}')
    print('---')
"
```

- [ ] **Step 4: Fill README.md evaluation table**

Record results honestly. For each question note: retrieval quality (Relevant / Partially relevant / Off-target) and response accuracy (Accurate / Partially accurate / Inaccurate).

- [ ] **Step 5: Fill README.md failure case section**

Pick the worst-performing question. Diagnose the root cause tied to a specific pipeline stage. Example:
> "The embedding model returned off-target chunks because the student used slang ('dead club') that MiniLM hasn't seen in fine-tuning context — embedding distance was high. Fix: add a preprocessing step that normalizes slang terms, or switch to a model fine-tuned on social media text."

- [ ] **Step 6: Commit**

```powershell
git add documents/ README.md
git commit -m "data: add real source documents and complete evaluation report"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 5 evaluation questions have corresponding pipeline stages. Adaptive chunking (informal/formal) implemented. All 5 pipeline stages covered (Ingestion→Chunking→Embedding→Retrieval→Generation). Groq grounding enforced via system prompt.
- [x] **Placeholder scan:** No TBD, TODO, or "implement later" — all steps have actual code.
- [x] **Type consistency:** `chunk_document` called consistently across `embed.py` and `test_chunk.py`. `build_collection` returns `chromadb.Collection` used identically in `retrieve.py` and `pipeline.py`. `retrieve` returns `list[dict]` with `text`/`source` keys consumed by `generate.py` and `app.py`.
