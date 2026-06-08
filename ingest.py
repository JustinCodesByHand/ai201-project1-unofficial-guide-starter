import os
import re
from config import DOCS_PATH


def load_documents():
    """Load all .txt documents from the documents folder."""
    documents = []
    for filename in sorted(os.listdir(DOCS_PATH)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_PATH, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            source_name = filename.replace(".txt", "").replace("_", " ").title()
            doc_type = "informal" if any(k in filename for k in ["reddit", "review", "post", "thread"]) else "formal"
            documents.append({
                "source": source_name,
                "filename": filename,
                "doc_type": doc_type,
                "text": text,
            })
    print(f"Loaded {len(documents)} document(s): {[d['source'] for d in documents]}")
    return documents


def _chunk_informal(text, source_name):
    """
    Boundary-aware chunker for Reddit Q:/A: format.

    Splits at Q: post boundaries. Each chunk starts with the Q: title
    (repeated if a post has many comments) so every chunk is self-contained.
    Never cuts mid-comment.
    """
    CHUNK_SIZE = 500
    MIN_LENGTH = 50
    prefix = source_name.lower().replace(" ", "_")
    chunks = []
    counter = 0

    # Split into post blocks at Q: boundaries
    blocks = re.split(r'\n(?=Q:)', text)

    for block in blocks:
        block = block.strip()
        if not block.startswith('Q:'):
            continue

        lines = block.split('\n')
        q_line = lines[0]  # Q: title line
        a_lines = [l for l in lines[1:] if l.startswith('A:') and len(l[2:].strip()) >= 30]

        if not a_lines:
            # Post with no usable comments — single chunk
            if len(block) >= MIN_LENGTH:
                chunks.append({"text": block[:CHUNK_SIZE], "source": source_name, "chunk_id": f"{prefix}_{counter}", "doc_type": "informal"})
                counter += 1
            continue

        # Pack A: lines into chunks; repeat Q: header at top of each chunk
        current = [q_line]
        current_len = len(q_line)

        for a_line in a_lines:
            fits = current_len + len(a_line) + 1 <= CHUNK_SIZE
            if not fits and len(current) > 1:
                chunk_text = '\n'.join(current)
                if len(chunk_text) >= MIN_LENGTH:
                    chunks.append({"text": chunk_text, "source": source_name, "chunk_id": f"{prefix}_{counter}", "doc_type": "informal"})
                    counter += 1
                current = [q_line, a_line]
                current_len = len(q_line) + len(a_line) + 1
            else:
                current.append(a_line)
                current_len += len(a_line) + 1

        if len(current) > 1:
            chunk_text = '\n'.join(current)
            if len(chunk_text) >= MIN_LENGTH:
                chunks.append({"text": chunk_text, "source": source_name, "chunk_id": f"{prefix}_{counter}", "doc_type": "informal"})
                counter += 1

    return chunks


def _chunk_formal(text, source_name):
    """Character sliding window for formal documents (Wikipedia, housing stats, transit)."""
    CHUNK_SIZE = 500
    OVERLAP = 150
    MIN_LENGTH = 50
    prefix = source_name.lower().replace(" ", "_")
    chunks = []
    counter = 0
    start = 0
    while start < len(text):
        chunk_text = text[start:start + CHUNK_SIZE].strip()
        if len(chunk_text) >= MIN_LENGTH:
            chunks.append({"text": chunk_text, "source": source_name, "chunk_id": f"{prefix}_{counter}", "doc_type": "formal"})
            counter += 1
        start += CHUNK_SIZE - OVERLAP
    return chunks


def chunk_document(text, source_name, doc_type="formal"):
    if doc_type == "informal":
        return _chunk_informal(text, source_name)
    return _chunk_formal(text, source_name)
