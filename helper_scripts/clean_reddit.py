"""
Clean, group, and sort Reddit txt files in documents/ before ingestion.
Modifies files in-place. Safe to re-run — idempotent.

Steps:
  1. Clean  — remove spam, image embeds, short comments, off-topic posts,
              strip markdown URLs, fix escaped characters, collapse blank lines
  2. Group  — wrap each post + its comments in explicit markers so chunking
              never splits a post from its replies
  3. Sort   — reorder post blocks so similar topics are adjacent
"""

import os
import re

DOCS = os.path.join(os.path.dirname(__file__), "documents")

SPAM_DOMAINS = ["featherab.com", "shopit?", "bit.ly", "tinyurl"]

HOUSING_KEYWORDS = [
    "apartment", "rent", "lease", "housing", "landlord", "building",
    "bedroom", "studio", "roommate", "sublease", "sublet", "broker",
    "neighborhood", "commute", "deposit", "tenant", "move", "nyc",
    "manhattan", "brooklyn", "queens", "village", "street", "floor",
    "management", "maintenance", "nyu", "campus", "guarantor", "stabilize",
]

# Topic buckets: order determines sort order in output
TOPICS = [
    ("neighborhoods",   ["neighborhood", "village", "brooklyn", "queens", "astoria",
                         "williamsburg", "crown heights", "bushwick", "east village",
                         "lower east side", "les", "harlem", "jersey city", "hoboken"]),
    ("rent_prices",     ["rent", "price", "cost", "budget", "afford", "cheap", "expensive",
                         "dollar", "per month", "studio", "1br", "1-bedroom", "2-bedroom"]),
    ("lease_legal",     ["lease", "contract", "broker fee", "deposit", "guarantor",
                         "rent stabilize", "tenant rights", "hpd", "dhcr", "evict",
                         "sublease", "sublet", "renew"]),
    ("landlord_mgmt",   ["landlord", "management", "super", "maintenance", "repair",
                         "responsive", "management company", "property manager", "heat",
                         "roach", "pest", "rat", "mice", "mold", "noisy", "noise"]),
    ("roommates",       ["roommate", "roomie", "share apartment", "split rent",
                         "co-tenant", "housemate", "looking for roommate"]),
    ("commute_transit", ["commute", "subway", "train", "bus", "walk", "transit",
                         "l train", "a train", "f train", "path", "minutes from"]),
    ("safety",          ["safe", "unsafe", "crime", "sketchy", "dangerous", "robbery",
                         "security", "well-lit", "avoid"]),
    ("process_tips",    ["how to", "advice", "tips", "process", "application", "paperwork",
                         "credit check", "background check", "income requirement",
                         "40x", "80x", "co-signer", "streeteasy", "facebook group"]),
]
DEFAULT_TOPIC = "general"


# ─── step 1: clean ─────────────────────────────────────────────────────────────

def is_spam(line):
    lower = line.lower()
    return any(d in lower for d in SPAM_DOMAINS)


def strip_markdown_urls(line):
    line = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', line)
    line = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', line)
    return line


def clean_escaped_markdown(line):
    for old, new in [('\\-', '-'), ('\\*', '*'), ('\\_', '_'), ('\\>', '>'), ('\\#', '#')]:
        line = line.replace(old, new)
    line = re.sub(r'\\n', ' ', line)
    return line


def post_has_housing_content(lines):
    combined = ' '.join(lines).lower()
    return any(kw in combined for kw in HOUSING_KEYWORDS)


def clean_block(block):
    """Clean one post block. Returns list of cleaned lines or None if block should be dropped."""
    if not block or not block[0].startswith('POST:'):
        return None
    if not post_has_housing_content(block):
        return None

    out = []
    for line in block:
        if is_spam(line):
            continue
        if line.startswith('!['):
            continue
        line = strip_markdown_urls(line)
        line = clean_escaped_markdown(line)
        line = line.strip()
        if line.startswith('COMMENT:') and len(line[8:].strip()) < 40:
            continue
        out.append(line)

    # Collapse consecutive blank lines
    collapsed, prev_blank = [], False
    for line in out:
        blank = line == ''
        if blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = blank

    return collapsed if collapsed else None


# ─── step 2: group ─────────────────────────────────────────────────────────────

def format_block(lines):
    """
    Format a post block as Q:/A: for boundary-aware chunking.

    Q: carries the post title + body (collapsed to one line).
    Each COMMENT becomes an A: line.
    The chunker in ingest.py repeats the Q: line when a post's comments
    span multiple chunks, so every chunk is self-contained.
    """
    title = lines[0][5:].strip()  # strip "POST: "
    body_parts = []
    comments = []
    for line in lines[1:]:
        if line.startswith('COMMENT:'):
            text = line[8:].strip()
            if text:
                comments.append(f"A: {text}")
        elif line:
            body_parts.append(line)

    q_text = title
    if body_parts:
        q_text = title + ' — ' + ' '.join(body_parts)

    parts = [f"Q: {q_text}"]
    parts.extend(comments)
    return parts


# ─── step 3: sort ──────────────────────────────────────────────────────────────

def detect_topic(block_lines):
    text = ' '.join(block_lines).lower()
    for topic, keywords in TOPICS:
        if any(kw in text for kw in keywords):
            return topic
    return DEFAULT_TOPIC


def sort_blocks(blocks):
    """Sort blocks so same-topic posts cluster together."""
    topic_order = {t: i for i, (t, _) in enumerate(TOPICS)}
    topic_order[DEFAULT_TOPIC] = len(TOPICS)

    tagged = [(detect_topic(b), b) for b in blocks]
    tagged.sort(key=lambda x: topic_order.get(x[0], 99))
    return tagged


# ─── main per-file pipeline ────────────────────────────────────────────────────

def process_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        raw = f.read()

    # Split into raw blocks by POST: boundary
    raw_blocks = []
    current = []
    for line in raw.split('\n'):
        if line.startswith('POST:') and current:
            raw_blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        raw_blocks.append(current)

    # Step 1: clean
    cleaned = [clean_block(b) for b in raw_blocks]
    cleaned = [b for b in cleaned if b]

    kept, dropped = len(cleaned), len(raw_blocks) - len(cleaned)

    # Step 2: sort by topic
    tagged = sort_blocks(cleaned)

    # Step 3: format with grouping markers + topic headers
    output_lines = []
    current_topic = None
    for topic, block in tagged:
        if topic != current_topic:
            if output_lines:
                output_lines.append('')
            output_lines.append(f'## TOPIC: {topic.upper().replace("_", " ")}')
            output_lines.append('')
            current_topic = topic
        output_lines.extend(format_block(block))
        output_lines.append('')

    result = '\n'.join(output_lines).strip()
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)

    original_chars = len(raw)
    new_chars = len(result)
    reduction = (1 - new_chars / original_chars) * 100
    print(f"  {os.path.basename(filepath)}: {kept} posts kept, {dropped} dropped, "
          f"{original_chars:,} → {new_chars:,} chars ({reduction:.0f}% reduction)")


def main():
    reddit_files = sorted(
        f for f in os.listdir(DOCS)
        if f.startswith('reddit_') and f.endswith('.txt')
    )
    if not reddit_files:
        print("No reddit_*.txt files found in documents/")
        return

    print(f"Processing {len(reddit_files)} Reddit files (clean → group → sort)...")
    for fname in reddit_files:
        process_file(os.path.join(DOCS, fname))
    print("\nDone.")


if __name__ == "__main__":
    main()
