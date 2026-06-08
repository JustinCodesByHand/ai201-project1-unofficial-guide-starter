""" "
Scrape Wikipedia articles for NYU housing corpus.
Uses Wikipedia REST API — returns clean plain text, no browser needed.
Saves to documents/ for ingestion.
"""

import os
import time

import requests

HEADERS = {"User-Agent": "nyu-housing-guide/1.0 (educational project)"}
SLEEP = 1.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(OUT_DIR, exist_ok=True)

WIKI_API = "https://en.wikipedia.org/w/api.php"

# 7 Wikipedia articles → 3 Reddit + 7 Wikipedia = 10 total sources
SOURCES = [
    {"title": "Greenwich_Village", "filename": "wiki_greenwich_village.txt"},
    {"title": "East_Village,_Manhattan", "filename": "wiki_east_village.txt"},
    {"title": "Williamsburg,_Brooklyn", "filename": "wiki_williamsburg_brooklyn.txt"},
    {"title": "Astoria,_Queens", "filename": "wiki_astoria_queens.txt"},
    {"title": "Crown_Heights,_Brooklyn", "filename": "wiki_crown_heights.txt"},
    {"title": "Lower_East_Side", "filename": "wiki_lower_east_side.txt"},
    {
        "title": "Rent_regulation_in_New_York_City",
        "filename": "wiki_nyc_rent_regulation.txt",
    },
]


def fetch_wikipedia(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    r = requests.get(WIKI_API, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    time.sleep(SLEEP)
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No extract returned for '{title}'")
    return page["extract"]


def save(text, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved → {path} ({len(text):,} chars)")


def main():
    for source in SOURCES:
        print(f"\nFetching: {source['title']}")
        try:
            text = fetch_wikipedia(source["title"])
            if len(text.strip()) < 200:
                print(
                    f"  WARNING: very short ({len(text)} chars) — check title spelling"
                )
            save(text, source["filename"])
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone. Check documents/ folder.")


if __name__ == "__main__":
    main()
