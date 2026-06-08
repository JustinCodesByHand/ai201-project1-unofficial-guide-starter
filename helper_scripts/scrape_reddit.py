"""
Scrape Reddit posts + comments for NYU housing corpus.
Uses Arctic Shift API (Reddit archive, no auth required).
https://arctic-shift.photon-reddit.com
Saves to documents/ as txt files for ingestion.
"""

import requests
import time
import os

HEADERS = {"User-Agent": "nyu-housing-guide-scraper/1.0"}
SLEEP = 1.0
BASE = "https://arctic-shift.photon-reddit.com/api"
OUT_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_json(url, params):
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.json()


def get_posts(subreddit, query, limit=100):
    data = fetch_json(f"{BASE}/posts/search", {
        "subreddit": subreddit,
        "query": query,
        "limit": limit,
        "sort": "desc",
    })
    return data.get("data", [])


def get_comments(post_id, limit=20):
    data = fetch_json(f"{BASE}/comments/search", {
        "link_id": post_id,
        "limit": limit,
        "sort": "desc",
    })
    return data.get("data", [])


def scrape_source(subreddit, queries, out_filename, max_posts=40):
    lines = []
    seen_ids = set()
    total_posts = 0

    for query in queries:
        print(f"  Fetching r/{subreddit} — '{query}'")
        posts = get_posts(subreddit, query, limit=100)

        count = 0
        for post in posts:
            if count >= max_posts:
                break
            pid = post.get("id", "")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            title = post.get("title", "").strip()
            body = post.get("selftext", "").strip()

            if not title:
                continue
            if body in ("[deleted]", "[removed]", ""):
                body = ""

            lines.append(f"POST: {title}")
            if body:
                lines.append(body)

            try:
                comments = get_comments(pid)
                for c in comments[:10]:
                    text = c.get("body", "").strip()
                    if text and text not in ("[deleted]", "[removed]"):
                        lines.append(f"COMMENT: {text}")
            except Exception as e:
                print(f"    comment fetch failed for {pid}: {e}")

            lines.append("")
            count += 1
            total_posts += 1

        print(f"    got {count} posts")

    out_path = os.path.join(OUT_DIR, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {out_path} ({total_posts} posts, {len(lines)} lines)")


def main():
    sources = [
        {
            "subreddit": "nyu",
            "queries": ["off campus housing", "apartment", "sublease", "rent"],
            "filename": "reddit_nyu_housing.txt",
            "max_posts": 40,
        },
        {
            "subreddit": "nyu",
            "queries": ["landlord", "lease", "avoid building", "management"],
            "filename": "reddit_nyu_landlord.txt",
            "max_posts": 40,
        },
        {
            "subreddit": "AskNYC",
            "queries": ["NYU student apartment", "Greenwich Village rent", "East Village student housing"],
            "filename": "reddit_asknyc_nyu.txt",
            "max_posts": 30,
        },
        {
            "subreddit": "nyc",
            "queries": ["broker fee", "rent stabilized", "lease tips", "management company", "apartment hunting"],
            "filename": "reddit_nyc_renting.txt",
            "max_posts": 40,
        },
        {
            "subreddit": "Brooklyn",
            "queries": ["Williamsburg apartment", "Crown Heights rent", "Bushwick housing", "commute Manhattan", "neighborhood safe"],
            "filename": "reddit_brooklyn_housing.txt",
            "max_posts": 30,
        },
    ]

    for s in sources:
        print(f"\nScraping r/{s['subreddit']} → {s['filename']}")
        try:
            scrape_source(
                subreddit=s["subreddit"],
                queries=s["queries"],
                out_filename=s["filename"],
                max_posts=s["max_posts"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone. Check documents/ folder.")


if __name__ == "__main__":
    main()
