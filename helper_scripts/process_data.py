"""
Convert raw downloaded data files into RAG-friendly txt summaries.
Outputs to documents/ alongside the Reddit txt files.

Input files expected in documents/:
  MN_NeighborhoodDataProfile.xlsx   NYC DCP Manhattan housing stats
  BK_NeighborhoodDataProfile.xlsx   NYC DCP Brooklyn housing stats
  QN_NeighborhoodDataProfile.xlsx   NYC DCP Queens housing stats (download if available)
  stops.txt                         MTA GTFS subway stops
"""

import os
import re
import math
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup

OUT_DIR = os.path.join(os.path.dirname(__file__), "documents")
DOCS = lambda f: os.path.join(OUT_DIR, f)

WIKI_API = "https://en.wikipedia.org/w/api.php"


# ─── helpers ───────────────────────────────────────────────────────────────────

def save(text, filename):
    path = DOCS(filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved -> {path} ({len(text):,} chars)")


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ─── 1. Manhattan / Brooklyn / Queens housing stats ────────────────────────────

HOUSING_INDICATORS = [
    "Median rent, studios and 1-bedrooms (2025$)",
    "Median rent, 2- and 3-bedrooms (2025$)",
    "Median rent, all (2025$)",
    "Median rent, recent movers (2025$)",
    "Rental vacancy rate",
    "Severely rent-burdened households",
    "Moderately rent-burdened households",
    "Serious housing code violations (per 1,000 privately owned rental units)",
    "Total housing code violations (per 1,000 privately owned rental units)",
    "Serious crime rate (per 1,000 residents)",
    "Serious crime rate, violent (per 1,000 residents)",
    "Serious crime rate, property (per 1,000 residents)",
    "Car-free commute (% of commuters)",
    "Mean travel time to work (minutes)",
    "Median household income, renters (2025$)",
    "Poverty rate",
    "Unemployment rate",
    "Homeownership rate",
    "Population",
    "Rental units affordable at  30% AMI (% of recently available units)",
    "Rental units affordable at  80% AMI (% of recently available units)",
]


def process_neighborhood_profile(excel_file, sheet_name, borough_name, out_filename):
    print(f"\nProcessing {borough_name} neighborhood profile...")
    df = pd.read_excel(DOCS(excel_file), sheet_name=sheet_name, header=None)

    # Row 0 = headers, rows 1+ = data
    # Cols: 0=CD, 1=Name, 2=Category, 3=Indicator, 4=Description, 5=2000, 6=2006, 7=2010, 8=2019, 9=2023, 10=2024, 11=2025
    df.columns = ["CD", "Name", "Category", "Indicator", "Description",
                  2000, 2006, 2010, 2019, 2023, 2024, 2025]
    df = df.iloc[1:].reset_index(drop=True)

    lines = [
        f"{borough_name.upper()} HOUSING & NEIGHBORHOOD STATISTICS",
        "Source: NYC Department of City Planning Neighborhood Data Profiles (2025)",
        "=" * 60,
        "",
    ]

    # Group by category
    categories = ["Renters and Rental Conditions", "Housing", "Neighborhood Services and Conditions", "Demographics"]
    for cat in categories:
        cat_df = df[df["Category"] == cat]
        if cat_df.empty:
            continue
        lines.append(f"\n{cat.upper()}")
        lines.append("-" * 40)
        for _, row in cat_df.iterrows():
            indicator = str(row["Indicator"]).strip()
            val_2025 = row[2025]
            val_2024 = row[2024]
            val_2023 = row[2023]
            if indicator not in HOUSING_INDICATORS:
                continue
            # Use most recent available year
            val = val_2025 if pd.notna(val_2025) else (val_2024 if pd.notna(val_2024) else val_2023)
            val_str = str(val) if pd.notna(val) else "N/A"
            lines.append(f"  {indicator}: {val_str}")

    lines += [
        "",
        f"Note: Statistics represent {borough_name} borough-wide averages.",
        "Individual neighborhoods vary — use these as baseline reference.",
    ]

    save("\n".join(lines), out_filename)


# ─── 3. MTA subway stops near NYU neighborhoods ────────────────────────────────

NEIGHBORHOODS = {
    "Greenwich Village (NYU Washington Square campus)": (40.7295, -73.9965),
    "East Village": (40.7264, -73.9818),
    "Lower East Side": (40.7157, -73.9863),
    "Williamsburg, Brooklyn": (40.7081, -73.9571),
    "Crown Heights, Brooklyn": (40.6678, -73.9442),
    "Astoria, Queens": (40.7721, -73.9302),
    "Bushwick, Brooklyn": (40.6944, -73.9213),
}


def process_subway():
    print("\nProcessing MTA subway stops...")
    stops = pd.read_csv(DOCS("stops.txt"))
    # Only parent stations (location_type == 1 or no parent)
    parent = stops[stops["location_type"] == 1.0].copy()

    lines = [
        "NYC SUBWAY ACCESS BY NEIGHBORHOOD",
        "Source: MTA GTFS Static Feed",
        "=" * 50,
        "",
        "Subway stations within 0.5 miles of key NYU-area neighborhoods.",
        "Note: NYU's Washington Square campus is in Greenwich Village.",
        "NYU Tandon (engineering) is in Downtown Brooklyn.",
        "",
    ]

    for hood, (lat, lon) in NEIGHBORHOODS.items():
        parent["dist"] = parent.apply(
            lambda r: haversine_miles(lat, lon, r["stop_lat"], r["stop_lon"]), axis=1
        )
        nearby = parent[parent["dist"] <= 0.5].sort_values("dist")
        lines.append(f"NEIGHBORHOOD: {hood}")
        if nearby.empty:
            lines.append("  No subway stations within 0.5 miles.")
        else:
            for _, s in nearby.iterrows():
                lines.append(f"  • {s['stop_name']} ({s['dist']:.2f} mi)")
        lines.append("")

    lines += [
        "Commute context:",
        "  Greenwich Village → NYU Washington Square: 0 min (on campus)",
        "  East Village → NYU Washington Square: ~10 min walk or 1 stop on 6 train",
        "  Williamsburg → NYU Washington Square: ~20-25 min via L train to 14th St",
        "  Crown Heights → NYU Washington Square: ~30-35 min via 3 train",
        "  Astoria → NYU Washington Square: ~30-40 min via N/W train",
        "  Bushwick → NYU Washington Square: ~30-35 min via L train",
    ]

    save("\n".join(lines), "nyc_subway_access.txt")


# ─── 4. Wikipedia — rent regulation ────────────────────────────────────────────

def fetch_wikipedia(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    r = requests.get(WIKI_API, headers={"User-Agent": "nyu-housing-guide/1.0"}, params=params, timeout=20)
    r.raise_for_status()
    time.sleep(1.0)
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No extract for '{title}'")
    return page["extract"]


def process_wiki_rent_regulation():
    print("\nFetching Wikipedia — Rent regulation in New York City...")
    try:
        text = fetch_wikipedia("Rent_regulation_in_New_York_City")
        save(text, "wiki_nyc_rent_regulation.txt")
    except Exception as e:
        print(f"  FAILED: {e}")


# ─── 5. HTML articles ──────────────────────────────────────────────────────────

_ARTIFACTS = [
    ('�', "'"),   # replacement char → apostrophe
    ('’', "'"),
    ('‘', "'"),
    ('“', '"'),
    ('”', '"'),
    ('–', '-'),
    ('—', '--'),
    (' ', ' '),
]


# Ad block signals — skip any <p> whose text starts with these
_AD_SIGNALS = re.compile(
    r'^(Pro Tip|Want peace of mind|With Blueground|Need help finding the perfect starter|'
    r'liveohana\.ai|sign up here)',
    re.IGNORECASE
)

# Stop extracting at these signals
_STOP_SIGNALS = re.compile(
    r'(You Might Also Like|Brick Underground articles occasionally|More from Brick)',
    re.IGNORECASE
)

# Calculator widget labels to skip
_CALCULATOR = re.compile(
    r'^(Gross Rent Calculator|What.s this\?|Net Monthly Advertised Rent|'
    r'Length of Lease|Number of Free Months|Your Real Monthly Rent|Per Month$|Months$|'
    r'If the landlord is offering partial)',
    re.IGNORECASE
)


def _fix_artifacts(text):
    for bad, good in _ARTIFACTS:
        text = text.replace(bad, good)
    return text


def _join_paragraph(raw):
    """Collapse internal whitespace/newlines from HTML column wrapping."""
    text = re.sub(r'\s+', ' ', raw).strip()
    return _fix_artifacts(text)


def _parse_html_article(html):
    """
    Extract article content using tag structure.
    H2 headings become === SECTION: heading === labels.
    <p> tags become paragraphs. Ad blocks and calculator widget skipped.
    """
    soup = BeautifulSoup(html, 'html.parser')
    h1 = soup.find('h1')
    if not h1:
        return ''

    title = _fix_artifacts(h1.get_text(strip=True))
    lines = [f"TITLE: {title}", ""]

    in_article = False
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p']):
        if tag == h1:
            in_article = True
            continue
        if not in_article:
            continue

        text = _join_paragraph(tag.get_text(separator=' ', strip=True))
        if not text:
            continue

        if _STOP_SIGNALS.search(text):
            break
        if _AD_SIGNALS.match(text) or _CALCULATOR.match(text):
            continue
        if re.match(r'^\[An earlier version', text) or re.match(r'^\d+.?\d* minutes?', text):
            continue

        if tag.name == 'h2':
            lines.append(f"\n=== SECTION: {text.upper()} ===\n")
        elif tag.name in ('h3', 'h4'):
            pass
        else:
            lines.append(text)

    return '\n'.join(lines).strip()


def process_html_files():
    """Parse any .htm/.html files in documents/ and save as section-labeled .txt."""
    html_files = [f for f in os.listdir(OUT_DIR) if f.lower().endswith(('.htm', '.html'))]
    if not html_files:
        print("\nNo HTML files found — skipping.")
        return

    print(f"\nProcessing {len(html_files)} HTML file(s)...")
    for fname in html_files:
        src = os.path.join(OUT_DIR, fname)
        stem = re.sub(r'[^\w\s-]', '', os.path.splitext(fname)[0])
        out_name = re.sub(r'\s+', '_', stem.strip().lower()) + '.txt'
        try:
            with open(src, encoding='utf-8', errors='replace') as f:
                html = f.read()
            clean = _parse_html_article(html)
            save(clean, out_name)
            print(f"    -> {out_name}")
        except Exception as e:
            print(f"  FAILED {fname}: {e}")


# ─── main ───────────────────────────────────────────────────────────────────────

def main():
    process_neighborhood_profile(
        "MN_NeighborhoodDataProfile.xlsx", "MN Data",
        "Manhattan", "nyc_manhattan_housing_stats.txt"
    )
    process_neighborhood_profile(
        "BK_NeighborhoodDataProfile.xlsx", "BK Data",
        "Brooklyn", "nyc_brooklyn_housing_stats.txt"
    )
    qn_path = DOCS("QN_NeighborhoodDataProfile.xlsx")
    if os.path.exists(qn_path):
        process_neighborhood_profile(
            "QN_NeighborhoodDataProfile.xlsx", "QN Data",
            "Queens", "nyc_queens_housing_stats.txt"
        )
    else:
        print("\nQueens profile not found — skipping. Download from NYC DCP neighborhood data profiles page.")
    process_subway()
    process_wiki_rent_regulation()
    process_html_files()
    print("\nDone. Check documents/ folder.")


if __name__ == "__main__":
    main()
