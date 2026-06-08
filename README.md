# The Unofficial Guide — Project 1

---

## Domain

NYU off-campus housing in New York City. NYU's official housing portal lists landlord partners and price tiers but provides no qualitative signal — it cannot tell students which landlords ignore maintenance requests, which buildings have roach or mold problems, which neighborhoods feel unsafe at night, or which lease terms are predatory. Students navigating NYC's housing market rely on scattered Reddit threads and word-of-mouth that is hard to search systematically. This RAG system surfaces that experiential knowledge alongside official statistical data and practical guides, so a student can ask a plain-English question — "Is Williamsburg too far from campus?" or "What is a guarantor and do I need one?" — and get a grounded answer drawn from the combined corpus.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/nyu — off-campus housing posts & comments | Informal / Reddit | https://www.reddit.com/r/nyu/ · `documents/reddit_nyu_housing.txt` |
| 2 | r/nyu — landlord, lease, and building complaints | Informal / Reddit | https://www.reddit.com/r/nyu/ · `documents/reddit_nyu_landlord.txt` |
| 3 | r/AskNYC — NYU student apartment advice threads | Informal / Reddit | https://www.reddit.com/r/AskNYC/ · `documents/reddit_asknyc_nyu.txt` |
| 4 | NYC DCP Manhattan Neighborhood Data Profile | Statistical / Official | https://www.nyc.gov/site/planning/planning-level/nyc-population/neighborhood-data-profiles.page · `documents/nyc_manhattan_housing_stats.txt` |
| 5 | NYC DCP Brooklyn Neighborhood Data Profile | Statistical / Official | https://www.nyc.gov/site/planning/planning-level/nyc-population/neighborhood-data-profiles.page · `documents/nyc_brooklyn_housing_stats.txt` |
| 6 | NYC DCP Queens Neighborhood Data Profile | Statistical / Official | https://www.nyc.gov/site/planning/planning-level/nyc-population/neighborhood-data-profiles.page · `documents/nyc_queens_housing_stats.txt` |
| 7 | MTA GTFS Static Feed — subway stops near NYU-area neighborhoods | Transit / Official | http://web.mta.info/developers/data/nyct/subway/google_transit.zip · `documents/nyc_subway_access.txt` |
| 8 | BrickUnderground — How to rent an apartment in NYC as a college or grad student | Formal / Guide | https://www.brickunderground.com/ · `documents/how_to_rent_an_apartment_in_nyc_if_youre_a_college_or_grad_student.txt` |

---

## Chunking Strategy

Two strategies depending on document type, implemented in `ingest.py`:

**Informal (Reddit) — boundary-aware Q:/A: chunker:**
- Chunk size: 500 characters max per chunk
- No sliding overlap — chunks split only at Q: post boundaries
- Each chunk starts with the original Q: question line (repeated if a post's comments span multiple chunks), so every chunk is self-contained and can be understood without surrounding context
- A: lines shorter than 30 chars are dropped as noise
- Preprocessing: `clean_reddit.py` converts raw POST:/COMMENT: format → Q:/A: groups, sorts by topic (neighborhoods, rent, lease, landlord, roommates, transit, safety, process tips), collapses blank lines, strips markdown URLs and spam

**Formal (housing stats, transit, BrickUnderground) — sliding window:**
- Chunk size: 500 characters
- Overlap: 150 characters (preserves context across window boundaries)
- Preprocessing: HTML articles parsed via BeautifulSoup (`<p>` + `<h2>` tags only, skips div duplication); H2 headings written as `=== SECTION: HEADING ===` labels; housing stats rewritten in natural question-answering language so queries like "average rent for a studio" match the document text

**Why these choices fit the documents:**
Reddit posts are conversational Q&A — cutting mid-comment destroys meaning and the question context. Boundary-aware chunking ensures every chunk is tied to an identifiable question. Formal documents are continuous prose where 150-char overlap prevents a key sentence from being split across two chunks that would retrieve separately and give incomplete answers.

**Final chunk count:** 988 total — 917 informal chunks (Reddit) across 3 files; 71 formal chunks across 5 files.

---

### Sample Chunks

**Chunk 1 — Informal (Reddit Nyu Housing)**
```
Q: Incoming Transfer - Off Campus Housing Advice — Hi! I'm an incoming transfer student at NYU
this fall and am trying to figure out off campus housing. My budget is ideally around $1,300 (or
below). I've been keeping an eye on the NYU off campus housing website and Facebook groups, but
most of what I'm seeing right now are summer sublets.
A: People typically use StreetEasy to find apartments in New York.
A: Most landlords don't hold an apartment for more than a few weeks. Have your documents ready
and be prepared to apply right after the showing.
```

**Chunk 2 — Informal (Reddit Asknyc Nyu)**
```
Q: finding an apartment — My friend and I are undergrads at NYU and are planning to live off
campus starting next semester. We've been browsing StreetEasy but haven't booked tours yet
because we're not sure how the timeline works.
A: You generally start searching around 1 month before your move-in. If you like a place, be
ready to go on the spot — have all your paperwork lined up, including financials, references,
and ID.
A: Watch out for broker fees — they can be 12-15% of a year's rent, which is a massive hit
upfront. Always see the place in person before signing anything.
```

**Chunk 3 — Formal (Nyc Manhattan Housing Stats)**
```
RENT PRICES IN MANHATTAN

How much does it cost to rent an apartment in Manhattan? The median monthly rent for a studio
or 1-bedroom apartment in Manhattan is $2,370. The average rent for a 2- or 3-bedroom apartment
is $2,080 per month. Overall median rent across all apartment sizes is $2,290 per month.
Students who recently moved pay more — the median rent for recent movers is $3,290 per month.
```

**Chunk 4 — Formal (Nyc Subway Access)**
```
WILLIAMSBURG, BROOKLYN
Williamsburg is a popular Brooklyn neighborhood for NYU students, about 20-25 minutes from
campus. Subway stations within 0.5 miles include:
  - Marcy Av (0.04 miles) — J/M/Z trains
  - Hewes St (0.21 miles) — J/M trains
  - Metropolitan Av (0.44 miles) — L train
Commute to NYU Washington Square: approximately 20-25 minutes via L train to 14th St.
```

**Chunk 5 — Formal (BrickUnderground)**
```
=== SECTION: LINE UP A GUARANTOR ===

Often enough, the only way students can rent in NYC is by having a guarantor — usually a parent
or guardian — who agrees to be legally responsible for the rent in case you default. A guarantor
must typically earn 80 times the monthly rent. That means for an apartment with a monthly rent
of $4,000, a guarantor must earn $320,000.
If you don't have a relative who can step in, you can use an institutional guarantor like
Insurent, which has lower annual income requirements and is accepted in around 9,000 buildings.
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` (sentence-transformers, runs locally, no API cost)

**Production tradeoff reflection:**
`all-MiniLM-L6-v2` has a 256-token context limit and was trained on general English text. For a real deployment serving NYU students, I would weigh: (1) **context length** — our formal chunks are ~500 chars (~125 tokens), which fits safely, but longer documents would require a model with 512+ token limits like `all-mpnet-base-v2`; (2) **domain specificity** — a model fine-tuned on real-estate or legal text would embed terms like "rent stabilization," "guarantor," and "broker fee" more precisely, improving retrieval on those queries where vocabulary mismatch is the main failure mode; (3) **latency** — local inference adds startup time but removes API round-trips, acceptable for a student tool but risky for a production web app under load; (4) **multilingual support** — NYU has a large international student population; a multilingual model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) would serve non-native English queries better without requiring translated documents.

---

## Grounded Generation

**System prompt grounding instruction:**
The system prompt in `generator.py` reads: *"You are a helpful assistant for NYU students looking for off-campus housing in New York City. Answer using only the context provided below. When the answer is present in the context, state it directly and confidently — do not hedge or say you cannot answer. If the answer is genuinely not in the context, say so clearly — do not guess or draw on outside knowledge. Always state which numbered source your answer comes from."*

Context is injected as numbered blocks before the user question:
```
[1] (Source Name)
{chunk text}

[2] (Source Name)
{chunk text}
...

Question: {user query}
```

**Structural grounding choices:**
- Retrieval applies a cosine distance threshold of 0.5 — chunks above this are discarded before the LLM sees them. If all chunks exceed the threshold, `query.py` returns "No relevant documents found" without calling the LLM at all, preventing hallucination on completely out-of-scope questions.
- `N_RESULTS = 5` (set in `config.py`) retrieves 5 chunks. Two-pass retrieval ensures formal sources (stats, guides) are always considered alongside Reddit posts.
- Formal chunks receive a 0.70 distance multiplier so statistically precise documents compete against the higher-volume Reddit corpus.

**How source attribution is surfaced in the response:**
The LLM is instructed to cite numbered sources inline (e.g., "according to source [2]"). The `ask()` function in `query.py` also returns a `sources` list with source name, boosted distance score, and a 120-character text preview, displayed as a separate "Retrieved from" field in the Gradio UI.

---

## Retrieval Test Results

### Query 1: "What is the typical price range for a studio or 1-bedroom apartment within walking distance of NYU?"

Top returned chunks:
1. `Nyc Manhattan Housing Stats` (dist: 0.218) — *"NYC MANHATTAN HOUSING STATISTICS FOR STUDENTS... The median monthly rent for a studio or 1-bedroom apartment in Manhattan is $2,370..."*
2. `Nyc Manhattan Housing Stats` (dist: 0.278) — *"The median monthly rent for a studio or 1-bedroom apartment in Manhattan is $2,370. Students who recently moved pay $3,290..."*
3. `Nyc Brooklyn Housing Stats` (dist: 0.291) — *"RENT PRICES IN BROOKLYN... median monthly rent for a studio or 1-bedroom apartment in Brooklyn is $1,780..."*
4. `Reddit Asknyc Nyu` (dist: 0.296) — *"Q: What is market price for summer 2 month apartment rental in Manhattan..."*
5. `Nyc Queens Housing Stats` (dist: 0.306) — *"...median monthly rent for a studio or 1-bedroom apartment in Queens is $1,820..."*

**Why these chunks are relevant:** The query uses "price range" and "studio or 1-bedroom" — the housing stats documents were rewritten to use this exact natural language ("How much does it cost to rent a studio or 1-bedroom apartment in Manhattan?"), enabling a direct embedding match. All three borough stat files surface because the query doesn't specify a borough. The Reddit chunk provides a real-world anecdote. Two-pass retrieval was necessary here — before the fix, Reddit posts scored lower raw distance (0.310) than the stats files (0.356), so stats never entered the candidate pool until the formal-only second pass forced them in.

---

### Query 2: "What should NYU students know about broker fees and lease terms before signing an NYC apartment lease?"

Top returned chunks:
1. `How To Rent An Apartment In Nyc...` (dist: 0.243) — *"...co-living operations that welcome students... limit your search to no-fee apartments..."*
2. `How To Rent An Apartment In Nyc...` (dist: 0.303) — *"TITLE: A guide to renting in NYC as a college or grad student..."*
3. `Reddit Nyu Housing` (dist: 0.324) — *"Q: NYU off campus Housing — looking for another student to sign a lease..."*
4. `Nyc Manhattan Housing Stats` (dist: 0.337) — *"NYC MANHATTAN HOUSING STATISTICS FOR STUDENTS..."*
5. `Reddit Asknyc Nyu` (dist: 0.357) — *"Q: finding an apartment — broker fees can be 12-15% of a year's rent..."*

**Why these chunks are relevant:** "Broker fees" and "lease terms" are explicit phrases in both the BrickUnderground article and Reddit posts, so the embedding similarity is high. The BrickUnderground article has a dedicated section on no-fee apartments and lease assignment that directly answers this query. The Reddit chunk contains practical student advice on broker fees (12-15%) observed through lived experience — a different perspective than the formal guide.

---

### Query 3: "How do I get a guarantor if my parents are overseas?"

Top returned chunks:
1. `How To Rent An Apartment In Nyc...` (dist: 0.364) — *"...is someone who promises to pay your rent if you don't. If you don't have a relative who can step in, you can use an institutional guarantor like Insurent..."*
2. `How To Rent An Apartment In Nyc...` (dist: 0.374) — *"...Insurent normally prequalifies renters 24/7 within one hour... For international students without US credit, the cost is 98.4% of one month's gross rent..."*
3. `How To Rent An Apartment In Nyc...` (dist: 0.394) — *"...if you are an international student who will be looking for a new place in the fall, you should try to secure qualified by an institutional guarantor as soon as possible..."*
4. `Reddit Nyu Landlord` (dist: 0.422) — *"Q: Please break it down for an out of state student — I'm an incoming grad student..."*
5. `Reddit Asknyc Nyu` (dist: 0.433) — *"...being an international student will be a very big and very hard to deal with issue. You'll need documentation of your parents income..."*

---

## Example Responses

### Example 1 — Broker fees (Q3 from evaluation plan)

**Query:** "What should NYU students know about broker fees and lease terms before signing an NYC apartment lease?"

**Response:**
> According to source [1] (How To Rent An Apartment In NYC), broker fees in the New York City rental market can be a significant cost, as brokers hired to represent the landlord's interests get paid by the renter — this can be one month's rent or up to 15 percent of the year's total rent. Source [1] advises limiting your search to no-fee apartments to avoid this cost entirely, and notes buildings like Silver Towers and StuyTown advertise their no-fee status.
>
> Source [5] (Reddit Asknyc Nyu) adds that you should be prepared with all paperwork — financials, references, and ID — and be ready to move the same day you see a place. International students may need to document parents' income and should expect some landlords to require a US-based guarantor.
>
> Sources [1] and [5] both recommend seeing the apartment in person before signing anything, as the actual condition often differs from online listings.

*Sources cited: How To Rent An Apartment In Nyc If Youre A College Or Grad Student · Reddit Asknyc Nyu*

---

### Example 2 — Rent prices (Q5 from evaluation plan)

**Query:** "What is the typical price range for a studio or 1-bedroom apartment within walking distance of NYU Washington Square campus?"

**Response:**
> The median monthly rent for a studio or 1-bedroom apartment in Manhattan is $2,370, according to sources [1] and [2] (Nyc Manhattan Housing Stats). Students who recently signed new leases pay more — the median rent for recent movers is $3,290 per month.
>
> If you are willing to commute, Brooklyn offers significantly lower rents: source [3] (Nyc Brooklyn Housing Stats) puts the median studio or 1-bedroom at $1,780 per month, with recent movers paying $2,650. Queens median is $1,820 per month (source [5], Nyc Queens Housing Stats).

*Sources cited: Nyc Manhattan Housing Stats · Nyc Brooklyn Housing Stats · Nyc Queens Housing Stats*

---

### Example 3 — Out-of-scope refusal

**Query:** "What are the best pizza places near NYU?"

**Response:**
> The provided context does not contain information about the best pizza places near NYU.

*No sources returned. System correctly declined rather than hallucinating restaurant recommendations.*

---

## Query Interface

The system exposes a Gradio web UI launched by `app.py` at `http://127.0.0.1:7860`.

**Input:** A single text field labeled "Your question" — accepts any plain-English question about NYU off-campus housing. Submits on Enter or clicking the "Ask" button.

**Output — two fields:**
- **Answer** (8-line text area): The LLM-generated response grounded in retrieved chunks, with inline source citations by number.
- **Retrieved from** (4-line text area): One bullet per retrieved chunk showing source name, boosted cosine distance, and a 120-character text preview.

**Sample interaction transcript:**

> **Input:** How long is the commute from Williamsburg to NYU?
>
> **Answer:** According to source [1] (Nyc Subway Access), the commute from Williamsburg to NYU Washington Square is approximately 20-25 minutes via the L train to 14th St. Subway stations within 0.5 miles of Williamsburg include Marcy Av (0.04 miles, J/M/Z trains), Hewes St (0.21 miles, J/M trains), and Metropolitan Av (0.44 miles, L train).
>
> **Retrieved from:**
> • Nyc Subway Access (dist: 0.271) — Williamsburg is a popular Brooklyn neighborhood for NYU students, about 20-25 minutes from campus...
> • Reddit Nyu Housing (dist: 0.318) — Q: Where do NYU Tandon Master's students live?...
> • Reddit Asknyc Nyu (dist: 0.334) — Q: Move from GV to Williamsburg?...

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which neighborhoods near NYU do students most recommend for off-campus housing, and why? | Names 1–3 neighborhoods (East Village, Crown Heights, Astoria) with student-cited reasons | Mentioned Bay Ridge, Dyker Heights, East Williamsburg, Greenwich Village — mixed; safety-focused post pulled in unusual neighborhoods (Bay Ridge) not typical student picks | Partially relevant — safety/crime post matched "recommend neighborhoods" query | Partially accurate — includes correct options (East Williamsburg) alongside unusual ones pulled from wrong intent |
| 2 | What do NYU students say about renting from Rose Associates management near campus? | Specific complaints or praise about Rose Associates' responsiveness, maintenance, lease practices | "Not in context" — corpus contains no Rose Associates-specific content | Off-target — no landlord review data in corpus | Accurate (honest refusal) — correctly declined rather than fabricating a review |
| 3 | What should NYU students know about broker fees and lease terms before signing an NYC apartment lease? | Broker fee laws, security deposit limits, lease red flags from guide or Reddit | Covered broker fees (up to 15%), no-fee buildings, paperwork prep, international student challenges; BrickUnderground as primary source | Relevant — BrickUnderground guide retrieved correctly | Accurate — comprehensive answer with specific figures and actionable advice |
| 4 | Which buildings or areas near NYU are frequently cited as having noise, safety, or maintenance problems? | Specific buildings or streets from Reddit threads with cited issue | "Not in context" — stats docs retrieved instead of Reddit complaint posts; gave HPD tip | Off-target — housing code stats retrieved, not building-specific complaints | Accurate (honest refusal) — corpus has no building-level complaint data; correctly declined |
| 5 | What is the typical price range for a studio or 1-bedroom apartment within walking distance of NYU? | Price range from housing data or Reddit, with neighborhood breakdown | "$2,370/month median for studio/1BR in Manhattan (sources [1] and [2]); Brooklyn $1,780, Queens $1,820" — direct and well-sourced | Relevant — all three borough stats files plus Reddit retrieved | Accurate — correct figures, multi-borough comparison, proper citation |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**
"Which buildings or areas near NYU are frequently cited as having noise, safety, or maintenance problems?" (Q4)

**What the system returned:**
Retrieved Manhattan housing stats and subway access data — neither contains building-specific complaint information. The LLM correctly said "not in context" but could only offer a generic tip about checking HPD records.

**Root cause (tied to a specific pipeline stage):**
This is a corpus gap, not a retrieval or generation bug. The original plan (planning.md sources 9–10) called for scraped Yelp/Google reviews of specific NYU-area buildings and landlord-specific Reddit threads. Those sources were never collected — Yelp requires authentication, and Reddit scraping returned general housing advice rather than building-specific complaint posts. The query "noise, safety, maintenance problems" semantically matched housing stats (which contain crime rate and violation data) rather than the conversational Reddit complaints that actually name buildings, because no such chunks exist in the index.

**What you would change to fix it:**
Collect building-specific data: (1) scrape HPD violation records by address for the 20–30 buildings most commonly mentioned in NYU Reddit threads and store as structured text; (2) add targeted Reddit scrapes with building names and addresses as queries (e.g., "20 Cooper Square", "Silver Towers complaints") rather than general topic queries. The corpus gap is the binding constraint — no retrieval improvement can surface data that was never ingested.

---

## Spec Reflection

**One way the spec helped you during implementation:**
The chunking strategy section of planning.md forced an early decision about document type detection and chunk size before any code was written. Specifying that informal sources use 0 overlap (because Reddit comments are self-contained) and formal sources use 150-char overlap (because legal and statistical text carries meaning across sentence boundaries) meant the ingest pipeline had clear branching logic from the start. When the chunker was implemented, the `doc_type` detection by filename pattern (`reddit`, `review`, `post`, `thread` → informal; everything else → formal) mapped directly from the spec, avoiding ambiguity about how to classify new files added later.

**One way your implementation diverged from the spec, and why:**
The spec planned 10 sources including StreetEasy neighborhood guides, NYC.gov tenant rights documents, Washington Square News articles, and Yelp/Google landlord reviews. In practice these were replaced by NYC Department of City Planning neighborhood data profiles (Excel downloads), MTA GTFS transit data, and a BrickUnderground guide. StreetEasy's guides are rendered client-side JavaScript and aren't scrapable with requests; NYC.gov tenant rights pages returned boilerplate navigation text rather than legal content; Yelp requires authentication. The statistical data sources (DCP, MTA) were more reliable and machine-readable even though they required more preprocessing, and BrickUnderground had denser practical content per page than the student newspaper articles originally planned.

---

## AI Usage

**Instance 1 — Reddit scraper**

- *What I gave the AI:* The five target subreddits (r/nyu, r/AskNYC, r/nyc, r/Brooklyn) and a list of search queries per subreddit. I asked Claude to write a scraper that fetched posts and their top comments.
- *What it produced:* A scraper using Reddit's JSON API (`reddit.com/r/{sub}/search.json`). This returned 403 errors because Reddit blocks unauthenticated API access.
- *What I changed or overrode:* Redirected Claude to use the Arctic Shift archive API (`arctic-shift.photon-reddit.com`) instead, which doesn't require authentication. I also changed the query parameter from `"q"` to `"query"` after the first run returned empty results — the API used a non-standard parameter name.

**Instance 2 — Boundary-aware chunker**

- *What I gave the AI:* The working ingest.py with a fixed 500-char sliding window chunker applied to all documents, plus the cleaned Reddit files showing their Q:/A: structure. I said I wanted smaller chunks for Reddit that don't cut mid-comment.
- *What it produced:* A `_chunk_informal()` function that splits at Q: boundaries, packs A: lines up to 500 chars, and repeats the Q: line at the top of each new chunk so every chunk is self-contained. It also added a 30-char minimum for A: lines to filter noise.
- *What I changed or overrode:* Kept the chunk size at 500 chars (Claude initially proposed 200 to match a planning.md note about informal chunk size). The 200-char limit produced too many tiny single-comment chunks — 500 with boundary-awareness gave better context per chunk without the duplication problem.
