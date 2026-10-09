# Public Discourse & Sentiment Analysis on the Impeachment Issue of VP Sara Duterte

Guided **Streamlit wizard** for a student text-mining project: collection →
Taglish preprocessing → similarity → topic modeling → rule-based sentiment →
honest validation. Every number shown is computed from **your data** —
nothing is fabricated. A persistent glossary + notes panel supports the oral defense.

**Python:** 3.10+ (developed & tested on **3.14.0**, which is installed here).

## Folder structure

```
DATA MINING/
├── app.py                  # Streamlit wizard (run this). 8 pages, locked in order.
├── config.py               # ALL benchmark thresholds in one editable place.
├── pipeline_notebook.py    # # %%-cell notebook-style core pipeline (course requirement).
├── requirements.txt        # Pinned-ish dependencies.
├── README.md               # This file.
├── modules/
│   ├── __init__.py
│   ├── helpers.py          # Badges, instruction boxes, save/load, page locking.
│   ├── data_collection.py  # YouTube/Reddit APIs, upload, paste, labeling, kappa.
│   ├── preprocessing.py    # Taglish cleaning pipeline + frequencies.
│   ├── similarity.py       # TF-IDF, cosine, Jaccard, shared/distinctive words.
│   ├── topics.py           # LDA (+LSA toggle), k-search, coherence/perplexity.
│   ├── sentiment.py        # VADER + custom lexicon, negation, net sentiment.
│   ├── evaluation.py       # Holdout lock, 10-fold CV, metrics, error table.
│   └── glossary.py         # 30+ terms: definition, example, formula, how-to-read.
├── data/
│   ├── raw_data.csv        # webscraped dataset (~2751 rows, X/YouTube/Facebook)
│   ├── stopwords_tl.txt    # vendored Tagalog stopwords (offline fallback for stopwordsiso)
│   └── taglish_slang.csv   # editable slang map (variant → canonical)
└── saved_state/            # Save/Load output (state.json, CSVs, my_notes.txt).
```

## Setup (Windows, `py` launcher)

```powershell
# 1. Check Python (want 3.10+; 3.14.0 confirmed on this machine)
py --version

# 2. (Recommended) virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

# 4. NLTK data (English stopwords; app falls back gracefully if skipped)
py -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Run the app
streamlit run app.py
```

Then open the printed `http://localhost:8501` URL. Walk pages **in order** —
later pages unlock as you finish earlier ones.

## API keys (where to put them — nothing is stored to disk)

| Source | How to get a key | Where it goes |
|---|---|---|
| YouTube Data API v3 | Google Cloud Console → new project → enable *YouTube Data API v3* → Credentials → API key | Page 2 tab (a), "YouTube API key" box (session-only) |
| Reddit (PRAW) | reddit.com/prefs/apps → *create app* (type: script) → client_id + secret | Page 2 tab (b) boxes (session-only) |
| Facebook / X / news | **No scraper shipped (ToS).** Copy comments manually or export, then use tab (c) upload or tab (d) paste | Paste/upload boxes |

Quota notes: YouTube enforces daily API quotas (top-level comments only fetched);
PRAW respects Reddit rate limits automatically.

## Typical workflow

1. **Page 1** — read the overview + requirement tracker.
2. **Page 2** — click *Load DEMO DATA* to test, or collect real data (target **200+**,
   label every row with sentiment + bloc). Export CSV for backup.
3. **Page 3** — *Run preprocessing*; inspect before/after + funnel + frequencies.
4. **Page 4** — group by platform/bloc/source; read heatmaps + distinctive words.
5. **Page 5** — *Auto-search k=2..10*, note best k + Coherence; name each topic.
6. **Page 6** — edit the lexicon table, set sliders, *Classify*; read net sentiment.
7. **Page 7** — *Create holdout* (locks) → *Run CV* (tune on dev only) →
   *Final evaluation* once; discuss the error table.
8. **Page 8** — study glossary entries; save defense notes (persisted to
   `saved_state/my_notes.txt`).

Project save/load buttons live in the sidebar (`saved_state/` folder).

## Notebook requirement

`pipeline_notebook.py` uses `# %%` cells:
- VS Code + Jupyter extension → open file → *Run Cell* per block, or
- `pip install jupytext` → `jupytext --to ipynb pipeline_notebook.py`, or
- `py pipeline_notebook.py` (runs end-to-end on demo data).

Point `INPUT_CSV` at your exported `impeachment_dataset.csv` to reproduce real results.

## Thresholds (rule-of-thumb guides, not strict rules)

All cut-offs live in `config.py` and are editable live in the sidebar:

Records red <200 / yellow 200–300 / green >300 · class warn <15% ·
kappa red <0.4 / yellow 0.4–0.6 / green >0.6 · Coherence red <0.35 /
yellow 0.35–0.50 / green >0.50 · cosine low <0.3 / mod 0.3–0.6 / high >0.6 ·
Jaccard low <0.1 / mod 0.1–0.3 / high >0.3 · accuracy/macro-F1 red <0.60 /
yellow 0.60–0.75 / green >0.75 · per-class floor 0.50 · CV SD flag >0.05 ·
overfit flag if CV−holdout >0.10 · neutral zone ±10 net-sentiment points.

## How to test every page works

1. `streamlit run app.py` opens with no errors; sidebar shows 8 pages.
2. Page 1: tracker shows 0 records, RED badge.
3. Page 2: *Load DEMO DATA* → 30 rows, counts per platform, progress 30/200;
   label one row with buttons; export CSV downloads.
4. Page 3: *Run preprocessing* → before/after table + funnel + frequency chart.
5. Page 4: group by platform → *Compute* → two heatmaps + shared/distinctive tables.
6. Page 5: *Auto-search k=2..10* → coherence line chart + best-k badge;
   train → top words + representative docs + share table.
7. Page 6: edit one lexicon score → *Classify* → distribution + net bars per
   platform/bloc change accordingly.
8. Page 7: *Create holdout* → locks; *Run CV* → mean±SD table;
   *Final evaluation* → accuracy/F1, confusion matrix, error table with reasons.
9. Page 8: search "recall" finds the entry; notes save + reload.
10. Sidebar *Save project* then *Load saved* restores state after refresh.
11. `py pipeline_notebook.py` runs end-to-end without errors.

## Troubleshooting

- `ModuleNotFoundError` → re-run `py -m pip install -r requirements.txt` in the
  right venv.
- NLTK stopwords missing → app uses a built-in fallback; run the `nltk.download`
  line above to silence it.
- `bertopic` not installed → fine, it's optional; LDA covers the course.
- Gensim + Python 3.14: if `pip` resolves an old gensim, upgrade with
  `py -m pip install -U gensim scipy numpy`.
- YouTube "quota exceeded" → wait for daily reset or lower max comments.
- Empty results after preprocessing → loosen stopwords / lower min tokens.
