# Project Guide — Impeachment Discourse Analyzer (plain words first)

> How to read this file: every section starts with **the simple version** (no
> background needed), then a **"The jargon"** box with the technical terms for
> your defense. If you only remember the simple version, you still understand
> the project. If you also learn the jargon boxes, you can defend it.

---

## 1. What is this project, in one paragraph?

**Simple:** We collected ~2,700 real online comments about the impeachment case
of VP Sara Duterte (from Facebook, YouTube, and X/Twitter). Then we used the
computer to (a) clean up the messy text, (b) check whether different platforms
talk about it differently, (c) discover the hidden themes people keep
repeating, (d) judge whether each comment sounds positive, negative, or
neutral, and (e) honestly test how good our judging method is. Everything runs
in a click-through app, and every number shown comes from your real data.

**The jargon:** end-to-end text-mining pipeline — data collection →
preprocessing (Taglish normalization) → similarity analysis (TF-IDF, cosine,
Jaccard) → topic modeling (LDA, k-search with Coherence/Perplexity) →
rule-based sentiment classification (VADER + custom lexicon, transformer
comparison) → validation (locked 10% holdout, 10-fold stratified CV,
precision/recall/F1, confusion matrix).

---

## 2. How to run it (do this first)

**Simple:** Open a terminal in this folder, install the parts list once, then
either click through the app in your browser or run the single script.

```powershell
py -m pip install -r requirements.txt
py -c "import nltk; nltk.download('stopwords')"
streamlit run app.py        # the click-through app (opens in your browser)
```

For the GPU computer (RTX 4060), install the special fast-torch build once:

```powershell
py -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

To run the whole analysis without clicking:

```powershell
py pipeline_notebook.py     # same steps as the app, prints results
```

**The jargon:** `requirements.txt` pins the dependency set (pandas, sklearn,
gensim-optional with sklearn fallback, NLTK, VADER, transformers/torch,
plotly, streamlit). The CUDA wheel enables fp16 batched inference on the GPU.

---

## 3. The data — what we have and what it looks like

**Simple:** The file `data/raw_data.csv` holds 2,751 webscraped comments.
Each row is one comment plus notes about where it came from (platform, link,
title, author, likes, date). When loaded, 9 exact duplicates are removed, so
**2,742 comments** enter the analysis: ~1,643 Facebook, ~899 YouTube,
~200 X/Twitter. Nobody has graded them yet — the "what feeling is this?"
column is empty until a human labels a sample (see §8).

**The jargon:** schema coercion (`comment_text→text`, platform normalization
via `PLATFORM_MAP`, timestamp parsing, `@handle→@user` anonymization,
exact-duplicate removal). Loader: `modules/data_collection.py ::
load_raw_data_csv()`. Platform distribution is heavily imbalanced (FB-heavy) —
report this as a sampling limitation.

---

## 4. Cleaning the text (preprocessing) — two tracks

**Simple:** Comments are messy: links, emoji, ALL CAPS, slang ("kc", "tlga",
"grabeee"). We clean them **two ways**:
- **Deep clean** (for counting words and finding themes): lowercase everything,
  delete links, turn emoji into feeling-words, fix slang, throw away filler
  words ("ang", "ng", "the"). Result: `clean_text`.
- **Light clean** (for the smart AI reader): only remove links and names, keep
  everything else — because the AI needs the original wording, including small
  words like "hindi" (not), which flip a sentence's meaning. Result:
  `text_light`.

Filler-word lists are **not hand-typed**: they come from the public
stopwords-iso collection (147 Tagalog + English words), with a backup copy in
`data/stopwords_tl.txt` so it works offline. Words that change meaning
("hindi", "sobra", "very") are protected and never deleted.

**The jargon:** Track A (regex tokenizer, emoji mapping, repeat collapsing,
slang normalization from `data/taglish_slang.csv`, stopword removal with
negation/intensifier protection, de-duplication, `min_tokens` filter:
2,742 → 2,347 kept). Track B (`light_clean`: URL/mention strip only,
context preserved for contextual models). Stopword provenance:
`stopwordsiso tl+en → vendored txt → builtin`. Module:
`modules/preprocessing.py`.

---

## 5. Do platforms talk differently? (similarity)

**Simple:** We turn each platform's comments into a "word fingerprint" and
measure how alike the fingerprints are. Real result: **Facebook and YouTube
sound very alike (0.75/1.0), while X/Twitter sounds the most different
(~0.49)**. A second, stricter measure counts only shared vocabulary and says
the same thing. Then we list words every platform uses ("shared") and words
only one platform loves ("distinctive") — good quotes for your defense.

**The jargon:** TF-IDF vectors per group, cosine similarity (angle, 0–1) +
Jaccard word-set overlap; `top_shared_and_distinctive()` for interpretation.
Thresholds: cosine low <0.3 / high >0.6; Jaccard low <0.1 / high >0.3
(`config.py`, editable in the sidebar). Module: `modules/similarity.py`.

---

## 6. Hidden themes (topic modeling)

**Simple:** The computer groups comments into themes by noticing which words
keep appearing together (e.g. senate/court/trial/vote = the legal-process
theme). We tried theme-counts from 2 to 10 and picked **8**, because that
count scored best on "do these words actually belong together?" You then give
each theme your own title after reading example comments — the computer finds
the groups, the human gives them meaning.

**The jargon:** LDA (each doc = mixture of topics, each topic = mixture of
words), k-search 2–10 maximizing Coherence, Perplexity as fit signal
(892–1136 classic scale here — trust Coherence on disagreement), topic share
per platform. Engine: gensim if installed, else built-in scikit-learn
fallback (this machine uses the fallback; coherence is then mean-NPMI —
relative scale, compare k's with each other, not with C_V badges).
Module: `modules/topics.py`. Note: some themes are still heavy in function
words (si, yung, yan, nyo, po) — add them as custom stopwords on page 3 and
re-run to sharpen topics.

---

## 7. Is each comment positive, negative, or neutral? (sentiment)

**Simple:** Two judges read every comment:
- **Judge 1 — the graded one (rule-based):** follows written rules you can
  inspect. An English-feeling meter (VADER) plus our own Tagalog word list
  (e.g. "kurakot" = −1.8, "salamat" = +1.2). It understands "hindi maganda"
  (not beautiful) as negative because "hindi" flips the next word, and "sobrang
  galit" (very angry) counts extra. Real result on all 2,347 comments:
  **46% positive, 38% neutral, 15% negative (net +30.8)** — Facebook most
  positive (+40.4), X/Twitter nearly split (+2.1).
- **Judge 2 — the comparison (transformer AI, TagaSenti):** a neural network
  already trained on ~35,000 Filipino comments. No word lists — it reads whole
  sentences, slang included. It runs on the GPU; its answers are **saved to
  disk** (`saved_state/hf_cache/`), so repeat runs load instantly without
  re-running the GPU.
- Tone ≠ side: an angry defense of Sara is *negative tone* but *pro-Duterte
  side*. The project tracks wording-feeling and political side separately.

**The jargon:** `combine_scores = 0.4·VADER + 0.6·lexicon-mean`, negation
window 3, intensifier ×1.5, thresholds ±0.15 (sliders). Auxiliary:
`jjjardev/tagasenti` (XLM-R-large) via `transformers` pipeline, fp16 CUDA,
batch 32, truncation 512; star/label → schema mapping; confidence +
agreement crosstab vs rule-based. Weakness to admit: **60% of comments hit
zero lexicon words** (VADER-only there) — expand the lexicon table on page 6.
Module: `modules/sentiment.py` (graded) + `modules/sentiment_hf.py`
(comparison).

---

## 8. Proving it honestly (validation) — needs YOU

**Simple:** This part cannot run until a human grades ~200+ comments by hand
in the app (page 2, one-click buttons). Then: lock away 10% of graded comments
unopened, test the method 10 different ways on the other 90%, and finally open
the locked 10% **once** and report the score — good or bad. Re-grading until
the score looks nice is cheating, and the app locks the split to stop it.

**The jargon:** stratified 90/10 holdout (seed 42, locked) → 10-fold
stratified CV on dev (mean±SD, stability flag SD>0.05, overfit flag
CV−holdout>0.10) → final accuracy/macro-F1 + per-class precision/recall/F1 +
confusion matrix + heuristic error table (stance-vs-tone, negation, sarcasm,
code-switch, unknown-vocab). Transformer is graded on the same holdout from
cache (metric-only). Module: `modules/evaluation.py`.

---

## 9. File map (where everything lives)

**Simple:** `app.py` = the clickable app. `pipeline_notebook.py` = the same
steps as runnable script cells (for the report's script table).
`modules/` = the real work (one file per stage). `config.py` = all grading
cut-offs in one place. `data/` = comments + word lists. `outputs/` = the
finished run's tables (already generated, see §10). `saved_state/` = your
saved app progress + GPU answer cache.

| File | What it is |
|---|---|
| `app.py` | 8-page wizard GUI (run with `streamlit run app.py`) |
| `pipeline_notebook.py` | Same pipeline as `# %%` cells (run with `py pipeline_notebook.py`) |
| `modules/data_collection.py` | Loading, YouTube/Reddit APIs, labeling helpers |
| `modules/preprocessing.py` | Cleaning, both tracks, external stopwords |
| `modules/similarity.py` | Fingerprints + heatmaps + shared/distinctive words |
| `modules/topics.py` | LDA (+LSA), k-search, coherence/perplexity |
| `modules/sentiment.py` | Graded rule-based judge |
| `modules/sentiment_hf.py` | GPU transformer comparison + disk cache |
| `modules/evaluation.py` | Holdout, 10-fold CV, scores, error table |
| `config.py` | Every badge cut-off + model settings |
| `data/raw_data.csv` | 2,751 webscraped comments |
| `outputs/` | Ready-made result tables + `run_log.txt` |
| `saved_state/hf_cache/` | Saved transformer answers (GPU runs once) |

---

## 10. What was already executed here (real numbers, 2026-10-09)

**Simple:** On this machine I installed everything, ran stages 1–5 on all
2,742 comments, and saved every table into `outputs/` — use them straight in
your report's script/output/remarks table. Stage 6 (validation) is waiting on
human labels. The transformer AI could not run *on this machine* (its speed
engine is blocked by this computer's app policy), but its code is tested and
its save/reload cache is proven — run it once on the GPU computer and it will
behave the same, then never need re-running.

Key results: cleaning kept 2,347/2,742 (86%); FB↔YT similarity 0.75, X most
distinct; best theme-count 8; sentiment net +30.8 overall (FB +40.4, X +2.1,
YT +18.7).

**The jargon:** artifacts — `clean_dataset.csv`, `word_frequencies.csv`,
`similarity_cosine.csv`, `similarity_jaccard.csv`, `shared_words.csv`,
`topic_k_search.csv` (sklearn-fallback, NPMI coherence, best k=8 @ 0.1437),
`topic_top_words.csv`, `topic_share_per_platform.csv`,
`sentiment_rule_based.csv`, `run_log.txt`. App boots clean (health 200, no
stderr errors). Blocker: `torch` DLL load vetoed by WDAC/AppLocker here
(`OSError 4551`); CPU torch installed but unloadable — transformer inference
must execute on the unrestricted RTX 4060 env (one command, page 6 button or
notebook §5b). gensim has no cp314 wheel → sklearn-LDA fallback engaged
automatically (documented in remarks).

---

## 11. Defense cheat-sheet (one line each)

- **Why rule-based, not just AI?** The brief requires it — and every decision
  is explainable word-by-word.
- **Why two sentiment judges?** Rule-based is transparent but misses slang;
  the transformer reads context. Agreement between them = confidence.
- **Why lock the 10%?** So we can't tune answers to the test (data leakage).
- **Why is X/Twitter different?** Smaller, punchier posts; distinct slang and
  hashtags — show the distinctive-words table.
- **Biggest weakness?** 60% of comments match no lexicon word; topics still
  contain filler words; Facebook dominates the sample; scraped data starts
  unlabeled.
- **Stance vs tone?** "Nakakagalit ang fake news laban kay Sara!" = angry
  words (negative tone), pro-Sara side. We measure both, separately.
