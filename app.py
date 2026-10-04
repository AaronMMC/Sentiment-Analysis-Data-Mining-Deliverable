"""
app.py — Streamlit guided-wizard GUI for the impeachment discourse project.

Run with:  streamlit run app.py   (Python 3.10+, tested on 3.14.0)

Wizard pages (sidebar radio + progress tracker):
  1. Start Here / Checklist   2. Data Collection   3. Preprocessing (Taglish)
  4. Similarity Analysis      5. Topic Modeling     6. Sentiment Classifier
  7. Validation & Evaluation  8. Glossary & Notes (always open)

Rules enforced here (not in the library modules):
  - Later pages LOCK until previous stage flags are set (see helpers.py).
  - Every number shown is computed from the user's real data; badges and
    Analysis Helper bullets only FORMAT/INTERPRET those numbers.
  - Demo data is clearly bannered and never silently mixed with real data.

State: everything lives in st.session_state + saved_state/ via save/load.
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# Make "modules/" importable whether launched from repo root or elsewhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_THRESHOLDS, THRESHOLD_HELP  # noqa: E402
from modules import data_collection as DC  # noqa: E402
from modules import preprocessing as PP  # noqa: E402
from modules import similarity as SIM  # noqa: E402
from modules import topics as TOP  # noqa: E402
from modules import sentiment as SEN  # noqa: E402
from modules import evaluation as EV  # noqa: E402
from modules import glossary as GLO  # noqa: E402
from modules.helpers import (analysis_helper_box, is_unlocked, load_state,
                             locked_message, render_badge, save_state,
                             tooltip, what_to_do_box)

st.set_page_config(page_title="Impeachment Discourse Analyzer",
                   page_icon="🗳️", layout="wide")

PAGES = [
    ("start", "1 · Start Here / Checklist"),
    ("collection", "2 · Data Collection"),
    ("preprocessing", "3 · Preprocessing (Taglish)"),
    ("similarity", "4 · Similarity Analysis"),
    ("topics", "5 · Topic Modeling"),
    ("sentiment", "6 · Sentiment Classifier"),
    ("validation", "7 · Validation & Evaluation"),
    ("glossary", "8 · Glossary & Notes"),
]


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
def init_state():
    """Set every session key the app reads, exactly once per session."""
    defaults = {
        "df_raw": None,            # collected/labeled dataset (schema columns)
        "df_clean": None,          # preprocessed dataset (+clean_text/tokens)
        "freq_table": None,        # word frequencies from preprocessing
        "prep_log": None,          # step-count funnel from preprocessing
        "demo_mode": False,        # True while DEMO DATA is loaded
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "lexicon_df": SEN.lexicon_to_df(dict(SEN.DEFAULT_LEXICON)),
        "pos_thresh": 0.15,
        "neg_thresh": -0.15,
        "label_idx": 0,            # labeling-tool cursor
        "annotator2": {},          # {row_idx: label} second annotator
        "holdout_idx": None,       # locked holdout row ids (leakage guard)
        "dev_df": None, "hold_df": None,
        "cv_df": None, "final_res": None,
        "topic_labels": {},        # {topic_id: human label}
        "topic_assign": None,      # dominant topic per doc
        "k_table": None, "coherence_best": None,
        "my_notes": "",
        "stage_collection_done": False,
        "stage_preprocessing_done": False,
        "stage_similarity_done": False,
        "stage_topics_done": False,
        "stage_sentiment_done": False,
        "stage_validation_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
TH = st.session_state["thresholds"]  # shortcut, always the LIVE edited dict


# ---------------------------------------------------------------------------
# Small display helpers local to the GUI
# ---------------------------------------------------------------------------
def badge_for_records(n: int) -> tuple:
    """(color, label, detail) for the records-collected benchmark."""
    if n < TH["records_red_max"]:
        return ("red", "RED — below target",
                f"{n} records; target ≥ {TH['records_red_max']}. Keep collecting/labeling.")
    if n <= TH["records_yellow_max"]:
        return ("yellow", "YELLOW — minimum met",
                f"{n} records; ≥ {TH['records_yellow_max'] + 1} is comfortably green.")
    return ("green", "GREEN — solid size",
            f"{n} records; above {TH['records_yellow_max']}.")


def badge_for_balance(df: pd.DataFrame):
    """Warn if any sentiment class < min_class_share (needs gold labels)."""
    lab = df[df["manual_sentiment"].isin(EV.LABELS)] if df is not None else None
    if lab is None or lab.empty:
        return ("grey", "GREY — no labels yet", "Label comments to check balance.")
    shares = lab["manual_sentiment"].value_counts(normalize=True)
    worst = shares.idxmin()
    if shares.min() < TH["min_class_share"]:
        return ("red", "RED — imbalanced",
                f"'{worst}' is only {shares.min():.0%} (< {TH['min_class_share']:.0%}).")
    return ("green", "GREEN — balanced",
            f"Smallest class '{worst}' = {shares.min():.0%}.")


def badge_for_metric(val: float, good_detail: str = "") -> tuple:
    """Generic accuracy/macro-F1 badge."""
    if val < TH["acc_red_max"]:
        return ("red", f"RED — {val:.2f}", f"Below {TH['acc_red_max']}. {good_detail}")
    if val <= TH["acc_yellow_max"]:
        return ("yellow", f"YELLOW — {val:.2f}",
                f"Between {TH['acc_red_max']}–{TH['acc_yellow_max']}. {good_detail}")
    return ("green", f"GREEN — {val:.2f}", f"Above {TH['acc_yellow_max']}. {good_detail}")


def demo_banner():
    """Warn whenever demo rows are loaded (honesty requirement)."""
    if st.session_state.get("demo_mode"):
        st.warning("⚠️ **DEMO DATA loaded** — built-in sample for testing only. "
                   "Replace it with your real collected comments before defending. "
                   "All charts below reflect the demo, not real discourse.")


def df_raw():
    """Return the working raw dataframe (empty-safe)."""
    d = st.session_state.get("df_raw")
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


def add_rows(new_df: pd.DataFrame, mark_demo=False):
    """Append new rows to df_raw, de-duplicating on text."""
    base = df_raw()
    new_df = new_df.copy()
    combined = pd.concat([base, new_df], ignore_index=True) \
        if not base.empty else new_df
    # Drop exact-duplicate comment texts (keep first occurrence).
    combined["__k"] = combined["text"].astype(str).str.strip().str.lower()
    combined = combined.drop_duplicates(subset=["__k"]).drop(
        columns=["__k"]).reset_index(drop=True)
    st.session_state["df_raw"] = combined
    st.session_state["demo_mode"] = bool(mark_demo)
    if len(combined) >= 5:
        st.session_state["stage_collection_done"] = True


# ---------------------------------------------------------------------------
# Sidebar: navigation, progress, thresholds, glossary search, save/load
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🗳️ Impeachment Discourse")
    st.caption("Guided text-mining wizard · VP Sara Duterte case")
    labels = [l for _, l in PAGES]
    ids = [i for i, _ in PAGES]
    current_label = st.radio("Go to page", labels, index=0)
    page = ids[labels.index(current_label)]

    st.divider()
    st.subheader("📈 Progress tracker")
    n_raw = len(df_raw())
    n_clean = len(st.session_state["df_clean"]) \
        if isinstance(st.session_state.get("df_clean"), pd.DataFrame) else 0
    checks = [
        ("Records (target 200+)", n_raw >= 200, f"{n_raw}"),
        ("Preprocessing done", bool(st.session_state["stage_preprocessing_done"]),
         f"{n_clean} clean"),
        ("3+ topics modeled", st.session_state.get("k_table") is not None
         and (st.session_state["k_table"]["k"].max() >= 3
              if not st.session_state["k_table"].empty else False), ""),
        ("10% holdout locked", st.session_state.get("hold_df") is not None,
         f"{len(st.session_state['hold_df'])}" if st.session_state.get("hold_df") is not None else ""),
        ("10-fold CV done", st.session_state.get("cv_df") is not None, ""),
        ("Final metrics computed", st.session_state.get("final_res") is not None, ""),
    ]
    for name, done, extra in checks:
        st.markdown(f"{'✅' if done else '⬜'} {name}" + (f" — {extra}" if extra else ""))
    st.progress(min(1.0, n_raw / 200.0),
                text=f"{n_raw} / 200 records toward target")

    st.divider()
    with st.expander("🎛️ Thresholds (rule-of-thumb guides, editable)"):
        st.caption("Change any cut-off; badges update everywhere instantly.")
        for key in ["records_red_max", "records_yellow_max", "min_class_share",
                    "kappa_red_max", "kappa_yellow_max", "coherence_red_max",
                    "coherence_yellow_max", "cosine_low_max", "cosine_mod_max",
                    "jaccard_low_max", "jaccard_mod_max", "acc_red_max",
                    "acc_yellow_max", "per_class_min", "cv_sd_flag",
                    "overfit_gap_flag", "neutral_zone"]:
            cur = float(TH[key])
            step = 0.05 if cur < 5 else 1.0
            TH[key] = st.number_input(key, value=cur, step=step,
                                     help=THRESHOLD_HELP.get(key, ""))
        st.caption("random_state=42 · holdout=10% · 10-fold CV (fixed for reproducibility).")

    st.divider()
    st.subheader("🔎 Glossary search")
    gq = st.text_input("Search a term", placeholder="e.g. recall, kappa…",
                       label_visibility="collapsed")
    if gq:
        hits = GLO.search_glossary(gq)
        if hits:
            for term in list(hits)[:5]:
                st.markdown(f"**{term}**: {hits[term]['definition']}")
        else:
            st.info("No glossary match.")
    st.caption("Full entries + formulas live on page 8.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save project"):
            st.success(save_state())
    with c2:
        if st.button("📂 Load saved"):
            st.success(load_state())
            st.rerun()

demo_banner()

# Gate locked pages.
if not is_unlocked(page):
    st.error(locked_message(page))
    st.info("Finish the earlier steps first — the wizard unlocks pages in order "
            "so methods stay valid (e.g., no topics before cleaning).")
    st.stop()


# ===========================================================================
# PAGE 1 — START HERE / CHECKLIST
# ===========================================================================
if page == "start":
    what_to_do_box(
        "Start Here", [
            "Read the workflow overview below so the 7 stages make sense.",
            "Go to **page 2** and load DEMO DATA (or collect real data) — later pages unlock after that.",
            "Watch the sidebar **Progress tracker** tick forward as you finish stages.",
            "Open **Thresholds** in the sidebar to see the rule-of-thumb cut-offs behind every badge.",
        ],
        "all six tracker boxes ticked: 200+ records, clean data, 3+ topics, locked holdout, CV done, final metrics.",
        ["Skipping collection and jumping to results — pages lock to stop this.",
         "Treating demo outputs as findings — the banner always warns when demo data is active.",
         "Chasing green badges by re-tuning on the holdout (that's leakage — page 7 locks the split)."])
    st.header("1 · Start Here — What this project does")
    st.markdown("""
This app walks you through a **complete text-mining project** on public discourse
about the impeachment issue of **VP Sara Duterte**, in plain language:

| Stage | Page | What you do | What you get |
|---|---|---|---|
| Collection | 2 | YouTube API · Reddit API · file upload · manual paste + labeling | Labeled dataset (target **200+** comments) |
| Preprocessing | 3 | Taglish cleaning, stopwords, duplicates | Before/after table + word frequencies |
| Similarity | 4 | TF-IDF · cosine · Jaccard between platforms/blocs | Heatmaps + shared/distinctive words |
| Topics | 5 | LDA (optional LSA), k-search 2–10 with Coherence | Named topics + shares per platform |
| Sentiment | 6 | VADER + custom Taglish lexicon, negation, sliders | Net sentiment per topic/platform |
| Validation | 7 | Locked 10% holdout → 10-fold CV → final eval | Precision/Recall/F1, confusion matrix, error table |
| Glossary | 8 | Definitions, formulas, your notes | Defense-ready explanations |

**Stance vs tone (read now, remember always):** `political_bloc` records which SIDE a
comment supports; sentiment records how its WORDING sounds. An angry defense of VP Sara is
*Negative tone* but *Pro-Duterte stance* — the error table on page 7 exists to catch these.
""")
    st.subheader("Requirement tracker (live, from your data)")
    col1, col2, col3 = st.columns(3)
    with col1:
        c, l, d = badge_for_records(n_raw)
        render_badge(c, f"Records: {l}", d)
    with col2:
        c, l, d = badge_for_balance(df_raw())
        render_badge(c, f"Class balance: {l}", d)
    with col3:
        kt = st.session_state.get("k_table")
        if kt is not None and not kt.empty and kt["k"].max() >= 3:
            render_badge("green", "Topics: GREEN — 3+ topics",
                         f"Best k={int(kt.loc[kt['coherence_cv'].idxmax(), 'k'])}.")
        else:
            render_badge("grey", "Topics: GREY — not yet",
                         "Model topics on page 5 (need 3+).")
    col4, col5, col6 = st.columns(3)
    with col4:
        render_badge(*(("green", "GREEN — holdout locked",
                        f"{len(st.session_state['hold_df'])} rows.")
                       if st.session_state.get("hold_df") is not None
                       else ("grey", "GREY — holdout pending",
                             "Create the 10% split on page 7.")))
    with col5:
        render_badge(*(("green", "GREEN — CV done", "")
                       if st.session_state.get("cv_df") is not None
                       else ("grey", "GREY — CV pending",
                             "Run 10-fold CV on page 7.")))
    with col6:
        fr = st.session_state.get("final_res")
        if fr is not None:
            c, l, d = badge_for_metric(fr["macro_f1"], "Macro-F1 on locked holdout.")
            render_badge(c, f"Final macro-F1: {l}", d)
        else:
            render_badge("grey", "GREY — metrics pending",
                         "Final one-time eval on page 7.")
    st.info("💡 Tip: click **page 2** in the sidebar and press **Load DEMO DATA** "
            "to try every page end-to-end in minutes, then replace it with real data.")


# ===========================================================================
# PAGE 2 — DATA COLLECTION
# ===========================================================================
elif page == "collection":
    what_to_do_box(
        "Data Collection", [
            "Collect comments via **A** YouTube API, **B** Reddit API, **C** CSV/Excel upload, or **D** manual paste (one comment per line). Facebook/X/news = import-only (see Why box).",
            "Press **Load DEMO DATA** first if you just want to test the pipeline.",
            "Use the **labeling tool** below: pick sentiment + bloc per comment, click Save & Next.",
            "Watch counts-per-platform and the 200-record progress bar; **Export CSV** when done.",
        ],
        "200+ rows, every row labeled (no blanks in manual_sentiment), counts spread across ≥2 platforms.",
        ["Uploading a file with no text column and assuming it worked — check the error message.",
         "Labeling TONE as STANCE (angry defense of Sara = Negative tone, Pro-Duterte bloc).",
         "Writing Facebook/X scrapers that need logins — use import/manual entry instead (ToS)."])
    st.header("2 · Data Collection")
    with st.expander("🤔 Why are Facebook and X import-only? (read for your defense)"):
        st.markdown("""
Facebook and X **prohibit automated scraping** without permission and require logins;
bypassing that violates their Terms of Service and risks account bans. The honest,
reproducible route is: use each platform's **export/share feature**, copy comments
**manually**, and paste them into upload (C) or the paste box (D) with the source noted.
YouTube and Reddit instead offer **official APIs** (Data API v3, PRAW) designed for
this — so the app automates only those two, respecting quotas and rate limits.
""")

    tabA, tabB, tabC, tabD = st.tabs(
        ["(a) YouTube API", "(b) Reddit API (PRAW)", "(c) Upload CSV/Excel",
         "(d) Manual paste (FB/X/news)"])

    with tabA:
        st.markdown("**YouTube comments** via the official Data API "
                    + tooltip("quota", "Google limits free API calls per day; the app fetches top-level comments only."))
        yt_url = st.text_input("Video URL or 11-char ID",
                               placeholder="https://www.youtube.com/watch?v=…")
        yt_key = st.text_input("YouTube API key", type="password",
                               help="Google Cloud Console → enable YouTube Data API v3 → API key. Stored only in this session.")
        yt_n = st.slider("Max comments", 10, 500, 100)
        if st.button("Fetch YouTube comments"):
            try:
                vid = DC.extract_youtube_video_id(yt_url)
                new = DC.fetch_youtube_comments(vid, yt_key, yt_n)
                new["source"] = f"YouTube:{vid}"
                add_rows(new)
                st.success(f"Added {len(new)} YouTube comments.")
            except Exception as e:
                st.error(f"Could not fetch: {e}")

    with tabB:
        st.markdown("**Reddit comments** via PRAW (official API, read-only).")
        r_sub = st.text_input("Subreddit (without r/)", value="Philippines")
        r_kw = st.text_input("Keyword", value="Sara Duterte impeachment")
        r_id = st.text_input("Reddit client_id", type="password")
        r_sec = st.text_input("Reddit client_secret", type="password")
        r_agent = st.text_input("User agent", value="impeachment-student-research/1.0")
        r_lim = st.slider("Max comments", 10, 300, 60)
        if st.button("Fetch Reddit comments"):
            try:
                new = DC.fetch_reddit_comments(r_sub, r_kw, r_id, r_sec,
                                               r_agent, r_lim)
                add_rows(new)
                st.success(f"Added {len(new)} Reddit comments.")
            except Exception as e:
                st.error(f"Could not fetch: {e}")

    with tabC:
        st.markdown("Upload a **CSV/Excel** export (columns auto-mapped to "
                    "id/platform/source/date/text/bloc/sentiment; usernames stripped).")
        up = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
        if up is not None:
            try:
                df_up = pd.read_csv(up) if up.name.lower().endswith(".csv") \
                    else pd.read_excel(up)
                st.dataframe(df_up.head(), use_container_width=True)
                if st.button("Import uploaded file"):
                    new = DC.coerce_schema(df_up)
                    add_rows(new)
                    st.success(f"Imported {len(new)} rows (usernames removed).")
                    st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

    with tabD:
        st.markdown("Paste comments from **Facebook / X / news sites** — one per line. "
                    "Usernames like @Maria are auto-anonymized to @user.")
        p_plat = st.selectbox("Platform label",
                              ["Facebook (imported)", "X (imported)",
                               "News site (imported)", "Other"])
        p_src = st.text_input("Source note", placeholder="e.g. FB post by GMA News, 2025-02-10")
        p_txt = st.text_area("Paste comments (one per line)", height=150)
        if st.button("Add pasted comments"):
            try:
                new = DC.parse_manual_paste(p_txt, p_plat, p_src)
                add_rows(new)
                st.success(f"Added {len(new)} pasted comments.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    colx, coly = st.columns([1, 1])
    with colx:
        if st.button("🧪 Load DEMO DATA (30 clearly-marked sample rows)"):
            try:
                add_rows(DC.load_demo_dataframe(), mark_demo=True)
                st.session_state["label_idx"] = 0
                st.success("Demo data loaded — banner is showing; replace before defending.")
                st.rerun()
            except Exception as e:
                st.error(f"Demo load failed: {e}")
    with coly:
        if st.button("🗑️ Clear dataset (start over with real data)"):
            for k in ["df_raw", "df_clean", "freq_table", "prep_log", "dev_df",
                      "hold_df", "cv_df", "final_res", "topic_assign", "k_table"]:
                st.session_state[k] = None
            for k in ["stage_collection_done", "stage_preprocessing_done",
                      "stage_similarity_done", "stage_topics_done",
                      "stage_sentiment_done", "stage_validation_done"]:
                st.session_state[k] = False
            st.session_state["demo_mode"] = False
            st.session_state["holdout_idx"] = None
            st.rerun()

    st.subheader("Live dataset")
    data = df_raw()
    if data.empty:
        st.info("No data yet — fetch, upload, paste, or load DEMO DATA above.")
    else:
        counts = data["platform"].value_counts()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Counts per platform**")
            st.dataframe(counts.rename("n"), use_container_width=True)
            labeled = int(data["manual_sentiment"].isin(EV.LABELS).sum())
            st.markdown(f"**Labeled:** {labeled} / {len(data)}")
            c, l, d = badge_for_records(len(data))
            render_badge(c, l, d)
            c, l, d = badge_for_balance(data)
            render_badge(c, l, d)
            st.progress(min(1.0, len(data) / 200.0),
                        text=f"{len(data)} / 200 toward target")
            st.download_button("⬇️ Export dataset (.csv)",
                               data.to_csv(index=False).encode("utf-8"),
                               "impeachment_dataset.csv", "text/csv")
        with c2:
            st.markdown("**All rows (edit labels inline, or use the tool below)**")
            edited = st.data_editor(
                data, use_container_width=True, num_rows="dynamic",
                column_config={
                    "manual_sentiment": st.column_config.SelectboxColumn(
                        "manual_sentiment", options=[""] + EV.LABELS),
                    "political_bloc": st.column_config.SelectboxColumn(
                        "political_bloc", options=DC.BLOCS)},
                key="raw_editor")
            if st.button("Apply table edits"):
                try:
                    st.session_state["df_raw"] = DC.coerce_schema(
                        edited, default_source="edited")
                    st.session_state["demo_mode"] = False \
                        if (edited["source"].str.contains(
                            "DEMO", na=False).sum() == 0) else \
                        st.session_state["demo_mode"]
                    st.success("Edits applied.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not apply edits: {e}")

        # ---- Labeling tool: one comment at a time ----
        st.subheader("🏷️ Labeling tool (one comment at a time)")
        with st.expander("📖 Labeling guide + stance-vs-tone caution (read first)",
                         expanded=False):
            st.markdown("""
**Sentiment = TONE of the wording.** Positive = praise/support/relief wording.
Negative = anger/condemnation/fraud wording. Neutral = factual/unsure/balanced.

**Bloc = STANCE (side).** Pro-Duterte / Pro-Accountability / Undecided-Neutral.

⚠️ **Caution:** tone ≠ stance. *"Nakakagalit ang fake news laban kay Sara!"*
is Negative TONE (galit) but Pro-Duterte STANCE. Label both columns independently.
When unsure, choose Neutral + Undecided and note it in page 8.
""")
        unlabeled = data[~data["manual_sentiment"].isin(EV.LABELS)].index.tolist()
        st.markdown(f"Unlabeled remaining: **{len(unlabeled)}**")
        idx = st.number_input("Row number to label", 0, max(0, len(data) - 1),
                              int(st.session_state.get("label_idx", 0) or 0))
        st.session_state["label_idx"] = int(idx)
        if not data.empty:
            row = data.iloc[int(idx)]
            st.info(f"**[{row['platform']}]** {row['text']}")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("👍 Positive"):
                    data.at[data.index[int(idx)], "manual_sentiment"] = "Positive"
                    st.session_state["df_raw"] = data
                    st.session_state["label_idx"] = min(int(idx) + 1, len(data) - 1)
                    st.rerun()
            with b2:
                if st.button("👎 Negative"):
                    data.at[data.index[int(idx)], "manual_sentiment"] = "Negative"
                    st.session_state["df_raw"] = data
                    st.session_state["label_idx"] = min(int(idx) + 1, len(data) - 1)
                    st.rerun()
            with b3:
                if st.button("😐 Neutral"):
                    data.at[data.index[int(idx)], "manual_sentiment"] = "Neutral"
                    st.session_state["df_raw"] = data
                    st.session_state["label_idx"] = min(int(idx) + 1, len(data) - 1)
                    st.rerun()
            new_bloc = st.selectbox("Political bloc", DC.BLOCS,
                                    index=DC.BLOCS.index(row["political_bloc"])
                                    if row["political_bloc"] in DC.BLOCS else 2)
            if new_bloc != row["political_bloc"]:
                data.at[data.index[int(idx)], "political_bloc"] = new_bloc
                st.session_state["df_raw"] = data
                st.rerun()

        # ---- Optional second annotator + kappa ----
        with st.expander("👥 Optional second-annotator mode (Cohen's kappa)"):
            st.markdown("A classmate labels the SAME rows (their labels stored separately); "
                        "kappa is computed on rows labeled by both.")
            a2_idx = st.number_input("Row for annotator 2", 0,
                                     max(0, len(data) - 1), 0, key="a2_idx")
            a2_lab = st.selectbox("Annotator-2 label", EV.LABELS, key="a2_lab")
            if st.button("Save annotator-2 label"):
                st.session_state["annotator2"][int(a2_idx)] = a2_lab
                st.success("Saved.")
            a2 = st.session_state.get("annotator2", {})
            if a2:
                rows = sorted(set(a2) & set(
                    data[data["manual_sentiment"].isin(EV.LABELS)].index.tolist()))
                if len(rows) >= 5:
                    try:
                        k = DC.cohens_kappa(
                            [data.loc[r, "manual_sentiment"] for r in rows],
                            [a2[r] for r in rows])
                        if k < TH["kappa_red_max"]:
                            render_badge("red", f"RED — κ={k:.2f}",
                                         "Poor agreement; re-align on the guide.")
                        elif k <= TH["kappa_yellow_max"]:
                            render_badge("yellow", f"YELLOW — κ={k:.2f}",
                                         "Fair; discuss disagreements.")
                        else:
                            render_badge("green", f"GREEN — κ={k:.2f}",
                                         "Good agreement.")
                        st.markdown(f"Computed on {len(rows)} doubly-labeled rows.")
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.info(f"Only {len(rows)} doubly-labeled rows — need ≥5.")


# ===========================================================================
# PAGE 3 — PREPROCESSING
# ===========================================================================
elif page == "preprocessing":
    what_to_do_box(
        "Preprocessing (Taglish)", [
            "Tick cleaning options (emoji mapping, extra stopwords, minimum length).",
            "Click **Run preprocessing** — read the before/after table row by row.",
            "Check the funnel (records at each step) and the word-frequency chart.",
            "If everything vanished, loosen stopwords / lower minimum tokens and re-run.",
        ],
        "a clean table with non-empty clean_text + tokens, duplicates removed, top words are content words (not 'ang/the/sa').",
        ["Deleting all data with too many stopwords or too high a minimum length.",
         "Forgetting that hashtag WORDS are kept ('#DueProcess' → 'dueprocess').",
         "Editing raw text after cleaning — always edit raw, then re-run this page."])
    st.header("3 · Preprocessing (Taglish pipeline)")
    st.markdown("Pipeline: lowercase → URLs stripped → emojis mapped/stripped → "
                "@mentions removed → " + tooltip("hashtag words kept",
                "The # symbol is dropped but the word stays, e.g. #DueProcess becomes dueprocess.")
                + " → repeats collapsed → slang normalized → "
                + tooltip("tokenization", "Splitting text into countable word units.")
                + " → stopwords removed → duplicates + short rows dropped.")
    data = df_raw()
    if data.empty:
        st.error("No data — collect or load data on page 2 first.")
        st.stop()
    map_emo = st.checkbox("Map emojis to sentiment tokens (else strip them)",
                          value=True,
                          help="💪→'laban', 😡→'galit'. Keeps emotional signal for the lexicon.")
    min_tok = st.slider("Drop rows with fewer than N tokens", 1, 10, 3)
    st.markdown("**Custom stopwords** (comma-separated, added to English + Filipino lists):")
    custom_sw = st.text_area("custom stopwords", value="",
                             placeholder="e.g. sara, duterte, impeachment",
                             label_visibility="collapsed")
    custom_list = [w.strip().lower() for w in custom_sw.split(",") if w.strip()]
    with st.expander("See default Filipino stopwords + slang map"):
        st.markdown(f"Filipino stopwords ({len(PP.FILIPINO_STOPWORDS)}): "
                    f"{', '.join(sorted(PP.FILIPINO_STOPWORDS))}")
        st.markdown("Slang sample: " + ", ".join(
            f"{k}→{v}" for k, v in list(PP.TAGLISH_SLANG.items())[:20]) + " …")
    if st.button("🧹 Run preprocessing"):
        try:
            clean, log, freq = PP.preprocess_dataframe(
                data, custom_stopwords=custom_list, map_emojis=map_emo,
                min_tokens=min_tok)
            st.session_state["df_clean"] = clean
            st.session_state["freq_table"] = freq
            st.session_state["prep_log"] = log
            st.session_state["stage_preprocessing_done"] = True
            st.success(f"Kept {len(clean)} of {len(data)} rows.")
            st.rerun()
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()
    if st.session_state.get("df_clean") is None:
        st.info("Click **Run preprocessing** to generate results.")
        st.stop()
    clean = st.session_state["df_clean"]
    st.subheader("Before / after comparison")
    st.dataframe(clean[["text", "clean_text", "n_tokens"]].head(20),
                 use_container_width=True)
    st.subheader("Records at each step")
    st.dataframe(pd.DataFrame(st.session_state["prep_log"],
                              columns=["step", "rows"]),
                 use_container_width=True, hide_index=True)
    st.subheader("Word-frequency chart (top 25, from YOUR cleaned data)")
    try:
        import plotly.express as px
        fig = px.bar(st.session_state["freq_table"].head(25),
                     x="word", y="count", title="Top words after cleaning")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart failed: {e}")
        st.dataframe(st.session_state["freq_table"].head(25))
    analysis_helper_box("Preprocessing", [
        f"Kept **{len(clean)}** of **{len(data)}** input rows "
        f"({len(clean)/max(1,len(data)):.0%} retention). Heavy drops usually mean "
        f"many duplicates or ultra-short comments.",
        ("Top words look like content words (issue terms) — good."
         if not set(st.session_state["freq_table"]["word"].head(5)) &
         {"ang", "ng", "sa", "the", "at"}
         else "Top words still include stopwords — add them to the custom box and re-run."),
        "If one word dominates (>3× the next), consider whether it is a slogan "
        "flooding every topic; stopword-ing it can sharpen topics."])


# ===========================================================================
# PAGE 4 — SIMILARITY
# ===========================================================================
elif page == "similarity":
    what_to_do_box(
        "Similarity Analysis", [
            "Pick a grouping (platform / political bloc / source).",
            "Click **Compute similarities**.",
            "Read the cosine heatmap first (overall framing), then Jaccard (strict overlap).",
            "Open shared vs distinctive words and quote 2–3 in your defense.",
        ],
        "two labeled heatmaps, a shared-words table, and per-group distinctive words that match what you read in the comments.",
        ["Comparing groups with <2 non-empty sides — add data or pick another grouping.",
         "Calling 0.15 cosine 'high' — check the badge cut-offs (low <0.3, high >0.6).",
         "Reporting numbers without reading example comments."])
    st.header("4 · Similarity Analysis (TF-IDF · cosine · Jaccard)")
    st.markdown("Each group becomes one TF-IDF vector " +
                tooltip("(weighted word profile)",
                        "Words frequent in one group but rare overall get high weight.") +
                "; cosine = angle between vectors, Jaccard = strict word-set overlap.")
    clean = st.session_state.get("df_clean")
    if clean is None or clean.empty:
        st.error("Run preprocessing (page 3) first.")
        st.stop()
    group_col = st.selectbox("Group comments by",
                             ["platform", "political_bloc", "source"])
    maxfeat = st.slider("TF-IDF vocabulary size", 500, 5000, 2000)
    if st.button("🔍 Compute similarities"):
        try:
            _, _, _, cos_df = SIM.tfidf_cosine_matrix(clean, group_col,
                                                      max_features=maxfeat)
            jac_df = SIM.jaccard_matrix(clean, group_col)
            shared, distinctive = SIM.top_shared_and_distinctive(clean, group_col)
            st.session_state["cos_df"] = cos_df
            st.session_state["jac_df"] = jac_df
            st.session_state["shared_df"] = shared
            st.session_state["distinctive"] = distinctive
            st.session_state["sim_group"] = group_col
            st.session_state["stage_similarity_done"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Similarity failed: {e}\n\n{traceback.format_exc(limit=3)}")
            st.stop()
    if st.session_state.get("cos_df") is None:
        st.info("Click **Compute similarities** to generate results.")
        st.stop()
    cos_df, jac_df = st.session_state["cos_df"], st.session_state["jac_df"]
    try:
        import plotly.express as px
        st.subheader("Cosine similarity heatmap (0=different → 1=identical)")
        st.plotly_chart(px.imshow(cos_df.values, x=list(cos_df.columns),
                                  y=list(cos_df.index), text_auto=".2f",
                                  zmin=0, zmax=1,
                                  title="Cosine between groups",
                                  color_continuous_scale="Blues"),
                        use_container_width=True)
        st.subheader("Jaccard similarity heatmap (strict word overlap)")
        st.plotly_chart(px.imshow(jac_df.values, x=list(jac_df.columns),
                                  y=list(jac_df.index), text_auto=".2f",
                                  zmin=0, zmax=1,
                                  title="Jaccard between groups",
                                  color_continuous_scale="Greens"),
                        use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap failed: {e}")
        st.dataframe(cos_df); st.dataframe(jac_df)
    st.subheader("Pair badges (thresholds from sidebar)")
    for a in cos_df.index:
        for b in cos_df.columns:
            if a < b:
                lvl, sent = SIM.interpret_cosine(float(cos_df.loc[a, b]), TH)
                color = {"low": "red", "moderate": "yellow",
                         "high": "green"}[lvl]
                render_badge(color, f"{a} vs {b}: cosine {cos_df.loc[a, b]:.2f}",
                             sent)
    st.subheader("Top shared words (in EVERY group)")
    st.dataframe(st.session_state["shared_df"], use_container_width=True)
    st.subheader("Top distinctive words per group")
    for g, dframe in st.session_state["distinctive"].items():
        with st.expander(f"Distinctive: {g}"):
            st.dataframe(dframe, use_container_width=True)
    analysis_helper_box("Similarity",
                        SIM.similarity_helper_bullets(cos_df, jac_df, TH))


# ===========================================================================
# PAGE 5 — TOPIC MODELING
# ===========================================================================
elif page == "topics":
    what_to_do_box(
        "Topic Modeling", [
            "Choose LDA (default) or LSA; set topics with the slider for a first look.",
            "Click **Auto-search k = 2..10** — the app plots Coherence + Perplexity and highlights the best k.",
            "Read top words + representative documents, then type your own label per topic.",
            "Check topic share per platform before moving on.",
        ],
        "a best-k with green/yellow coherence, topics you can name confidently from examples, no two topics identical.",
        ["Picking k by gut feel without running the search.",
         "Naming topics from numbers alone without reading example comments.",
         "Trusting Perplexity over Coherence when they disagree."])
    st.header("5 · Topic Modeling (LDA · optional LSA)")
    clean = st.session_state.get("df_clean")
    if clean is None or clean.empty:
        st.error("Run preprocessing (page 3) first.")
        st.stop()
    method = st.radio("Method", ["LDA (gensim, recommended)", "LSA (fast, linear-algebra)"],
                      horizontal=True)
    k = st.slider("Number of topics (manual run)", 2, 10, 4)
    col1, col2 = st.columns(2)
    with col1:
        run_one = st.button("🧠 Train topics (k above)")
    with col2:
        run_search = st.button("🔎 Auto-search k = 2..10 (Coherence + Perplexity)")
    try:
        token_lists = TOP._get_tokens(clean)
    except Exception as e:
        st.error(str(e))
        st.stop()
    if run_search:
        with st.spinner("Training LDA for k=2..10 (this can take a minute)…"):
            try:
                kt = TOP.search_k(token_lists, 2, min(10, len(token_lists)))
                st.session_state["k_table"] = kt
                best = kt.loc[kt["coherence_cv"].idxmax()]
                st.session_state["coherence_best"] = float(best["coherence_cv"])
                k = int(best["k"])
                st.success(f"Best k = {k} (Coherence {best['coherence_cv']:.3f}). "
                           f"Training final model…")
            except Exception as e:
                st.error(f"k-search failed: {e}")
                st.stop()
    if run_one or run_search:
        try:
            if method.startswith("LDA"):
                kk = int(st.session_state["k_table"].loc[
                    st.session_state["k_table"]["coherence_cv"].idxmax(), "k"]) \
                    if (run_search and st.session_state.get("k_table") is not None) else k
                model, dictionary, corpus = TOP.train_lda(token_lists, num_topics=kk)
                mets = TOP.lda_metrics(model, token_lists, dictionary, corpus)
                tw = TOP.top_words_per_topic(model)
                assign = TOP.assign_dominant_topics(model, corpus)
                reps = TOP.representative_docs(clean, assign)
                share = TOP.topic_share_per_platform(clean, assign)
                st.session_state.update(
                    lda_model=model, topic_assign=assign, topic_words=tw,
                    topic_reps=reps, topic_share=share, topic_k=kk,
                    topic_method="LDA", coherence_best=float(mets["coherence_cv"]),
                    topic_perplexity=float(mets["perplexity"])
                    if np.isfinite(mets["perplexity"]) else None)
            else:
                tw, doc_topic = TOP.train_lsa(token_lists, num_topics=k)
                assign = [(int(np.argmax(r)), float(np.max(r)))
                          if r.max() > 0 else (0, 0.0) for r in doc_topic]
                reps = TOP.representative_docs(clean, assign)
                share = TOP.topic_share_per_platform(clean, assign)
                st.session_state.update(
                    topic_assign=assign, topic_words=tw, topic_reps=reps,
                    topic_share=share, topic_k=int(tw["topic_id"].max() + 1),
                    topic_method="LSA", coherence_best=None,
                    topic_perplexity=None)
            st.session_state["stage_topics_done"] = True
            st.rerun()
        except Exception as e:
            st.error(f"Topic training failed: {e}\n\n{traceback.format_exc(limit=3)}")
            st.stop()
    kt = st.session_state.get("k_table")
    if kt is not None and not kt.empty:
        st.subheader("k-search results (from YOUR data)")
        try:
            import plotly.express as px
            st.plotly_chart(px.line(kt, x="k", y="coherence_cv",
                                    markers=True, title="Coherence Cv per k (higher = better)"),
                            use_container_width=True)
            if kt["perplexity"].notna().any():
                st.plotly_chart(px.line(kt, x="k", y="perplexity",
                                        markers=True, title="Perplexity per k (lower = better fit)"),
                                use_container_width=True)
        except Exception:
            pass
        st.dataframe(kt.style.highlight_max(subset=["coherence_cv"], color="#dafbe1"),
                     use_container_width=True)
        best = kt.loc[kt["coherence_cv"].idxmax()]
        cv = float(best["coherence_cv"])
        if cv < TH["coherence_red_max"]:
            render_badge("red", f"RED — Cv {cv:.3f}", "Topics may be hard to interpret.")
        elif cv <= TH["coherence_yellow_max"]:
            render_badge("yellow", f"YELLOW — Cv {cv:.3f}", "Usable; verify with example docs.")
        else:
            render_badge("green", f"GREEN — Cv {cv:.3f}", "Topics hang together well.")
        analysis_helper_box("Topic k-search", TOP.topic_helper_bullets(kt, TH))
    if st.session_state.get("topic_words") is None:
        st.info("Train topics (or run the k-search) to see results.")
        if TOP.bertopic_available():
            st.caption("BERTopic is installed — mention it as optional future work; "
                       "LDA is enough for this course.")
        st.stop()
    st.subheader(f"Top words per topic ({st.session_state['topic_method']}, "
                 f"k={st.session_state['topic_k']})")
    st.dataframe(st.session_state["topic_words"], use_container_width=True)
    st.subheader("✏️ Name each topic (your interpretation — saved with the project)")
    for tid in sorted(st.session_state["topic_words"]["topic_id"].unique()):
        words = ", ".join(st.session_state["topic_words"][
            st.session_state["topic_words"]["topic_id"] == tid
        ].head(5)["word"].tolist())
        st.session_state["topic_labels"][int(tid)] = st.text_input(
            f"Topic {tid} [{words}]",
            value=st.session_state["topic_labels"].get(int(tid), f"Topic {tid}"))
    st.subheader("Representative documents (highest topic probability)")
    for tid, rdf in st.session_state["topic_reps"].items():
        label = st.session_state["topic_labels"].get(int(tid), f"Topic {tid}")
        with st.expander(f"Topic {tid} — {label}"):
            st.dataframe(rdf, use_container_width=True)
    st.subheader("Topic share per platform (%)")
    try:
        import plotly.express as px
        share = st.session_state["topic_share"]
        st.dataframe(share, use_container_width=True)
        st.plotly_chart(px.imshow(share.values, x=list(share.columns),
                                  y=list(share.index), text_auto=".1f",
                                  title="Topic share per platform (%)"),
                        use_container_width=True)
    except Exception as e:
        st.error(f"Share chart failed: {e}")


# ===========================================================================
# PAGE 6 — SENTIMENT CLASSIFIER
# ===========================================================================
elif page == "sentiment":
    what_to_do_box(
        "Sentiment Classifier", [
            "Review/edit the lexicon table (word → score in −2..+2); add Tagalog words your data actually uses.",
            "Set Positive/Negative threshold sliders.",
            "Click **Classify all comments**; read distribution + net-sentiment bars.",
            "If a side looks wrong, fix the LEXICON (not the results) and re-run.",
        ],
        "a predicted column for every row, net-sentiment bars per platform/bloc, and lexicon words you can defend one by one.",
        ["Deleting all negative words to force green badges — panels check this.",
         "Setting thresholds to 0.0/0.0 so nothing is neutral.",
         "Confusing stance with tone (see page-2 caution)."])
    st.header("6 · Sentiment Classifier (VADER + custom Taglish lexicon)")
    st.markdown("Final score = **0.4 × VADER** (English baseline " +
                tooltip("compound −1..+1",
                        "VADER's overall sentiment signal including punctuation/caps/emoji emphasis.") +
                ") + **0.6 × lexicon mean** (your Taglish table), with "
                + tooltip("negation flips", "hindi/not/never reverses the next sentiment word within 3 tokens.")
                + " and intensifiers (×1.5).")
    clean = st.session_state.get("df_clean")
    if clean is None or clean.empty:
        st.error("Run preprocessing (page 3) first.")
        st.stop()
    st.subheader("Editable lexicon (word → score −2..+2)")
    st.caption("Double-click a cell to edit. Positive = praise/support; "
               "negative = anger/fraud/condemnation wording.")
    lex_edited = st.data_editor(st.session_state["lexicon_df"],
                                num_rows="dynamic", use_container_width=True,
                                key="lex_editor")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["pos_thresh"] = st.slider("Positive threshold (≥ → Positive)",
                                                   0.0, 1.0, float(st.session_state["pos_thresh"]), 0.01)
    with c2:
        st.session_state["neg_thresh"] = st.slider("Negative threshold (≤ → Negative)",
                                                   -1.0, 0.0, float(st.session_state["neg_thresh"]), 0.01)
    if st.session_state["pos_thresh"] <= st.session_state["neg_thresh"]:
        st.error("Positive threshold must be ABOVE the negative threshold.")
        st.stop()
    if st.button("💬 Classify all comments"):
        try:
            lex = SEN.df_to_lexicon(lex_edited)
            st.session_state["lexicon_df"] = lex_edited
            st.session_state["lexicon"] = lex
            sdf = SEN.classify_dataframe(clean, lex,
                                         pos_thresh=st.session_state["pos_thresh"],
                                         neg_thresh=st.session_state["neg_thresh"])
            # Attach topic names if topics were modeled.
            if st.session_state.get("topic_assign") is not None and \
                    len(st.session_state["topic_assign"]) == len(sdf):
                sdf["topic"] = [
                    st.session_state["topic_labels"].get(t, f"T{t}")
                    for t, _ in st.session_state["topic_assign"]]
            st.session_state["sentiment_df"] = sdf
            st.session_state["df_clean"] = sdf  # keep predictions downstream
            st.session_state["stage_sentiment_done"] = True
            st.success(f"Classified {len(sdf)} comments.")
            st.rerun()
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.stop()
    sdf = st.session_state.get("sentiment_df")
    if sdf is None:
        st.info("Click **Classify all comments** to generate results.")
        st.stop()
    st.subheader("Prediction distribution (your data)")
    dist = sdf["predicted"].value_counts().reindex(["Positive", "Negative",
                                                    "Neutral"]).fillna(0)
    try:
        import plotly.express as px
        st.plotly_chart(px.bar(x=dist.index, y=dist.values,
                               labels={"x": "predicted", "y": "count"},
                               title="Predicted sentiment counts"),
                        use_container_width=True)
    except Exception:
        st.dataframe(dist)
    overall = SEN.net_sentiment(sdf["predicted"])
    nz = TH["neutral_zone"]
    render_badge("grey" if abs(overall["net"]) <= nz else
                 ("green" if overall["net"] > 0 else "red"),
                 f"Net sentiment {overall['net']:+.1f} "
                 f"({overall['pos_pct']:.0f}%+ / {overall['neg_pct']:.0f}%− / "
                 f"{overall['neu_pct']:.0f}%○)",
                 "Inside ±{:.0f} = neutral zone.".format(nz))
    for grp in ["platform", "political_bloc"] + (["topic"] if "topic" in sdf else []):
        st.subheader(f"Net sentiment per {grp} (−100..+100)")
        nb = SEN.net_by_group(sdf, grp)
        st.dataframe(nb, use_container_width=True)
        try:
            import plotly.express as px
            fig = px.bar(nb, x="group", y="net", color="net",
                         color_continuous_scale="RdYlGn", range_y=[-100, 100],
                         title=f"Net sentiment by {grp}")
            fig.add_hrect(y0=-nz, y1=nz, fillcolor="grey", opacity=0.15,
                          annotation_text="neutral zone")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart failed: {e}")
    with st.expander("Show scored rows (audit the explanations)"):
        show_cols = [c for c in ["text", "predicted", "final_score",
                                 "vader_compound", "lexicon_sum",
                                 "lexicon_hits", "score_explanation",
                                 "manual_sentiment"]
                     if c in sdf.columns]
        st.dataframe(sdf[show_cols].head(50), use_container_width=True)
    analysis_helper_box("Sentiment", SEN.sentiment_helper_bullets(sdf, TH))


# ===========================================================================
# PAGE 7 — VALIDATION & EVALUATION
# ===========================================================================
elif page == "validation":
    what_to_do_box(
        "Validation & Evaluation", [
            "STEP 1: click **Create 10% holdout** ONCE (stratified, seed 42) — it locks immediately.",
            "STEP 2: tune on the 90% dev set with **10-fold CV** (try different thresholds/lexicon edits on page 6).",
            "STEP 3: click **Final evaluation on holdout** exactly ONCE and report it.",
            "Read per-class rows, the confusion matrix, stability/overfit badges, and the error table.",
        ],
        "locked holdout + CV mean±SD + one holdout report with honest weak spots and an error table you can discuss.",
        ["Re-splitting the holdout after peeking (leakage — the lock exists to stop this).",
         "Tuning thresholds ON the holdout instead of the dev folds.",
         "Reporting accuracy alone on imbalanced classes."])
    st.header("7 · Validation & Evaluation (no leakage)")
    sdf = st.session_state.get("sentiment_df")
    if sdf is None or sdf.empty:
        st.error("Run the sentiment classifier (page 6) first.")
        st.stop()
    st.markdown("Protocol: **holdout BEFORE tuning → CV on dev → final eval ONCE.** "
                "Thresholds/sliders may only be tuned on dev folds.")
    # ---- STEP 1: holdout ----
    st.subheader("Step 1 · One-click stratified 10% holdout (seed 42, locked)")
    if st.session_state.get("hold_df") is None:
        if st.button("🔒 Create 10% holdout split"):
            try:
                lab = EV.labeled_only(sdf)
                dev, hold = EV.make_holdout(
                    lab, holdout_frac=float(TH["holdout_frac"]),
                    random_state=int(TH.get("random_state", 42)))
                st.session_state["dev_df"] = dev
                st.session_state["hold_df"] = hold
                st.session_state["holdout_idx"] = hold["id"].tolist()
                st.success(f"Locked: dev {len(dev)} rows, holdout {len(hold)} rows. "
                           f"It will NOT be re-split (reset button below warns about leakage).")
                st.rerun()
            except Exception as e:
                st.error(f"Holdout failed: {e}")
                st.stop()
    else:
        hold, dev = st.session_state["hold_df"], st.session_state["dev_df"]
        render_badge("green", "GREEN — holdout locked",
                     f"dev {len(dev)} rows · holdout {len(hold)} rows · seed "
                     f"{TH.get('random_state', 42)}.")
        st.dataframe(hold["manual_sentiment"].value_counts().rename("holdout_n"),
                     use_container_width=False)
        if st.button("⚠️ Reset holdout (DANGER: re-splitting after peeking = leakage)"):
            st.session_state.update(dev_df=None, hold_df=None,
                                    holdout_idx=None, cv_df=None, final_res=None)
            st.warning("Holdout cleared. Only do this BEFORE any tuning/eval.")
            st.rerun()
    if st.session_state.get("hold_df") is None:
        st.info("Create the holdout to unlock steps 2–3.")
        st.stop()
    # ---- STEP 2: CV on dev ----
    st.subheader("Step 2 · 10-fold stratified CV on the dev 90%")
    st.caption("CV re-classifies each dev fold with the CURRENT lexicon + sliders "
               "(go back to page 6 to change them, then re-run CV here).")
    if st.button("🔁 Run 10-fold CV"):
        try:
            lex = SEN.df_to_lexicon(st.session_state["lexicon_df"])
            pt, nt = float(st.session_state["pos_thresh"]), float(
                st.session_state["neg_thresh"])

            def classify_fn(fold_df):
                out = SEN.classify_dataframe(fold_df, lex, pos_thresh=pt,
                                             neg_thresh=nt)
                return out["predicted"].tolist()

            cv = EV.cross_validate(st.session_state["dev_df"], classify_fn,
                                   n_splits=int(TH.get("cv_folds", 10)),
                                   random_state=int(TH.get("random_state", 42)))
            st.session_state["cv_df"] = cv
            if cv.attrs.get("note"):
                st.warning(cv.attrs["note"])
            st.success("CV finished.")
            st.rerun()
        except Exception as e:
            st.error(f"CV failed: {e}\n\n{traceback.format_exc(limit=3)}")
            st.stop()
    if st.session_state.get("cv_df") is not None:
        cv = st.session_state["cv_df"]
        st.dataframe(cv, use_container_width=True)
        summ = EV.cv_summary(cv)
        st.markdown("**Mean ± SD across folds:**")
        st.dataframe(summ, use_container_width=True, hide_index=True)
        try:
            sd = float(summ.loc[summ["metric"] == "macro_f1", "sd"].iloc[0])
            if sd > TH["cv_sd_flag"]:
                render_badge("red", f"RED — unstable (SD {sd:.3f})",
                             f"SD > {TH['cv_sd_flag']}: results swing with the split.")
            else:
                render_badge("green", f"GREEN — stable (SD {sd:.3f})",
                             "Tuning should generalize.")
        except Exception:
            pass
    else:
        st.info("Run CV before the final evaluation.")
    # ---- STEP 3: final eval ----
    st.subheader("Step 3 · Final one-time evaluation on the locked holdout")
    if st.button("✅ Final evaluation on holdout (one-time)"):
        try:
            hold = st.session_state["hold_df"]
            res = EV.final_metrics(hold["manual_sentiment"].tolist(),
                                   hold["predicted"].tolist())
            err = EV.error_table(hold)
            st.session_state["final_res"] = res
            st.session_state["error_df"] = err
            st.session_state["stage_validation_done"] = True
            st.success("Final evaluation computed — report these numbers as-is.")
            st.rerun()
        except Exception as e:
            st.error(f"Final eval failed: {e}")
            st.stop()
    fr = st.session_state.get("final_res")
    if fr is None:
        st.info("Click **Final evaluation** to generate the report.")
        st.stop()
    m1, m2, m3 = st.columns(3)
    with m1:
        c, l, d = badge_for_metric(fr["accuracy"], "Holdout accuracy.")
        render_badge(c, f"Accuracy: {l}", d)
    with m2:
        c, l, d = badge_for_metric(fr["macro_f1"], "Holdout macro-F1.")
        render_badge(c, f"Macro-F1: {l}", d)
    with m3:
        render_badge("grey", f"Weighted-F1 {fr['weighted_f1']:.2f}",
                     "Accounts for class sizes.")
    st.markdown("**Per-class Precision / Recall / F1:**")
    st.dataframe(fr["report_df"], use_container_width=True)
    weak = [lab for lab, v in fr["per_class"].items()
            if v["precision"] < TH["per_class_min"]
            or v["recall"] < TH["per_class_min"]]
    if weak:
        render_badge("red", f"RED — weak classes: {', '.join(weak)}",
                     f"Precision or recall < {TH['per_class_min']}. Discuss why.")
    else:
        render_badge("green", "GREEN — all classes ≥ 0.50",
                     "No class falls below the per-class floor.")
    st.subheader("Confusion matrix (rows = gold, columns = predicted)")
    try:
        import plotly.express as px
        st.plotly_chart(px.imshow(fr["confusion_matrix"], x=EV.LABELS,
                                  y=EV.LABELS, text_auto=True,
                                  labels={"x": "predicted", "y": "gold"},
                                  title="Holdout confusion matrix",
                                  color_continuous_scale="Blues"),
                        use_container_width=True)
    except Exception as e:
        st.error(f"Matrix chart failed: {e}")
        st.dataframe(pd.DataFrame(fr["confusion_matrix"], index=EV.LABELS,
                                  columns=EV.LABELS))
    # Overfit badge.
    if st.session_state.get("cv_df") is not None:
        summ = EV.cv_summary(st.session_state["cv_df"])
        try:
            gap = float(summ.loc[summ["metric"] == "macro_f1", "mean"].iloc[0]) \
                - fr["macro_f1"]
            if gap > TH["overfit_gap_flag"]:
                render_badge("red", f"RED — possible overfit (gap {gap:+.2f})",
                             f"CV − holdout > {TH['overfit_gap_flag']}.")
            else:
                render_badge("green", f"GREEN — no overfit (gap {gap:+.2f})", "")
        except Exception:
            pass
    coh = st.session_state.get("coherence_best")
    if coh is not None:
        if coh < TH["coherence_red_max"]:
            render_badge("red", f"Topic Coherence Cv {coh:.3f} — RED", "")
        elif coh <= TH["coherence_yellow_max"]:
            render_badge("yellow", f"Topic Coherence Cv {coh:.3f} — YELLOW", "")
        else:
            render_badge("green", f"Topic Coherence Cv {coh:.3f} — GREEN", "")
    st.subheader("Error analysis (misclassified holdout comments)")
    err = st.session_state.get("error_df")
    if err is None or err.empty:
        st.success("No misclassifications on the holdout — verify the holdout "
                   "isn't trivially small before celebrating.")
    else:
        st.markdown(f"**{len(err)}** of **{len(st.session_state['hold_df'])}** "
                    f"holdout comments misclassified. Reasons are heuristic guesses — "
                    f"read each row yourself.")
        st.dataframe(err, use_container_width=True)
    analysis_helper_box("Evaluation", EV.evaluation_helper_bullets(
        fr, EV.cv_summary(st.session_state["cv_df"])
        if st.session_state.get("cv_df") is not None else pd.DataFrame(),
        st.session_state.get("cv_df"), len(st.session_state["hold_df"]),
        coh, TH))


# ===========================================================================
# PAGE 8 — GLOSSARY & NOTES (always unlocked)
# ===========================================================================
elif page == "glossary":
    st.header("8 · Glossary & My Notes")
    st.caption("Hover (?) tooltips across the app come from these entries. "
               "If you can explain each 'how to read it' line, you can handle "
               "most defense questions.")
    q = st.text_input("Filter terms", placeholder="type to filter…")
    for term, entry in GLO.search_glossary(q).items():
        with st.expander(f"📖 {term}"):
            st.markdown(f"**Plain definition:** {entry['definition']}")
            st.markdown(f"**Example:** {entry['example']}")
            if entry.get("formula") and entry["formula"] != "—":
                st.code(entry["formula"])
            st.markdown(f"**How to read it in this project:** {entry['how_to_read']}")
    st.divider()
    st.subheader("📝 My Notes (saved to disk)")
    notes = st.text_area("Personal notes (methods, defense answers, to-do)",
                         value=st.session_state.get("my_notes", ""),
                         height=200, key="notes_box")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save notes"):
            st.session_state["my_notes"] = notes
            os.makedirs("saved_state", exist_ok=True)
            with open("saved_state/my_notes.txt", "w",
                      encoding="utf-8") as f:
                f.write(notes)
            st.success("Notes saved to saved_state/my_notes.txt")
    with col2:
        if os.path.exists("saved_state/my_notes.txt") and st.button("Reload notes"):
            with open("saved_state/my_notes.txt", encoding="utf-8") as f:
                st.session_state["my_notes"] = f.read()
            st.rerun()
