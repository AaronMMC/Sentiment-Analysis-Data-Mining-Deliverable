"""
helpers.py — Small shared utilities used by every page of the app.

Contents (each explained for oral-defense purposes):
  1. badge() / status helpers  -> turn a number + thresholds into a
     color-coded HTML badge (green/yellow/red) with a plain-language line.
  2. what_to_do_box()          -> the "What you need to do here" instruction
     box shown at the top of every page.
  3. analysis_helper_box()      -> renders the "Analysis Helper" bullet list.
  4. save_state()/load_state()  -> persist st.session_state to saved_state/
     as JSON + CSV so work survives a browser refresh.
  5. tooltip()                 -> hover (?) hint next to technical terms.
  6. stage helpers             -> decide which wizard pages are unlocked.

Nothing here fabricates data: every message takes the REAL computed numbers
as input and only formats/interprets them.
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

SAVE_DIR = "saved_state"


# ---------------------------------------------------------------------------
# 1. Badges
# ---------------------------------------------------------------------------
def badge(color: str, label: str, detail: str = "") -> str:
    """Return a small HTML badge string.

    Args:
        color: one of "green", "yellow", "red", "grey".
        label: short status text, e.g. "GREEN - on track".
        detail: one-line plain-language interpretation.

    Students: this is PURE presentation logic. The thresholds that decide
    the color live in config.py and in the modules that compute metrics.
    """
    colors = {
        "green": ("#1a7f37", "#dafbe1"),
        "yellow": ("#9a6700", "#fff8c5"),
        "red": ("#cf222e", "#ffebe9"),
        "grey": ("#57606a", "#eaeef2"),
    }
    fg, bg = colors.get(color, colors["grey"])
    html = (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
        f"font-weight:700;font-size:0.85em;color:{fg};background:{bg};"
        f"border:1px solid {fg};'>{label}</span>"
    )
    if detail:
        html += f" <span style='font-size:0.9em;'>{detail}</span>"
    return html


def render_badge(color: str, label: str, detail: str = "") -> None:
    """Convenience wrapper that writes a badge with st.markdown."""
    st.markdown(badge(color, label, detail), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Instruction box + analysis helper box
# ---------------------------------------------------------------------------
def what_to_do_box(title: str, steps: list, good_result: str, mistakes: list) -> None:
    """Render the standard 'What you need to do here' box for a page."""
    with st.container(border=True):
        st.markdown(f"### 🧭 What you need to do here: {title}")
        st.markdown("**Follow these steps in order:**")
        for i, s in enumerate(steps, 1):
            st.markdown(f"{i}. {s}")
        st.success(f"**What a good result looks like:** {good_result}")
        if mistakes:
            with st.expander("⚠️ Common mistakes to avoid"):
                for m in mistakes:
                    st.markdown(f"- {m}")


def analysis_helper_box(title: str, bullets: list) -> None:
    """Render the 'Analysis Helper' panel: 3-5 plain-English observations.

    The caller (each results page) COMPUTES the bullets from real numbers;
    this function only displays them. Never pass invented bullets.
    """
    with st.container(border=True):
        st.markdown(f"### 💡 Analysis Helper — {title}")
        st.caption(
            "These notes are generated automatically from YOUR numbers above. "
            "Thresholds are rule-of-thumb guides, not strict rules — results "
            "depend on dataset size and difficulty."
        )
        if not bullets:
            st.info("Not enough data yet to write observations.")
            return
        for b in bullets:
            st.markdown(f"- {b}")


# ---------------------------------------------------------------------------
# 3. Tooltips (hover ? next to technical terms)
# ---------------------------------------------------------------------------
def tooltip(term: str, definition: str) -> str:
    """Return an HTML hover tooltip:  Term (?)  with definition on hover."""
    safe = definition.replace('"', "&quot;")
    return (
        f"{term} "
        f"<span title=\"{safe}\" style='cursor:help;color:#0969da;"
        f"font-weight:700;'>(?)</span>"
    )


def tip(term: str, definition: str) -> None:
    """Write a tooltip inline via markdown (use inside st.markdown)."""
    st.markdown(tooltip(term, definition), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Save / load project state
# ---------------------------------------------------------------------------
def _state_to_jsonable(state: dict) -> dict:
    """Convert session values to JSON-safe types (skip DataFrames)."""
    # Recomputable / non-serializable live objects: never persist (a str()
    # snapshot would restore as a useless string and crash `.items()`).
    SKIP_KEYS = {"distinctive", "topic_reps", "lda_model"}
    out = {}
    for k, v in state.items():
        if k.startswith("_") or k in SKIP_KEYS:
            continue
        if isinstance(v, pd.DataFrame):
            continue  # DataFrames saved separately as CSV
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def save_state() -> str:
    """Persist key session_state items to saved_state/ folder.

    Saves: settings/thresholds/lexicon/labels as JSON, plus each DataFrame
    (df_raw, df_clean, ...) as CSV. Returns the timestamped folder message.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # JSON part
    snap = _state_to_jsonable(dict(st.session_state))
    snap["_saved_at"] = ts
    with open(os.path.join(SAVE_DIR, "state.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    # CSV parts (datasets + computed result artifacts so reopen is instant).
    # cos/jac/share matrices are symmetric group x group: keep the row-label
    # index (plain to_csv would replace it with 0,1,2 on reload and break
    # the pair-badge loop).
    INDEXED_KEYS = {"cos_df", "jac_df", "topic_share"}
    for key in ["df_raw", "df_clean", "sentiment_df", "hf_df", "error_df",
                "cos_df", "jac_df", "shared_df", "k_table", "topic_words",
                "topic_share"]:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            p = os.path.join(SAVE_DIR, f"{key}.csv")
            if key in INDEXED_KEYS:
                df.to_csv(p, index_label="group")
            else:
                df.to_csv(p, index=False)
    # Lexicon / stopwords explicitly (may be DataFrame or list)
    lex = st.session_state.get("lexicon_df")
    if isinstance(lex, pd.DataFrame) and not lex.empty:
        lex.to_csv(os.path.join(SAVE_DIR, "lexicon.csv"), index=False)
    return f"Project saved to '{SAVE_DIR}/' at {ts}."


def load_state() -> str:
    """Reload a previously saved project state. Returns status message."""
    state_path = os.path.join(SAVE_DIR, "state.json")
    if not os.path.exists(state_path):
        return "No saved project found yet."
    with open(state_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    for k, v in snap.items():
        if k == "_saved_at":
            continue
        st.session_state[k] = v
    for key in ["df_raw", "df_clean", "sentiment_df", "hf_df", "error_df",
                "cos_df", "jac_df", "shared_df", "k_table", "topic_words",
                "topic_share"]:
        p = os.path.join(SAVE_DIR, f"{key}.csv")
        if os.path.exists(p):
            try:
                if key in ("cos_df", "jac_df", "topic_share"):
                    st.session_state[key] = pd.read_csv(p, index_col=0)
                else:
                    st.session_state[key] = pd.read_csv(p)
            except Exception:
                pass
    # Repair: list-columns (tokens) come back from CSV as strings.
    for key in ["df_clean", "sentiment_df"]:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty and \
                "tokens" in df.columns and "clean_text" in df.columns:
            try:
                if not isinstance(df["tokens"].iloc[0], list):
                    df["tokens"] = df["clean_text"].astype(str).str.split()
                    st.session_state[key] = df
            except Exception:
                pass
    # Repair: all-empty label columns come back from CSV as float64, which
    # makes pandas 3 refuse string assignment (LossySetitemError) in the
    # labeling tool. Force them back to nullable-string semantics.
    for key in ["df_raw", "df_clean", "sentiment_df", "hf_df"]:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                if "manual_sentiment" in df.columns:
                    df["manual_sentiment"] = df["manual_sentiment"] \
                        .fillna("").astype("string")
                if "political_bloc" in df.columns:
                    df["political_bloc"] = df["political_bloc"] \
                        .fillna("Undecided/Neutral").astype("string")
                st.session_state[key] = df
            except Exception:
                pass
    # Repair: JSON turns topic label keys into strings; re-key to int, and
    # coerce restored assignments to (int, float) tuples.
    try:
        tl = st.session_state.get("topic_labels")
        if isinstance(tl, dict):
            st.session_state["topic_labels"] = {
                int(k): v for k, v in tl.items()}
    except Exception:
        pass
    try:
        ta = st.session_state.get("topic_assign")
        if isinstance(ta, list):
            st.session_state["topic_assign"] = [
                (int(t), float(p)) for t, p in ta]
    except Exception:
        pass
    # Restore stage flags implied by present artifacts.
    if isinstance(st.session_state.get("df_raw"), pd.DataFrame):
        st.session_state["stage_collection_done"] = True
    if isinstance(st.session_state.get("df_clean"), pd.DataFrame):
        st.session_state["stage_preprocessing_done"] = True
    if isinstance(st.session_state.get("cos_df"), pd.DataFrame):
        st.session_state["stage_similarity_done"] = True
    if isinstance(st.session_state.get("topic_words"), pd.DataFrame):
        st.session_state["stage_topics_done"] = True
    if isinstance(st.session_state.get("sentiment_df"), pd.DataFrame):
        st.session_state["stage_sentiment_done"] = True
    lex_p = os.path.join(SAVE_DIR, "lexicon.csv")
    if os.path.exists(lex_p):
        try:
            st.session_state["lexicon_df"] = pd.read_csv(lex_p)
        except Exception:
            pass
    return "Saved project reloaded from disk."


# ---------------------------------------------------------------------------
# 5. Wizard stage logic
# ---------------------------------------------------------------------------
# Ordered page ids. Later pages lock until the previous stage flag is set.
STAGE_ORDER = [
    "start",
    "collection",
    "preprocessing",
    "similarity",
    "topics",
    "sentiment",
    "validation",
    "glossary",
]

# session_state flag that unlocks the NEXT page after each stage:
STAGE_FLAGS = {
    "collection": "stage_collection_done",       # df_raw with >= 5 rows
    "preprocessing": "stage_preprocessing_done",  # df_clean exists
    "similarity": "stage_similarity_done",
    "topics": "stage_topics_done",
    "sentiment": "stage_sentiment_done",
    "validation": "stage_validation_done",
}


def is_unlocked(page: str) -> bool:
    """Return True if the given page may be viewed.

    Rules: start/collection/glossary always open; any other page needs ALL
    previous stage flags to be True.
    """
    if page in ("start", "collection", "glossary"):
        return True
    idx = STAGE_ORDER.index(page)
    for earlier in STAGE_ORDER[1:idx]:
        flag = STAGE_FLAGS.get(earlier)
        if flag and not st.session_state.get(flag, False):
            return False
    return True


def locked_message(page: str) -> str:
    """Human-readable message explaining which step must be finished first."""
    idx = STAGE_ORDER.index(page)
    for earlier in STAGE_ORDER[1:idx]:
        flag = STAGE_FLAGS.get(earlier)
        if flag and not st.session_state.get(flag, False):
            pretty = earlier.replace("_", " ").title()
            return (
                f"🔒 This page is locked until you finish **{pretty}**. "
                f"Go back to that page and complete the steps there first."
            )
    return "🔒 This page is locked until previous steps are complete."
