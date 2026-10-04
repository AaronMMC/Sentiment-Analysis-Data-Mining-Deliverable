"""
similarity.py — TF-IDF, cosine similarity, and Jaccard similarity.

What each answers (for the defense):
  - TF-IDF: turns each GROUP of comments (e.g. all YouTube texts joined
    together) into a weighted word vector. Rare-but-distinctive words get
    higher weight than words appearing everywhere ("the", "ang").
  - Cosine similarity: angle between two TF-IDF vectors, 0 (nothing alike)
    to 1 (identical direction). Robust to different group sizes.
  - Jaccard similarity: |words in BOTH| / |words in EITHER| on plain sets
    of top words — a stricter, easy-to-explain overlap measure.

All functions take the REAL cleaned dataframe and return REAL matrices;
badges/interpretations only format those numbers using config thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _group_texts(df: pd.DataFrame, group_col: str,
                 text_col: str = "clean_text") -> tuple:
    """Join all cleaned texts per group into one document per group.

    Returns (group_names, documents). Raises a helpful error if the group
    column is missing or has < 2 groups (similarity needs at least a pair).
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in the data.")
    if text_col not in df.columns:
        raise ValueError("Run preprocessing first (need 'clean_text').")
    grouped = df.groupby(group_col)[text_col].apply(
        lambda s: " ".join(s.astype(str)))
    grouped = grouped[grouped.str.strip() != ""]
    if len(grouped) < 2:
        raise ValueError(f"Need at least 2 non-empty groups in '{group_col}' "
                         f"to compare — found {len(grouped)}.")
    return list(grouped.index), list(grouped.values)


def tfidf_cosine_matrix(df: pd.DataFrame, group_col: str,
                        text_col: str = "clean_text",
                        max_features: int = 2000) -> tuple:
    """TF-IDF vectors per group + cosine similarity matrix.

    Returns (groups, vectorizer, tfidf_matrix, cosine_df) where cosine_df
    is a labeled DataFrame (groups x groups) with values in [0, 1].
    """
    groups, docs = _group_texts(df, group_col, text_col)
    vec = TfidfVectorizer(max_features=max_features)
    mat = vec.fit_transform(docs)
    cos = cosine_similarity(mat)
    cos_df = pd.DataFrame(cos, index=groups, columns=groups).round(4)
    return groups, vec, mat, cos_df


def jaccard_matrix(df: pd.DataFrame, group_col: str,
                   text_col: str = "clean_text",
                   top_n: int = 200) -> pd.DataFrame:
    """Jaccard similarity matrix over each group's top-N word sets.

    For each group we take its top_n most frequent tokens as the group's
    "vocabulary set", then J(A,B) = |A∩B| / |A∪B| for every pair.
    """
    groups, docs = _group_texts(df, group_col, text_col)
    vocab_sets = []
    for d in docs:
        toks = d.split()
        freq = pd.Series(toks).value_counts().head(top_n).index.tolist()
        vocab_sets.append(set(freq))
    n = len(groups)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            union = vocab_sets[i] | vocab_sets[j]
            mat[i, j] = (len(vocab_sets[i] & vocab_sets[j]) / len(union)
                         if union else 0.0)
    return pd.DataFrame(mat, index=groups, columns=groups).round(4)


def top_shared_and_distinctive(df: pd.DataFrame, group_col: str,
                               text_col: str = "clean_text",
                               top_n: int = 15) -> tuple:
    """Find top shared words (in EVERY group) and distinctive words per group.

    - Shared: words present in all groups, ranked by total frequency.
    - Distinctive: per group, words frequent in that group but rare/ absent
      elsewhere (simple ratio score, transparent for a defense).

    Returns (shared_df, {group: distinctive_df}).
    """
    groups, docs = _group_texts(df, group_col, text_col)
    counts = {}
    for g, d in zip(groups, docs):
        counts[g] = pd.Series(d.split()).value_counts()
    all_words = set().union(*[set(c.index) for c in counts.values()])
    shared_rows = []
    for w in all_words:
        if all(w in counts[g] for g in groups):
            shared_rows.append((w, sum(counts[g][w] for g in groups)))
    shared_df = pd.DataFrame(
        sorted(shared_rows, key=lambda x: -x[1])[:top_n],
        columns=["word", "total_count"])
    distinctive = {}
    for g in groups:
        rows = []
        others = [o for o in groups if o != g]
        for w, c in counts[g].head(200).items():
            other_max = max([counts[o].get(w, 0) for o in others] + [0])
            # Distinctiveness: high own-count, low elsewhere (+1 smoothing).
            score = c / (other_max + 1)
            rows.append((w, c, other_max, round(score, 3)))
        distinctive[g] = pd.DataFrame(
            sorted(rows, key=lambda x: -x[3])[:top_n],
            columns=["word", f"count_in_{g}", "max_count_elsewhere",
                     "distinctiveness"])
    return shared_df, distinctive


# --- Plain-language interpreters (thresholds passed in from config) ------
def interpret_cosine(value: float, th: dict) -> tuple:
    """Map a cosine value to (level, sentence) using config thresholds."""
    if value < th["cosine_low_max"]:
        return ("low", f"{value:.2f} is LOW — the two groups use quite "
                       "different language.")
    if value <= th["cosine_mod_max"]:
        return ("moderate", f"{value:.2f} is MODERATE — some shared "
                            "framing, but each group has its own vocabulary.")
    return ("high", f"{value:.2f} is HIGH — the two groups talk about the "
                    "issue in very similar terms.")


def interpret_jaccard(value: float, th: dict) -> tuple:
    """Map a Jaccard value to (level, sentence) using config thresholds."""
    if value < th["jaccard_low_max"]:
        return ("low", f"{value:.2f} is LOW — very little word overlap.")
    if value <= th["jaccard_mod_max"]:
        return ("moderate", f"{value:.2f} is MODERATE — noticeable overlap "
                            "with room for group-specific language.")
    return ("high", f"{value:.2f} is HIGH — strong word overlap between "
                    "the groups.")


def similarity_helper_bullets(cos_df: pd.DataFrame, jac_df: pd.DataFrame,
                              th: dict) -> list:
    """Build 3-5 plain-English observations from REAL similarity matrices."""
    bullets = []
    # Work only on off-diagonal pairs (group vs. different group).
    pairs = [(a, b, cos_df.loc[a, b]) for a in cos_df.index
             for b in cos_df.columns if a < b]
    if not pairs:
        return ["Not enough groups to compare yet."]
    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    lo = pairs_sorted[0]
    hi = pairs_sorted[-1]
    lvl_lo, _ = interpret_cosine(lo[2], th)
    lvl_hi, _ = interpret_cosine(hi[2], th)
    bullets.append(
        f"Most different pair: **{lo[0]} vs {lo[1]}** (cosine {lo[2]:.2f}, "
        f"{lvl_lo}). If this is low, the two camps frame the issue differently.")
    bullets.append(
        f"Most similar pair: **{hi[0]} vs {hi[1]}** (cosine {hi[2]:.2f}, "
        f"{lvl_hi}). A high value here can mean shared facts or shared talking points.")
    avg_cos = float(cos_df.where(~(cos_df == 1.0)).stack().mean())
    bullets.append(
        f"Average cross-group cosine is **{avg_cos:.2f}**. Low averages (< 0.3) "
        f"suggest echo chambers; high averages (> 0.6) suggest one shared debate.")
    # Jaccard counterpart for the most-different cosine pair.
    try:
        jv = float(jac_df.loc[lo[0], lo[1]])
        lvl_j, _ = interpret_jaccard(jv, th)
        bullets.append(
            f"For that most-different pair, Jaccard word overlap is **{jv:.2f}** "
            f"({lvl_j}) — stricter than cosine because it ignores frequencies.")
    except KeyError:
        pass
    bullets.append(
        "Next step: check the distinctive-terms tables below. If group-specific "
        "words are mostly slogans or insults, mention that as a discourse finding.")
    return bullets[:5]
