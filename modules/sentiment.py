"""
sentiment.py — Rule-based sentiment: VADER baseline + custom Taglish
political lexicon with negation handling and intensifiers.

How classification works (explainable in a defense):
  1. VADER scores the raw text -> compound in [-1, +1] (English-tuned;
     weak on Tagalog, which is WHY we add a custom lexicon).
  2. Custom lexicon scores each cleaned token: word scores in [-2, +2]
     from the editable GUI table (students own this list).
  3. Negation: a negation word ("hindi", "huwag", "not", "never", ...)
     flips the NEXT sentiment-bearing word within 3 tokens.
  4. Intensifiers ("sobra", "masyado", "very", "grabe", ...) multiply the
     next sentiment word by 1.5.
  5. Combined score = 0.4 * vader_compound*2 scaled + 0.6 * lexicon_sum
     normalized... (see combine_scores). Threshold sliders then map the
     final score to Positive / Neutral / Negative.

Net sentiment per group = positive% − negative%, range −100..+100.

IMPORTANT stance-vs-tone caution (also shown in the labeling guide):
  A comment can have an ANGRY tone but a PRO-Duterte stance
  ("Nakakagalit ang fake news laban kay Sara!"). Our classifier measures
  TONE ( wording ), while political_bloc measures STANCE (side). The
  evaluation page's error table helps students spot these mismatches.
"""

import re

import pandas as pd

# --- Default custom lexicon: word -> score in [-2, +2] ---------------------
# Positive = praise/support/relief; negative = anger/fraud/condemnation.
# Students EDIT this table in the GUI; every change re-runs classification.
DEFAULT_LEXICON = {
    # pro-support / positive tone
    "mabuhay": 1.5, "salamat": 1.2, "love": 1.5, "proud": 1.3,
    "nakakaproud": 1.5, "suporta": 1.2, "sumusuporta": 1.2, "laban": 0.8,
    "solid": 1.2, "husay": 1.2, "ganda": 1.0, "saya": 1.2, "galing": 1.2,
    "totoo": 0.8, "tama": 0.8, "due": 0.3, "process": 0.3, "katotohanan": 0.8,
    "kalmado": 0.8, "payapa": 0.8, "sangayon": 1.0, "protect": 0.8,
    "tawa": 0.5, "palakpak": 1.0, "dasal": 0.5, "presidential": 1.0,
    "verify": 0.4,
    # negative tone
    "corrupt": -1.8, "kurakot": -1.8, "sinungaling": -1.7, "magnanakaw": -1.8,
    "anomalya": -1.4, "ghost": -1.0, "dismaya": -1.2, "nakakadismaya": -1.4,
    "galit": -1.4, "nakakagalit": -1.5, "kawawa": -0.8, "divisive": -1.0,
    "fake": -1.0, "harassment": -1.2, "persecution": -1.0, "drama": -0.8,
    "away": -0.8, "gulo": -0.8, "magulo": -0.9, "nakakahiya": -1.3,
    "dismayado": -1.2, "tutol": -1.0, "condemn": -1.2, "imbestigahan": -0.5,
    "managot": -0.8, "evidensya": -0.3, "sarcasm": -0.5, "delusional": -1.2,
    "laban_daw": -0.3,
    # accountability-flavored negative-tone words (tone, not stance)
    "impeach": -0.7, "impeachment": -0.5,
}

NEGATIONS = {"hindi", "huwag", "wag", "wala", "walang", "never", "not",
             "no", "none", "ni", "di", "ayaw", "bawal", "hnd"}
INTENSIFIERS = {"sobra": 1.5, "sobrang": 1.5, "masyado": 1.5, "masyadong": 1.5,
                "grabe": 1.5, "napaka": 1.6, "very": 1.5, "super": 1.5,
                "talaga": 1.3, "tunay": 1.3, "labis": 1.5}


def lexicon_to_df(lex: dict) -> pd.DataFrame:
    """Convert {word: score} dict to the editable GUI table."""
    return pd.DataFrame(sorted(lex.items()),
                        columns=["word", "score"])


def df_to_lexicon(df: pd.DataFrame) -> dict:
    """Convert the edited GUI table back to {word: score}, validated."""
    lex = {}
    for _, row in df.iterrows():
        w = str(row.get("word", "")).strip().lower()
        if not w:
            continue
        try:
            s = float(row.get("score", 0))
        except (TypeError, ValueError):
            raise ValueError(f"Lexicon score for '{w}' must be a number.")
        if not -2.0 <= s <= 2.0:
            raise ValueError(f"Lexicon score for '{w}' must be in [-2, +2].")
        lex[w] = s
    if not lex:
        raise ValueError("Lexicon is empty — add at least a few words.")
    return lex


def vader_compound(text: str) -> float:
    """VADER compound score in [-1, +1]; 0.0 if vader is unavailable."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return 0.0
    if not hasattr(vader_compound, "_analyzer"):
        vader_compound._analyzer = SentimentIntensityAnalyzer()
    return float(vader_compound._analyzer.polarity_scores(
        str(text or ""))["compound"])


def lexicon_score_tokens(tokens: list, lex: dict) -> tuple:
    """Score a token list with negation + intensifier handling.

    Returns (total_score, hits, explanation_string). Explanation lists each
    scoring event so the error-analysis table can show WHY a comment got
    its score (transparent for the defense).
    """
    total, hits, notes = 0.0, 0, []
    i = 0
    pending_neg = False
    pending_mult = 1.0
    neg_window = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATIONS:
            pending_neg, neg_window = True, 3
            notes.append(f"neg({tok})")
            i += 1
            continue
        if tok in INTENSIFIERS:
            pending_mult = INTENSIFIERS[tok]
            notes.append(f"intens({tok}×{pending_mult})")
            i += 1
            continue
        base = lex.get(tok, 0.0)
        if base != 0.0:
            val = base * pending_mult
            if pending_neg:
                val = -val
                notes.append(f"{tok}:{base}→{val:.2f}(negated)")
            elif pending_mult != 1.0:
                notes.append(f"{tok}:{base}→{val:.2f}(intensified)")
            else:
                notes.append(f"{tok}:{val:+.2f}")
            total += val
            hits += 1
            pending_neg, pending_mult, neg_window = False, 1.0, 0
        else:
            if pending_neg:
                neg_window -= 1
                if neg_window <= 0:
                    pending_neg = False
            pending_mult = 1.0
        i += 1
    return total, hits, "; ".join(notes[:8])


def combine_scores(vader_c: float, lex_total: float, lex_hits: int) -> float:
    """Blend VADER ([-1,1]) and lexicon sum into ONE final score.

    Scaling: lexicon average per hit is ~[-2,+2]; we squash with
    lex_total / (lex_total_abs + 2) style normalization... simpler: use the
    MEAN lexicon hit score clipped to [-2,2] divided by 2 -> [-1,1], then
    weighted average 0.4*vader + 0.6*lexicon. If the lexicon saw nothing
    (0 hits), fall back to pure VADER so rare words don't force neutral.
    """
    if lex_hits == 0:
        return float(vader_c)
    lex_mean = max(-2.0, min(2.0, lex_total / max(1, lex_hits))) / 2.0
    return round(0.4 * float(vader_c) + 0.6 * lex_mean, 4)


def classify_score(score: float, pos_thresh: float = 0.15,
                   neg_thresh: float = -0.15) -> str:
    """Map a final score to Positive / Neutral / Negative via sliders."""
    if score >= pos_thresh:
        return "Positive"
    if score <= neg_thresh:
        return "Negative"
    return "Neutral"


def classify_dataframe(df: pd.DataFrame, lex: dict,
                       pos_thresh: float = 0.15,
                       neg_thresh: float = -0.15,
                       text_col: str = "text",
                       tokens_col: str = "tokens") -> pd.DataFrame:
    """Classify every row; add vader, lexicon, final_score, predicted columns."""
    if df is None or df.empty:
        raise ValueError("No cleaned data — run preprocessing first.")
    out = df.copy()
    if tokens_col not in out.columns:
        # Fallback: tokenize clean_text naively.
        src = out["clean_text"] if "clean_text" in out.columns else out[text_col]
        out["_toks"] = src.astype(str).str.lower().apply(
            lambda s: re.findall(r"[a-z0-9ñ]+", s))
        tokens_col = "_toks"
    vaders, lexs, hits, finals, preds, expls = [], [], [], [], [], []
    for _, row in out.iterrows():
        toks = list(row[tokens_col]) if isinstance(row[tokens_col], list) \
            else str(row[tokens_col]).split()
        v = vader_compound(row.get(text_col, ""))
        lt, h, expl = lexicon_score_tokens(toks, lex)
        f = combine_scores(v, lt, h)
        vaders.append(round(v, 4))
        lexs.append(round(lt, 3))
        hits.append(h)
        finals.append(f)
        preds.append(classify_score(f, pos_thresh, neg_thresh))
        expls.append(expl)
    out["vader_compound"] = vaders
    out["lexicon_sum"] = lexs
    out["lexicon_hits"] = hits
    out["final_score"] = finals
    out["predicted"] = preds
    out["score_explanation"] = expls
    if "_toks" in out.columns and tokens_col == "_toks":
        out = out.drop(columns=["_toks"])
    return out


def net_sentiment(predicted: pd.Series) -> dict:
    """Net sentiment = positive% − negative% over a label series.

    Returns dict(pos_pct, neg_pct, neu_pct, net) with net in [-100, +100].
    """
    n = len(predicted)
    if n == 0:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "neu_pct": 0.0, "net": 0.0}
    pos = (predicted == "Positive").mean() * 100
    neg = (predicted == "Negative").mean() * 100
    neu = (predicted == "Neutral").mean() * 100
    return {"pos_pct": round(float(pos), 1), "neg_pct": round(float(neg), 1),
            "neu_pct": round(float(neu), 1),
            "net": round(float(pos - neg), 1)}


def net_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Net-sentiment breakdown per group (platform / topic / bloc)."""
    if "predicted" not in df.columns:
        raise ValueError("Run the classifier first (need 'predicted').")
    rows = []
    for g, sub in df.groupby(group_col):
        m = net_sentiment(sub["predicted"])
        rows.append({"group": g, "n": len(sub), **m})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def sentiment_helper_bullets(df: pd.DataFrame, th: dict) -> list:
    """3-5 plain-English observations from REAL classification output."""
    if df is None or df.empty or "predicted" not in df.columns:
        return ["Classify comments first to get sentiment numbers."]
    overall = net_sentiment(df["predicted"])
    bullets = [f"Overall net sentiment is **{overall['net']:+.1f}** "
               f"({overall['pos_pct']:.0f}% positive − {overall['neg_pct']:.0f}% "
               f"negative, {overall['neu_pct']:.0f}% neutral)."]
    nz = th.get("neutral_zone", 10.0)
    if abs(overall["net"]) <= nz:
        bullets.append(f"That sits inside the neutral zone (±{nz:.0f}) — the "
                       f"discourse as collected is balanced, not one-sided.")
    elif overall["net"] > 0:
        bullets.append("Net sentiment leans positive — check whether praise words "
                       "come from one bloc or are spread across groups.")
    else:
        bullets.append("Net sentiment leans negative — read the top negative "
                       "examples to see if anger targets the process, a person, "
                       "or the other camp.")
    # Coverage warning: comments the lexicon never fired on.
    nohit = (df["lexicon_hits"] == 0).mean() * 100 if "lexicon_hits" in df else 0
    if nohit > 30:
        bullets.append(f"**{nohit:.0f}%** of comments had zero lexicon hits — the "
                       f"classifier leaned on VADER alone there. Add missing "
                       f"Tagalog words to the lexicon table.")
    # Threshold sensitivity hint.
    near = ((df["final_score"].abs() < 0.25)).mean() * 100
    if near > 40:
        bullets.append(f"**{near:.0f}%** of scores sit near zero — small slider "
                       f"moves will flip many labels. Report the thresholds you used.")
    bullets.append("Next step: validate on Page 7. If Negative recall is low there, "
                   "expand the negative side of the lexicon first.")
    return bullets[:5]
