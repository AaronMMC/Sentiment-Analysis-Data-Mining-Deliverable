"""
evaluation.py — Validation & evaluation done RIGHT (no leakage).

Protocol enforced by the app (explain this in your defense!):
  STEP 1 — One-click stratified 10% HOLDOUT split (fixed random_state=42),
           created BEFORE any tuning. Once created it is LOCKED: the app
           refuses to re-split (a "Reset holdout" button exists but warns
           that re-splitting after peeking = leakage).
  STEP 2 — 10-fold STRATIFIED cross-validation on the remaining 90% for
           tuning thresholds / lexicon weights. Reports mean ± SD per metric.
  STEP 3 — FINAL one-time evaluation on the locked holdout. If the CV
           score minus holdout score > 0.10 -> overfit flag.

Metrics (all from sklearn, all computed on REAL labels):
  Precision / Recall / F1 per class + macro + weighted, Accuracy,
  confusion-matrix heatmap data, and the topic Coherence score (from Page 5).

Error analysis: every misclassified holdout comment is listed with a
guessed reason from transparent heuristics (NOT a second classifier):
  sarcasm markers, code-switching, negation present, stance-vs-tone
  mismatch (bloc vs. predicted), or unknown-vocabulary.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split

LABELS = ["Positive", "Negative", "Neutral"]

SARCASM_MARKERS = {"charot", "sarcasm", "lol", "lmao", "smh", "🙄", "😏",
                   "daw", "kuno", "eme"}
NEGATION_MARKERS = {"hindi", "huwag", "wag", "wala", "walang", "never",
                    "not", "no", "ayaw", "di"}


def labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that HAVE a gold (manual) label — unlabeled rows cannot
    be used for supervised evaluation."""
    if df is None or df.empty:
        raise ValueError("No classified data — run sentiment first.")
    if "manual_sentiment" not in df.columns or "predicted" not in df.columns:
        raise ValueError("Need 'manual_sentiment' (gold) and 'predicted'.")
    sub = df[df["manual_sentiment"].isin(LABELS)].copy()
    if len(sub) < 15:
        raise ValueError(f"Only {len(sub)} labeled rows — need at least 15 "
                         f"(label more comments in the labeling tool).")
    # Each class needs >= 2 examples for stratified splitting.
    counts = sub["manual_sentiment"].value_counts()
    rare = counts[counts < 2]
    if not rare.empty:
        raise ValueError("Stratified splitting needs ≥2 examples per class. "
                         f"Rare: {rare.to_dict()}. Label more comments.")
    return sub.reset_index(drop=True)


def make_holdout(df_labeled: pd.DataFrame, holdout_frac: float = 0.10,
                 random_state: int = 42) -> tuple:
    """Stratified holdout split. Returns (dev_df_90pct, holdout_df_10pct)."""
    dev, hold = train_test_split(
        df_labeled, test_size=float(holdout_frac),
        random_state=int(random_state),
        stratify=df_labeled["manual_sentiment"])
    return dev.reset_index(drop=True), hold.reset_index(drop=True)


def _scores_for_frames(dev: pd.DataFrame, classify_fn) -> pd.DataFrame:
    """Helper: re-classify a frame with a candidate classify_fn.

    classify_fn(df) -> predicted Series. Used to re-tune thresholds inside
    CV without leaking the holdout.
    """
    return classify_fn(dev)


def cross_validate(dev: pd.DataFrame, classify_fn, n_splits: int = 10,
                   random_state: int = 42) -> pd.DataFrame:
    """10-fold stratified CV on the DEV set with a candidate classifier.

    Args:
        dev: the 90% dev frame (must include gold + features the
            classify_fn needs, e.g. tokens/text/lexicon settings).
        classify_fn: callable dev_fold_df -> predicted labels (list/Series).
        n_splits: folds (reduced automatically if a class is rarer than
            n_splits — with a note — so tiny student datasets still run).

    Returns a per-fold metrics DataFrame (accuracy, macro_f1, per-class F1…).
    """
    y = dev["manual_sentiment"].values
    min_count = pd.Series(y).value_counts().min()
    k = max(2, min(int(n_splits), int(min_count)))
    skf = StratifiedKFold(n_splits=k, shuffle=True,
                          random_state=int(random_state))
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(dev, y), 1):
        va = dev.iloc[va_idx]
        try:
            pred = list(classify_fn(va))
        except Exception as e:
            raise RuntimeError(f"Classifier failed inside CV fold {fold}: "
                               f"{e}") from e
        true = va["manual_sentiment"].tolist()
        rep = classification_report(true, pred, labels=LABELS,
                                    output_dict=True, zero_division=0)
        rows.append({
            "fold": fold,
            "accuracy": round(accuracy_score(true, pred), 4),
            "macro_f1": round(rep["macro avg"]["f1-score"], 4),
            "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
            "f1_Positive": round(rep["Positive"]["f1-score"], 4),
            "f1_Negative": round(rep["Negative"]["f1-score"], 4),
            "f1_Neutral": round(rep["Neutral"]["f1-score"], 4),
        })
    note = None
    if k < n_splits:
        note = (f"Used {k}-fold instead of {n_splits}-fold because the "
                f"rarest class has only {min_count} dev examples.")
    out = pd.DataFrame(rows)
    out.attrs["note"] = note
    return out


def final_metrics(true: list, pred: list) -> dict:
    """Full metric bundle for the FINAL holdout evaluation."""
    rep = classification_report(true, pred, labels=LABELS,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(true, pred, labels=LABELS)
    per_class = {lab: {"precision": round(rep[lab]["precision"], 4),
                       "recall": round(rep[lab]["recall"], 4),
                       "f1": round(rep[lab]["f1-score"], 4),
                       "support": int(rep[lab]["support"])}
                 for lab in LABELS}
    return {
        "accuracy": round(float(accuracy_score(true, pred)), 4),
        "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(rep["weighted avg"]["f1-score"]), 4),
        "macro_precision": round(float(rep["macro avg"]["precision"]), 4),
        "macro_recall": round(float(rep["macro avg"]["recall"]), 4),
        "per_class": per_class,
        "confusion_matrix": cm,  # rows=true, cols=predicted, order=LABELS
        "report_df": pd.DataFrame(rep).transpose().round(4),
    }


def cv_summary(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD across folds for each metric column."""
    mets = [c for c in cv_df.columns if c != "fold"]
    rows = [{"metric": m,
             "mean": round(float(cv_df[m].mean()), 4),
             "sd": round(float(cv_df[m].std(ddof=1)), 4)
             if len(cv_df) > 1 else 0.0}
            for m in mets]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Error analysis with transparent heuristics
# ---------------------------------------------------------------------------
def guess_error_reason(row) -> str:
    """Guess WHY a row was misclassified (heuristic, human-checkable).

    Priority order mirrors what students should look for first:
      1. stance-vs-tone mismatch, 2. negation, 3. sarcasm, 4. code-switch,
      5. unknown vocabulary, else 'unclear — read manually'.
    """
    text = str(row.get("text", "")).lower()
    toks = set(text.replace("/", " ").split())
    pred, gold = row.get("predicted"), row.get("manual_sentiment")
    bloc = str(row.get("political_bloc", ""))
    # 1. stance vs tone: angry words but pro-bloc labeled positive etc.
    if pred == "Negative" and gold == "Positive" and "Pro-Duterte" in bloc:
        return ("stance-vs-tone: angry wording but pro-stance gold label — "
                "classifier read tone, annotator read stance")
    if pred == "Positive" and gold == "Negative":
        if any(w in toks for w in ("protect", "solid", "love", "support")):
            return ("stance-vs-tone: supportive words for one side read as "
                    "positive tone but gold label is negative")
    # 2. negation present?
    if toks & NEGATION_MARKERS:
        return "negation: 'hindi/not/wala…' may have flipped meaning"
    # 3. sarcasm markers?
    if toks & SARCASM_MARKERS or "?" in text and "!" in text:
        return "sarcasm/irony markers present (charot/lol/🙄/?!…)"
    # 4. code-switching: substantial Tagalog AND English content words?
    tag = {"ang", "ng", "sa", "na", "hindi", "mga", "ako", "ikaw"}
    eng = {"the", "and", "is", "impeachment", "accountability", "evidence"}
    if toks & tag and toks & eng:
        return "code-switching (Taglish mix) — lexicon may cover one language only"
    # 5. unknown vocab: classifier saw no lexicon words?
    if int(row.get("lexicon_hits", 1)) == 0:
        return "unknown vocabulary: no lexicon word fired (VADER-only decision)"
    return "unclear — read the comment manually"


def error_table(hold_df: pd.DataFrame) -> pd.DataFrame:
    """Misclassified holdout rows + guessed reasons (for manual review)."""
    mis = hold_df[hold_df["predicted"] != hold_df["manual_sentiment"]].copy()
    if mis.empty:
        return mis
    mis["likely_reason"] = mis.apply(guess_error_reason, axis=1)
    cols = [c for c in ["text", "platform", "political_bloc",
                        "manual_sentiment", "predicted", "final_score",
                        "score_explanation", "likely_reason"]
            if c in mis.columns]
    return mis[cols].reset_index(drop=True)


def evaluation_helper_bullets(final: dict, cv_sum: pd.DataFrame,
                              cv_df: pd.DataFrame, holdout_n: int,
                              coherence: float, th: dict) -> list:
    """3-5 plain-English observations from REAL evaluation numbers."""
    bullets = []
    acc, mf1 = final["accuracy"], final["macro_f1"]
    if acc < th["acc_red_max"] or mf1 < th["acc_red_max"]:
        bullets.append(f"Holdout accuracy **{acc:.2f}** / macro-F1 **{mf1:.2f}** "
                       f"is RED (< {th['acc_red_max']}) — the classifier struggles. "
                       f"Expand the lexicon and re-tune thresholds on the DEV set only.")
    elif acc <= th["acc_yellow_max"] or mf1 <= th["acc_yellow_max"]:
        bullets.append(f"Holdout accuracy **{acc:.2f}** / macro-F1 **{mf1:.2f}** "
                       f"is YELLOW — acceptable for a class project; report per-class "
                       f"weaknesses honestly.")
    else:
        bullets.append(f"Holdout accuracy **{acc:.2f}** / macro-F1 **{mf1:.2f}** "
                       f"is GREEN — strong for a rule-based system. Still check the "
                       f"error table for systematic blind spots.")
    # Per-class flags.
    weak = [f"{lab} (P {v['precision']:.2f}/R {v['recall']:.2f})"
            for lab, v in final["per_class"].items()
            if v["precision"] < th["per_class_min"]
            or v["recall"] < th["per_class_min"]]
    if weak:
        bullets.append("Weak classes (< 0.50 precision or recall): **"
                       + "; ".join(weak) + "**. Low NEGATIVE recall means the "
                       "classifier misses many negative comments — add negative "
                       "lexicon words.")
    # CV stability.
    try:
        sd = float(cv_sum.loc[cv_sum["metric"] == "macro_f1", "sd"].iloc[0])
        if sd > th["cv_sd_flag"]:
            bullets.append(f"CV macro-F1 SD is **{sd:.3f}** (> {th['cv_sd_flag']}) — "
                           f"UNSTABLE across folds. Collect more data for rare classes.")
        else:
            bullets.append(f"CV is stable (macro-F1 SD **{sd:.3f}**) — tuning results "
                           f"should generalize.")
    except Exception:
        pass
    # Overfit check.
    try:
        cv_mean = float(cv_sum.loc[cv_sum["metric"] == "macro_f1",
                                   "mean"].iloc[0])
        gap = cv_mean - mf1
        if gap > th["overfit_gap_flag"]:
            bullets.append(f"Possible OVERFITTING: CV macro-F1 ({cv_mean:.2f}) exceeds "
                           f"holdout ({mf1:.2f}) by **{gap:.2f}** (> "
                           f"{th['overfit_gap_flag']}). You may have tuned to the dev "
                           f"set — simplify the lexicon.")
        else:
            bullets.append(f"No overfit signal (CV − holdout = **{gap:+.2f}**).")
    except Exception:
        pass
    if coherence is not None:
        bullets.append(f"Topic Coherence Cv is **{coherence:.3f}** — report it next to "
                       f"classifier metrics as your unsupervised quality signal.")
    bullets.append(f"Evaluated on **{holdout_n}** locked holdout comments. Never "
                   f"re-tune on these — describe errors, don't fix them here.")
    return bullets[:6]
