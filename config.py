"""
config.py — Single place for ALL benchmark thresholds used in the app.

Why this file exists (for your oral defense):
  Every badge (green/yellow/red) and every "Analysis Helper" message in the
  app reads its cut-offs from DEFAULT_THRESHOLDS below. That means:
    1. Nothing is hard-coded deep inside the code.
    2. You can change a threshold in the GUI sidebar and immediately see
       badges update — useful to discuss sensitivity in your defense.
    3. These are RULE-OF-THUMB guides, not strict rules. Small or difficult
       datasets will score lower even when everything is done correctly.

How to use:
  from config import DEFAULT_THRESHOLDS
"""

DEFAULT_THRESHOLDS = {
    # --- Records collected ---
    "records_red_max": 200,      # red if n < 200
    "records_yellow_max": 300,   # yellow if 200 <= n <= 300, green if n > 300

    # --- Class balance ---
    "min_class_share": 0.15,     # warn if ANY sentiment class < 15% of data

    # --- Annotator agreement (Cohen's kappa) ---
    "kappa_red_max": 0.4,        # red if kappa < 0.4
    "kappa_yellow_max": 0.6,     # yellow if 0.4 <= kappa <= 0.6, green if > 0.6

    # --- Topic coherence (Cv) ---
    "coherence_red_max": 0.35,   # red if Cv < 0.35
    "coherence_yellow_max": 0.50,  # yellow 0.35-0.50, green > 0.50

    # --- Cosine similarity between groups ---
    "cosine_low_max": 0.3,       # low if < 0.3
    "cosine_mod_max": 0.6,       # moderate 0.3-0.6, high > 0.6

    # --- Jaccard similarity ---
    "jaccard_low_max": 0.1,      # low if < 0.1
    "jaccard_mod_max": 0.3,      # moderate 0.1-0.3, high > 0.3

    # --- Accuracy / Macro-F1 ---
    "acc_red_max": 0.60,         # red if < 0.60
    "acc_yellow_max": 0.75,      # yellow 0.60-0.75, green > 0.75

    # --- Per-class precision/recall floor ---
    "per_class_min": 0.50,       # flag any class below 0.50

    # --- CV stability: std across folds ---
    "cv_sd_flag": 0.05,          # flag if SD across folds > 0.05

    # --- Overfit check: CV score minus holdout score ---
    "overfit_gap_flag": 0.10,    # flag if (CV - holdout) > 0.10

    # --- Net sentiment neutral zone (in percentage points) ---
    # Net sentiment = positive% - negative%, range -100..+100.
    # Values inside [-neutral_zone, +neutral_zone] are called "neutral zone".
    "neutral_zone": 10.0,

    # --- Reproducibility ---
    "random_state": 42,          # fixed seed for holdout split + CV shuffling
    "holdout_frac": 0.10,        # 10% holdout
    "cv_folds": 10,              # 10-fold stratified CV
}

# Transformer comparison track (auxiliary — the graded model is rule-based).
# TagaSenti = XLM-R-large finetuned on Tagalog/Taglish 3-class sentiment.
HF_DEFAULTS = {
    "model_id": "jjjardev/tagasenti",
    "fallback_model_id": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "batch_size": 32,            # fits RTX 4060 8GB at 512 tokens (fp16)
    "max_length": 512,           # HF truncation length
    "cache_dir": "saved_state/hf_cache",
}

# Human-readable one-liners shown next to the threshold editor.
THRESHOLD_HELP = {
    "records_red_max": "Fewer than this many records = red badge.",
    "records_yellow_max": "Up to this many records = yellow; above = green.",
    "min_class_share": "Warn if any sentiment class is rarer than this share.",
    "kappa_red_max": "Kappa below this = poor annotator agreement.",
    "kappa_yellow_max": "Kappa above this = good agreement.",
    "coherence_red_max": "Coherence below this = topics hard to interpret.",
    "coherence_yellow_max": "Coherence above this = topics hang together well.",
    "cosine_low_max": "Cosine below this = groups use very different language.",
    "cosine_mod_max": "Cosine above this = groups use very similar language.",
    "jaccard_low_max": "Jaccard below this = little word overlap.",
    "jaccard_mod_max": "Jaccard above this = strong word overlap.",
    "acc_red_max": "Accuracy/Macro-F1 below this = red.",
    "acc_yellow_max": "Accuracy/Macro-F1 above this = green.",
    "per_class_min": "Flag any class precision/recall below this.",
    "cv_sd_flag": "Flag CV instability if SD across folds exceeds this.",
    "overfit_gap_flag": "Flag possible overfitting if CV minus holdout exceeds this.",
    "neutral_zone": "Net-sentiment values inside ±this are called neutral.",
    "random_state": "Fixed seed so the holdout split is reproducible.",
    "holdout_frac": "Share of labeled data locked away as the final holdout.",
    "cv_folds": "Number of stratified folds on the remaining 90%.",
}
