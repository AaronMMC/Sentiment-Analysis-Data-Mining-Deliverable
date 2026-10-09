"""
sentiment_hf.py — Auxiliary transformer sentiment (comparison track).

Role (see Main_obj.md): the GRADED classifier stays rule-based
(VADER + lexicon in sentiment.py). This module adds a transformer
comparison that needs NO manual lexicon and handles Taglish context better.

Default model: jjjardev/tagasenti (XLM-R-large, Tagalog/Taglish 3-class).
Alternatives in one line: cardiffnlp/twitter-xlm-roberta-base-sentiment
(multilingual tweets, 3-class) or nlptown/... (EU product reviews, 1-5
stars — domain-mismatched, baseline only).

GPU: uses CUDA when available (RTX 4060 tested target), fp16 on CUDA,
fallback to CPU with a clear message. Batch inference with a disk cache
(saved_state/hf_cache/<fingerprint>.csv) so repeat runs NEVER re-run the
GPU: if the cache file for this exact (model_id + texts + max_length)
exists, it is loaded instead of re-inferring.

Simple version of the CV question:
  full re-inference CV = read all comments 10 times (slow).
  cache-once          = read once, save answers, grade the saved answers
                        10 times (fast, same numbers). We do cache-once.
"""

import hashlib
import os
from pathlib import Path

import pandas as pd

DEFAULT_MODEL_ID = "jjjardev/tagasenti"
CACHE_DIR = "saved_state/hf_cache"

_PIPELINE_CACHE = {}


# ---------------------------------------------------------------------------
# Device / pipeline (lazy so `import` never requires torch)
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Return 'cuda' if a GPU is available else 'cpu' (no import crash)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def device_detail() -> str:
    """Human-readable GPU description for the GUI/remarks table."""
    try:
        import torch
    except ImportError:
        return "cpu (torch not installed — install the CUDA build from " \
               "requirements.txt for GPU speed)"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"cuda ({name}, sm_{cap}, {total:.1f} GB)"
    except Exception as e:
        return f"cpu (cuda check failed: {e})"
    return "cpu (no CUDA GPU detected)"


def get_hf_pipeline(model_id: str = DEFAULT_MODEL_ID):
    """Lazy singleton HF text-classification pipeline (GPU when present).

    Raises a helpful error if transformers/torch or the model weights are
    unavailable (offline machine, wrong Python) instead of crashing.
    """
    if model_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_id]
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transformers + torch are required for the comparison track. "
            "Run: py -m pip install -r requirements.txt "
            "(torch CUDA wheel for your RTX 4060).") from e
    device = 0 if get_device() == "cuda" else -1
    dtype = None
    try:
        if device == 0:
            dtype = torch.float16
    except Exception:
        dtype = None
    try:
        pipe = pipeline("text-classification", model=model_id,
                        device=device, top_k=None,
                        torch_dtype=dtype, truncation=True)
    except TypeError:
        # Older transformers without torch_dtype/top_k kwargs.
        pipe = pipeline("text-classification", model=model_id,
                        device=device, return_all_scores=True)
    _PIPELINE_CACHE[model_id] = pipe
    return pipe


# ---------------------------------------------------------------------------
# Label mapping: any HF label set -> Positive / Negative / Neutral
# ---------------------------------------------------------------------------
def map_hf_label(raw_label: str) -> str:
    """Normalize an HF label to our Positive/Negative/Neutral schema."""
    s = str(raw_label or "").strip().lower()
    if "pos" in s or s in {"4 stars", "5 stars", "4", "5", "label_2"}:
        return "Positive"
    if "neg" in s or s in {"1 star", "2 stars", "1", "2", "label_0"}:
        return "Negative"
    return "Neutral"


def star_of_label(raw_label: str):
    """Return 1-5 star int for star models (nlptown), else None."""
    s = str(raw_label or "").strip().lower()
    for n in (1, 2, 3, 4, 5):
        if s == str(n) or s.startswith(f"{n} star"):
            return n
    return None


# ---------------------------------------------------------------------------
# Disk cache: fingerprint -> CSV (never re-run GPU when cache exists)
# ---------------------------------------------------------------------------
def fingerprint(texts: list, model_id: str, max_length: int) -> str:
    """Short sha1 over model_id + max_length + ordered texts."""
    h = hashlib.sha1()
    h.update(f"{model_id}|{int(max_length)}|".encode("utf-8"))
    for t in texts:
        h.update(str(t or "").encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def cache_path(fp: str, cache_dir: str = CACHE_DIR) -> Path:
    """Full path of the cache CSV for a fingerprint."""
    return Path(cache_dir) / f"hf_preds_{fp}.csv"


def try_load_cache(texts: list, model_id: str, max_length: int,
                   cache_dir: str = CACHE_DIR):
    """Return cached DataFrame if it exists AND matches these texts."""
    fp = fingerprint(texts, model_id, max_length)
    p = cache_path(fp, cache_dir)
    if not p.exists():
        return None, fp
    try:
        cached = pd.read_csv(p)
    except Exception:
        return None, fp
    if len(cached) != len(texts):
        return None, fp
    try:
        if "text_light" in cached.columns:
            if cached["text_light"].astype(str).tolist() != \
                    [str(t or "") for t in texts]:
                return None, fp
    except Exception:
        return None, fp
    return cached, fp


def classify_transformer_dataframe(
        df: pd.DataFrame, model_id: str = DEFAULT_MODEL_ID,
        text_col: str = "text_light", batch_size: int = 32,
        max_length: int = 512, cache_dir: str = CACHE_DIR,
        use_cache: bool = True, progress_cb=None) -> pd.DataFrame:
    """Batch-classify with the HF model; load disk cache when available.

    Args:
        df: frame with text_col (prefer 'text_light' from preprocessing).
        progress_cb: optional fn(done, total) for Streamlit progress bars.

    Returns a copy of df + [hf_label_raw, hf_confidence, hf_stars,
    predicted_hf, final_score_hf, score_explanation_hf, hf_model, hf_fp].
    predicted_hf follows the Positive/Negative/Neutral schema so
    net_sentiment()/evaluation work unchanged.
    """
    if df is None or df.empty:
        raise ValueError("No data to classify — run preprocessing first.")
    src_col = text_col if text_col in df.columns else \
        ("text" if "text" in df.columns else None)
    if src_col is None:
        raise ValueError("Need a text column (text_light or text).")
    texts = df[src_col].astype(str).fillna("").tolist()

    if use_cache:
        cached, fp = try_load_cache(texts, model_id, max_length, cache_dir)
        if cached is not None:
            out = df.copy().reset_index(drop=True)
            for c in ["hf_label_raw", "hf_confidence", "hf_stars",
                      "predicted_hf", "final_score_hf",
                      "score_explanation_hf"]:
                if c in cached.columns:
                    out[c] = cached[c].values
            out["hf_model"] = model_id
            out["hf_fp"] = fp
            out["hf_cached"] = True
            return out

    pipe = get_hf_pipeline(model_id)
    raw_labels, confs = [], []
    total = len(texts)
    for start in range(0, total, max(1, int(batch_size))):
        chunk = [t if t.strip() else "." for t in
                 texts[start:start + int(batch_size)]]
        try:
            results = pipe(chunk, truncation=True, max_length=int(max_length))
        except TypeError:
            results = pipe(chunk)
        # pipeline returns per-text list-of-dicts (top_k) or dict.
        for r in results:
            if isinstance(r, dict):
                raw_labels.append(str(r.get("label", "neutral")))
                confs.append(float(r.get("score", 0.0)))
            elif isinstance(r, (list, tuple)) and r:
                best = max(r, key=lambda d: float(d.get("score", 0.0)))
                raw_labels.append(str(best.get("label", "neutral")))
                confs.append(float(best.get("score", 0.0)))
            else:
                raw_labels.append("neutral")
                confs.append(0.0)
        if progress_cb is not None:
            try:
                progress_cb(min(start + int(batch_size), total), total)
            except Exception:
                pass

    fp = fingerprint(texts, model_id, max_length)
    out = df.copy().reset_index(drop=True)
    out["text_light"] = texts
    out["hf_label_raw"] = raw_labels
    out["hf_confidence"] = [round(float(c), 4) for c in confs]
    out["hf_stars"] = [star_of_label(l) for l in raw_labels]
    out["predicted_hf"] = [map_hf_label(l) for l in raw_labels]
    # final_score_hf in [-1, +1]: signed confidence (neutral -> ~0).
    scores = []
    for lab, c in zip(out["predicted_hf"], out["hf_confidence"]):
        if lab == "Positive":
            scores.append(round(float(c), 4))
        elif lab == "Negative":
            scores.append(round(-float(c), 4))
        else:
            scores.append(round((float(c) - 0.5) * 0.2, 4))
    out["final_score_hf"] = scores
    out["score_explanation_hf"] = [
        f"{m} {lab} {conf:.2f}" for m, lab, conf in
        zip(out["hf_label_raw"], out["predicted_hf"], out["hf_confidence"])]
    out["hf_model"] = model_id
    out["hf_fp"] = fp
    out["hf_cached"] = False
    # Persist cache (never re-run GPU for this exact input again).
    try:
        os.makedirs(cache_dir, exist_ok=True)
        keep = ["text_light", "hf_label_raw", "hf_confidence", "hf_stars",
                "predicted_hf", "final_score_hf", "score_explanation_hf"]
        out[keep].to_csv(cache_path(fp, cache_dir), index=False)
    except Exception:
        pass
    return out


def hf_helper_bullets(hf_df: pd.DataFrame) -> list:
    """Plain-English bullets for the transformer comparison (real numbers)."""
    if hf_df is None or hf_df.empty or "predicted_hf" not in hf_df.columns:
        return ["Run the transformer comparison to get its numbers."]
    n = len(hf_df)
    pos = (hf_df["predicted_hf"] == "Positive").mean() * 100
    neg = (hf_df["predicted_hf"] == "Negative").mean() * 100
    neu = (hf_df["predicted_hf"] == "Neutral").mean() * 100
    bullets = [
        f"Transformer ({hf_df['hf_model'].iloc[0]}) on {n} rows: "
        f"{pos:.0f}% positive / {neg:.0f}% negative / {neu:.0f}% neutral "
        f"(vs rule-based above — agreement table shows where they differ)."]
    if "hf_confidence" in hf_df.columns:
        low = (hf_df["hf_confidence"] < 0.6).mean() * 100
        bullets.append(
            f"Mean confidence {hf_df['hf_confidence'].mean():.2f}; "
            f"{low:.0f}% rows below 0.60 — read those first in the audit table.")
    if bool(hf_df.get("hf_cached", pd.Series([False])).iloc[0]):
        bullets.append("Loaded from disk cache — GPU was NOT re-run "
                       f"(cache {hf_df['hf_fp'].iloc[0]}).")
    else:
        bullets.append("Fresh GPU run — predictions saved to "
                       f"saved_state/hf_cache/{hf_df['hf_fp'].iloc[0]}.csv "
                       "so the next run is instant.")
    return bullets[:4]
