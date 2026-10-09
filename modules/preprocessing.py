"""
preprocessing.py — Taglish (Tagalog-English code-switch) cleaning pipeline.

Pipeline order (each step is counted so the app can show "records at each
step" and a before/after table):
  1. lowercase
  2. remove URLs, optionally map emojis -> sentiment tokens (or strip them)
  3. remove @mentions; keep hashtag WORDS but drop the '#' symbol
  4. collapse repeated characters ("graaabe" -> "grabe", "!!!" -> "!")
  5. normalize Filipino slang / spelling variants via TAGLISH_SLANG map
  6. tokenize (simple regex keeping Filipino/English word chars)
  7. remove English + Filipino + custom stopwords (editable in GUI)
  8. drop duplicates and very short rows (< min_tokens tokens)

Outputs:
  - df_clean with columns: ... + clean_text, tokens
  - log: list of (step_name, n_rows) for the funnel display
  - word-frequency table for the chart

Design note: we use a light regex tokenizer + rule maps instead of a heavy
dependency so behavior is transparent and explainable in a defense. NLTK is
used for the English stopword list (with a built-in fallback if the NLTK
data download is missing).

Dual-track design (see Main_obj.md workflow):
  Track A (LDA / TF-IDF): full cleaning below -> clean_text + tokens
    (stopwords removed, slang normalized). Bag-of-words models need this.
  Track B (transformer / VADER comparison): light_clean() -> text_light
    (URLs/mentions stripped, case kept simple, NO stopword removal, NO slang
    collapsing). Contextual models need the original wording, including
    negations like 'hindi' and intensifiers like 'sobra'.

Stopword sources (priority order):
  1. `stopwordsiso` pip package: stopwords("tl") + stopwords("en")
     (stopwords-iso / stopwords-tl, MIT).
  2. Vendored data/stopwords_tl.txt (+ NLTK English) for offline use.
  3. Built-in FILIPINO_STOPWORDS + NLTK fallback below.
Slang map: data/taglish_slang.csv if present, else TAGLISH_SLANG.
"""

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

# --- Emoji -> sentiment token map (used when keep_emojis_as_tokens=True) ---
EMOJI_MAP = {
    "😂": " tawa ", "🤣": " tawa ", "😊": " saya ", "😍": " love ",
    "❤": " love ", "❤️": " love ", "💪": " laban ", "👏": " palakpak ",
    "😡": " galit ", "😠": " galit ", "🤬": " galit ", "😢": " lungkot ",
    "😭": " lungkot ", "👍": " sangayon ", "👎": " tutol ",
    "🙏": " dasal ", "🔥": " intense ", "💯": " totoo ",
    "😏": " sarcasm ", "🙄": " sarcasm ",
}

# --- Common Filipino slang / spelling variants -> normalized form ---
# Keep this small and explainable; students can extend it in the GUI.
TAGLISH_SLANG = {
    "d2": "dito", "d2o": "dito",
    "hndi": "hindi", "hinde": "hindi", "ndi": "hindi", "di": "hindi",
    "wla": "wala", "walaa": "wala", "meron": "mayroon",
    "ako": "ako", "akoo": "ako", "ikaw": "ikaw", "kayo": "kayo",
    "grabe": "grabe", "grabee": "grabe", "graaabe": "grabe",
    "sobra": "sobra", "sobrang": "sobra",
    "kc": "kasi", "kase": "kasi", "kasi": "kasi",
    "lng": "lang", "lamang": "lang", "lang": "lang",
    "naman": "naman", "nman": "naman",
    "talaga": "talaga", "tlga": "talaga", "tlagah": "talaga",
    "syempre": "siyempre", "siyempre": "siyempre", "syempreh": "siyempre",
    "oo": "oo", "opo": "oo", "hindi": "hindi", "po": "po",
    "pls": "please", "plss": "please", "please": "please",
    "tbh": "sa totoo", "lol": "tawa", "lmao": "tawa", "smh": "dismaya",
    "ely": "eli", "eli5": "eli",
    "dds": "dds", "dilawan": "dilawan", "kakampink": "kakampink",
    "fake": "fake", "news": "news",
    "impeach": "impeach", "impeachment": "impeachment",
    "corrupt": "corrupt", "kurakot": "kurakot",
    "charot": "charot", "char": "charot",
}

# --- Core Filipino stopwords (function words with little topic meaning) ---
FILIPINO_STOPWORDS = {
    "ang", "ng", "sa", "na", "ay", "at", "mga", "ang", "yung", "nang",
    "para", "kasi", "kundi", "ni", "kay", "sina", "sila", "kami", "tayo",
    "ako", "ikaw", "siya", "kami", "kayo", "nila", "namin", "natin",
    "ito", "iyan", "iyon", "dito", "diyan", "doon", "rin", "din", "daw",
    "raw", "ba", "po", "opo", "lang", "lamang", "naman", "pala", "kaya",
    "kundi", "kung", "kapag", "dahil", "upang", "habang", "mismo", "pa",
    "man", "nga", "eh", "oh", "ano", "alin", "sino", "kanino", "ilan",
    "ito", "iyon", "niya", "kaniya", "nito", "niyan", "noon", "ngayon",
    "dito", "ganyan", "ganito", "ganoon", "mas", "may", "wala", "hindi",
    "oo", "yata", "ata", "siguro", "dapat", "pwede", "puwede", "sobra",
}


def get_english_stopwords() -> set:
    """English stopwords: stopwordsiso(en) -> NLTK -> built-in fallback.

    We try nltk.corpus.stopwords; if the NLTK data pack was never
    downloaded (offline machine), we fall back to a compact built-in set
    so the app still runs — and the GUI tells the student to run the
    nltk.download lines from the README.
    """
    try:
        from stopwordsiso import stopwords as iso_stopwords
        words = set(iso_stopwords("en"))
        if words:
            return words
    except Exception:
        pass
    try:
        from nltk.corpus import stopwords as nltk_sw
        return set(nltk_sw.words("english"))
    except Exception:
        return {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "he", "him", "his", "she", "her",
            "it", "its", "they", "them", "their", "theirs", "what",
            "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing",
            "a", "an", "the", "and", "but", "if", "or", "because", "as",
            "until", "while", "of", "at", "by", "for", "with", "about",
            "into", "through", "during", "before", "after", "above",
            "below", "to", "from", "up", "down", "in", "out", "on",
            "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "can", "will", "just", "don", "should",
            "now",
        }


def get_tagalog_stopwords() -> tuple:
    """Tagalog stopwords from stopwordsiso('tl'), with offline fallbacks.

    Returns (words, source_name) where source_name is one of
    'stopwordsiso:tl', 'vendored:data/stopwords_tl.txt', 'builtin'.
    """
    try:
        from stopwordsiso import stopwords as iso_stopwords
        words = {w.strip().lower() for w in iso_stopwords("tl") if w.strip()}
        if words:
            return words, "stopwordsiso:tl"
    except Exception:
        pass
    for cand in [Path("data/stopwords_tl.txt"),
                 Path(__file__).resolve().parent.parent / "data" /
                 "stopwords_tl.txt"]:
        try:
            if cand.exists():
                words = {ln.strip().lower() for ln in
                         cand.read_text(encoding="utf-8").splitlines()
                         if ln.strip() and not ln.strip().startswith("#")}
                if words:
                    return words, "vendored:data/stopwords_tl.txt"
        except Exception:
            continue
    return set(FILIPINO_STOPWORDS), "builtin"


def stopwords_source() -> str:
    """Human-readable stopword provenance for the GUI/remarks table."""
    try:
        tl, tl_src = get_tagalog_stopwords()
    except Exception:
        return "builtin (fallback)"
    try:
        from stopwordsiso import stopwords as iso_stopwords
        en = set(iso_stopwords("en"))
        en_src = "stopwordsiso:en" if en else "nltk/builtin"
    except Exception:
        en_src = "nltk/builtin"
    return f"tl={tl_src} ({len(tl)} words) + en={en_src}"


@lru_cache(maxsize=1)
def get_slang_map() -> dict:
    """Slang map: data/taglish_slang.csv if present, else TAGLISH_SLANG."""
    for cand in [Path("data/taglish_slang.csv"),
                 Path(__file__).resolve().parent.parent / "data" /
                 "taglish_slang.csv"]:
        try:
            if cand.exists():
                df = pd.read_csv(cand)
                cols = {str(c).lower(): c for c in df.columns}
                if "variant" in cols and "canonical" in cols:
                    m = {str(v).strip().lower(): str(c).strip().lower()
                         for v, c in
                         zip(df[cols["variant"]], df[cols["canonical"]])
                         if str(v).strip()}
                    if m:
                        return m
        except Exception:
            continue
    return dict(TAGLISH_SLANG)


def get_default_stopwords(custom_extra=None) -> set:
    """Union of English + Tagalog + optional custom stopwords.

    IMPORTANT: negation words ('hindi', 'not', ...) and intensifiers
    ('sobra', 'very', ...) are REMOVED from the stopword set even if they
    appear in the NLTK/Filipino lists — the sentiment classifier NEEDS
    those tokens to flip/amplify scores (see sentiment.py). Dropping them
    here would silently disable negation handling.
    """
    tl_words, _src = get_tagalog_stopwords()
    base = set(get_english_stopwords()) | set(tl_words)
    try:
        from modules.sentiment import INTENSIFIERS, NEGATIONS
        base -= set(NEGATIONS) | {k.lower() for k in INTENSIFIERS}
    except Exception:
        base -= {"hindi", "huwag", "wag", "wala", "not", "never", "no"}
    if custom_extra:
        base |= {w.strip().lower() for w in custom_extra if w.strip()}
    return base


# ---------------------------------------------------------------------------
# Single-text cleaning
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ repeats -> 1 ("graaabe"->"grabe")
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")
WORD_RE = re.compile(r"[a-z0-9\u00f1]+(?:'[a-z]+)?")  # keeps ñ, n't


def collapse_repeats(text: str) -> str:
    """Collapse 3+ repeated letters to one, repeated punctuation to one."""
    text = REPEAT_CHAR_RE.sub(r"\1", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    return text


def clean_single_text(text: str, stopwords: set,
                      map_emojis: bool = True) -> tuple:
    """Clean ONE comment string.

    Returns (clean_text, tokens). Steps mirror the module docstring so the
    student can trace each transformation in the before/after table.
    """
    t = str(text or "")
    t = t.lower()
    t = URL_RE.sub(" ", t)
    if map_emojis:
        for emo, tok in EMOJI_MAP.items():
            if emo in t:
                t = t.replace(emo, tok)
    else:
        # Strip most common emoji ranges if NOT mapping them.
        t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]+",
                   " ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r"\1", t)  # keep the WORD, drop '#'
    t = collapse_repeats(t)
    # Normalize slang token-by-token (word boundaries).
    words = WORD_RE.findall(t)
    words = [get_slang_map().get(w, w) for w in words]
    words = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(words), words


def light_clean(text: str) -> str:
    """Track-B cleaning for transformer/VADER input (context preserved).

    Only strips URLs and @mentions, collapses whitespace, and trims. Keeps
    stopwords, slang spellings, casing-neutralized text and emojis intact so
    contextual models see negations ('hindi'), intensifiers ('sobra') and
    emphasis. Use this column for HF inference, NOT clean_text.
    """
    t = str(text or "")
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" @user ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# DataFrame-level pipeline with step counts
# ---------------------------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame, custom_stopwords=None,
                         map_emojis: bool = True,
                         min_tokens: int = 3) -> tuple:
    """Run the full pipeline; return (df_clean, log, freq_table).

    Args:
        df: raw dataset (must contain a 'text' column).
        custom_stopwords: extra list from the GUI editable box.
        map_emojis: True -> emojis become sentiment tokens; False -> strip.
        min_tokens: drop rows with fewer tokens after cleaning.

    Returns:
        df_clean: copy of df + ['clean_text', 'tokens', 'text_light'].
            text_light is the Track-B input for transformer/VADER (raw text
            with only URLs/mentions stripped). Rows are filtered on Track-A
            quality (clean_text non-empty, >= min_tokens).
        log: [(step, n_rows)] counts for the funnel chart/table.
        freq: DataFrame(word, count) sorted desc for the frequency chart.
    """
    if df is None or df.empty:
        raise ValueError("No data to preprocess — collect or load data first.")
    if "text" not in df.columns:
        raise ValueError("Dataset has no 'text' column.")
    stops = get_default_stopwords(custom_stopwords)
    work = df.copy()
    log = [("0. Input rows", len(work))]
    # Drop empty texts.
    work = work[work["text"].astype(str).str.strip() != ""].copy()
    log.append(("1. After dropping empty texts", len(work)))
    # Clean each row.
    cleaned, tok_lists = [], []
    for txt in work["text"].astype(str):
        c, toks = clean_single_text(txt, stops, map_emojis=map_emojis)
        cleaned.append(c)
        tok_lists.append(toks)
    work["clean_text"] = cleaned
    work["tokens"] = tok_lists
    # Track B: light cleaning on the ORIGINAL text (before filtering), kept
    # aligned so transformer inference never sees stopword-stripped input.
    work["text_light"] = work["text"].astype(str).apply(light_clean)
    work = work[work["clean_text"].str.strip() != ""].copy()
    log.append(("2. After cleaning (non-empty result)", len(work)))
    # De-duplicate on clean_text (catches copy-paste spam across platforms).
    work = work.drop_duplicates(subset=["clean_text"]).copy()
    log.append(("3. After removing duplicates", len(work)))
    # Drop very short rows.
    work["n_tokens"] = work["tokens"].apply(len)
    work = work[work["n_tokens"] >= int(min_tokens)].copy()
    log.append((f"4. After dropping short rows (< {min_tokens} tokens)",
                len(work)))
    if work.empty:
        raise ValueError("All rows were removed by cleaning. Try fewer "
                         "stopwords, a smaller minimum token length, or "
                         "check that your texts are not all URLs/mentions.")
    # Word frequencies over kept tokens.
    counter = Counter()
    for toks in work["tokens"]:
        counter.update(toks)
    freq = pd.DataFrame(counter.most_common(40),
                        columns=["word", "count"])
    return work.reset_index(drop=True), log, freq
