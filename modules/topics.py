"""
topics.py — Topic modeling with LDA (gensim), optional LSA and BERTopic.

For the defense, know these one-liners:
  - LDA: each comment is a MIX of topics; each topic is a MIX of words.
    It learns those mixes unsupervised (no labels needed).
  - Coherence (Cv): do the top words of a topic actually co-occur in real
    comments? Higher = more human-interpretable. We maximize this.
  - Perplexity: how "surprised" the model is by held-out words. Lower =
    better fit, but it often disagrees with human interpretability, so we
    show both and let Coherence pick the winner.
  - LSA: a cheaper linear-algebra alternative (TF-IDF + SVD). Fast, but
    topics are usually less interpretable than LDA.
  - BERTopic: transformer embeddings + clustering. Best quality on big
    data, but heavy to install — hence OPTIONAL here.

All scores are computed from the student's real cleaned tokens.
"""

import numpy as np
import pandas as pd

MIN_DOCS_FOR_TOPICS = 5


def lda_backend() -> str:
    """Which LDA engine is available: 'gensim' or 'sklearn-fallback'.

    gensim has no prebuilt wheel for very new Pythons (e.g. 3.14 needs a C++
    compiler), so when it cannot be imported we fall back to scikit-learn's
    LatentDirichletAllocation with the SAME function signatures. Coherence is
    then mean-NPMI over top words (relative scale — compare k values against
    each other, NOT against the C_V badge cut-offs).
    """
    try:
        import gensim  # noqa: F401
        return "gensim"
    except Exception:
        return "sklearn-fallback"


def _get_tokens(df: pd.DataFrame) -> list:
    """Pull the token lists produced by preprocessing."""
    if df is None or df.empty:
        raise ValueError("No cleaned data — run preprocessing first.")
    if "tokens" not in df.columns:
        raise ValueError("Missing 'tokens' column — re-run preprocessing.")
    toks = [list(t) if isinstance(t, (list, tuple)) else str(t).split()
            for t in df["tokens"]]
    toks = [[w for w in d if w.strip()] for d in toks]
    if len(toks) < MIN_DOCS_FOR_TOPICS:
        raise ValueError(f"Need at least {MIN_DOCS_FOR_TOPICS} documents "
                         f"for topic modeling (have {len(toks)}).")
    return toks


def build_gensim_corpus(token_lists: list):
    """Build gensim Dictionary + BoW corpus from token lists."""
    try:
        from gensim.corpora import Dictionary
    except ImportError as e:
        raise ImportError("gensim is not installed. Run: "
                          "py -m pip install -r requirements.txt") from e
    dictionary = Dictionary(token_lists)
    # Filter extremes: drop words in <2 docs or >80% of docs (noise control).
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    if len(dictionary) < 10:
        # Too aggressive for tiny demo sets — keep everything.
        dictionary = Dictionary(token_lists)
    corpus = [dictionary.doc2bow(d) for d in token_lists]
    return dictionary, corpus


def train_lda(token_lists: list, num_topics: int = 4,
              random_state: int = 42, passes: int = 15):
    """Train one LDA model; return (model, dictionary, corpus).

    Uses gensim when installed, else scikit-learn fallback (same interface:
    model.num_topics, model.show_topic(), model.get_document_topics(),
    model.log_perplexity()).
    """
    try:
        from gensim.models import LdaModel
    except ImportError:
        return _train_lda_sklearn(token_lists, num_topics, random_state)
    dictionary, corpus = build_gensim_corpus(token_lists)
    model = LdaModel(corpus=corpus, id2word=dictionary,
                     num_topics=int(num_topics), random_state=random_state,
                     passes=int(passes), alpha="auto", eta="auto")
    return model, dictionary, corpus


class _SklearnLDAModel:
    """Duck-typed LDA model backed by sklearn (gensim-free environments)."""

    backend = "sklearn-fallback"

    def __init__(self, lda, vectorizer, doc_topic, X):
        self._lda = lda
        self._vectorizer = vectorizer
        self._doc_topic = doc_topic
        self._X = X
        self.num_topics = int(lda.n_components)

    def show_topic(self, tid: int, topn: int = 10):
        vocab = self._vectorizer.get_feature_names_out()
        comp = self._lda.components_[int(tid)]
        # Normalize to a word distribution for comparable weights.
        total = comp.sum()
        top_idx = comp.argsort()[::-1][: int(topn)]
        return [(vocab[i], float(comp[i] / total if total else 0.0))
                for i in top_idx]

    def get_document_topics(self, bow, minimum_probability=0):
        # In the sklearn backend, `corpus` items are integer doc row ids.
        dist = self._doc_topic[int(bow)]
        return [(tid, float(p)) for tid, p in enumerate(dist)
                if p >= minimum_probability]

    def log_perplexity(self, corpus):
        # sklearn perplexity() is already the classic word-level perplexity
        # (lower = better). Return its negative log2 so lda_metrics'
        # exp2(-x) conversion reproduces the classic scale, matching the
        # gensim path's output convention.
        import math
        perp = float(self._lda.perplexity(self._X))
        if not np.isfinite(perp) or perp <= 0:
            return float("nan")
        return -math.log(perp, 2)


def _train_lda_sklearn(token_lists: list, num_topics: int = 4,
                       random_state: int = 42):
    """sklearn LDA with gensim-style filtering (min_df=2, max_df=0.8)."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    docs = [" ".join(d) for d in token_lists]
    vectorizer = CountVectorizer(min_df=2, max_df=0.8)
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError:
        # Too aggressive for tiny sets — keep everything (mirrors gensim path).
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(docs)
    k = max(2, min(int(num_topics), X.shape[0], X.shape[1]))
    lda = LatentDirichletAllocation(n_components=k, random_state=random_state,
                                    learning_method="batch",
                                    max_iter=15, evaluate_every=-1)
    doc_topic = lda.fit_transform(X)
    if hasattr(doc_topic, "toarray"):
        doc_topic = doc_topic.toarray()
    model = _SklearnLDAModel(lda, vectorizer, doc_topic, X)
    corpus = list(range(X.shape[0]))  # doc row ids (see get_document_topics)
    return model, vectorizer, corpus


def _npmi_coherence(token_lists: list, top_words: list) -> float:
    """Mean normalized-PMI over all word pairs in a topic's top words.

    Document-level co-occurrence on the project's own tokens. Range
    roughly [-1, 1]; higher = words actually appear together. This is the
    gensim-free fallback coherence (NOT C_V — do not compare to C_V badges).
    """
    import math
    n_docs = len(token_lists)
    if n_docs == 0 or len(top_words) < 2:
        return float("nan")
    doc_sets = [set(d) for d in token_lists]
    df = {}
    for s in doc_sets:
        for w in s:
            df[w] = df.get(w, 0) + 1
    scores = []
    for i in range(len(top_words)):
        for j in range(i + 1, len(top_words)):
            a, b = top_words[i], top_words[j]
            fa, fb = df.get(a, 0), df.get(b, 0)
            fab = sum(1 for s in doc_sets if a in s and b in s)
            if fa == 0 or fb == 0 or fab == 0:
                continue
            pa, pb, pab = fa / n_docs, fb / n_docs, fab / n_docs
            pmi = math.log(pab / (pa * pb))
            npmi = pmi / (-math.log(pab))
            scores.append(max(-1.0, min(1.0, npmi)))
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def lda_metrics(model, token_lists: list, dictionary, corpus) -> dict:
    """Compute Coherence and Perplexity for a model.

    gensim backend: Coherence C_V + perplexity from the likelihood bound.
    sklearn-fallback backend: mean-NPMI coherence (key 'coherence_cv' holds
    it for table compatibility; 'coherence_metric' says which one it is).
    """
    use_gensim = getattr(model, "backend", "gensim") == "gensim"
    if use_gensim:
        try:
            from gensim.models.coherencemodel import CoherenceModel
            cm = CoherenceModel(model=model, texts=token_lists,
                                dictionary=dictionary, coherence="c_v")
            coherence = float(cm.get_coherence())
            metric = "c_v (gensim)"
        except ImportError:
            use_gensim = False
    if not use_gensim:
        tops = []
        for tid in range(model.num_topics):
            tops.append([w for w, _ in model.show_topic(tid, topn=10)])
        vals = [_npmi_coherence(token_lists, t) for t in tops]
        vals = [v for v in vals if np.isfinite(v)]
        coherence = float(sum(vals) / len(vals)) if vals else float("nan")
        metric = "mean-NPMI fallback (no gensim; relative scale only)"
    # gensim log_perplexity returns negative log-likelihood bound; convert
    # with exp2 to classic perplexity. Lower = better fit.
    try:
        perplexity = float(np.exp2(-model.log_perplexity(corpus)))
        if not np.isfinite(perplexity):
            perplexity = float("nan")
    except Exception:
        perplexity = float("nan")
    return {"coherence_cv": coherence, "perplexity": perplexity,
            "coherence_metric": metric}


def search_k(token_lists: list, k_min: int = 2, k_max: int = 10,
             random_state: int = 42) -> pd.DataFrame:
    """Train LDA for each k in [k_min, k_max]; return metrics table.

    The row with the highest Coherence is the recommended k (highlighted
    in the app). k_max is capped by the number of documents.
    """
    k_max = max(k_min, min(int(k_max), len(token_lists)))
    rows = []
    metric_name = None
    for k in range(int(k_min), k_max + 1):
        model, dictionary, corpus = train_lda(token_lists, num_topics=k,
                                              random_state=random_state)
        m = lda_metrics(model, token_lists, dictionary, corpus)
        metric_name = m.get("coherence_metric", "c_v (gensim)")
        rows.append({"k": k, "coherence_cv": round(m["coherence_cv"], 4),
                     "perplexity": (round(m["perplexity"], 3)
                                    if np.isfinite(m["perplexity"])
                                    else None)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["is_best"] = out["coherence_cv"] == out["coherence_cv"].max()
    out.attrs["coherence_metric"] = metric_name or "c_v (gensim)"
    out.attrs["lda_backend"] = lda_backend()
    return out


def top_words_per_topic(model, num_words: int = 10) -> pd.DataFrame:
    """Return a tidy table: one row per (topic, rank) with word + weight."""
    rows = []
    for tid in range(model.num_topics):
        for rank, (word, weight) in enumerate(
                model.show_topic(tid, topn=int(num_words)), 1):
            rows.append({"topic_id": tid, "rank": rank, "word": word,
                         "weight": round(float(weight), 4)})
    return pd.DataFrame(rows)


def assign_dominant_topics(model, corpus) -> list:
    """Assign each document its highest-probability topic + that prob."""
    out = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0)
        tid, prob = max(dist, key=lambda x: x[1])
        out.append((int(tid), float(prob)))
    return out


def representative_docs(df: pd.DataFrame, assignments: list,
                        text_col: str = "text", top_n: int = 3) -> dict:
    """Pick the top-N highest-probability comments per topic as examples."""
    tmp = df.copy()
    tmp["_topic"] = [t for t, _ in assignments]
    tmp["_prob"] = [p for _, p in assignments]
    reps = {}
    for tid in sorted(tmp["_topic"].unique()):
        sub = tmp[tmp["_topic"] == tid].sort_values(
            "_prob", ascending=False).head(top_n)
        reps[int(tid)] = sub[[c for c in
                              ["id", "platform", "political_bloc",
                               text_col, "_prob"] if c in sub.columns]]
    return reps


def topic_share_per_platform(df: pd.DataFrame, assignments: list) -> pd.DataFrame:
    """Share of each topic within each platform (row-normalized %)."""
    tmp = df.copy()
    tmp["_topic"] = [f"T{t}" for t, _ in assignments]
    ct = pd.crosstab(tmp["platform"], tmp["_topic"], normalize="index") * 100
    return ct.round(1)


# --- Optional: LSA toggle -------------------------------------------------
def train_lsa(token_lists: list, num_topics: int = 4):
    """LSA via TF-IDF + TruncatedSVD. Returns (terms_per_topic_df, doc_topic).

    LSA is deterministic and fast — a good classroom contrast to LDA's
    probabilistic topics. Quality is usually lower on short comments.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(d) for d in token_lists]
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(docs)
    k = max(2, min(int(num_topics), X.shape[0] - 1, X.shape[1]))
    svd = TruncatedSVD(n_components=k, random_state=42)
    doc_topic = svd.fit_transform(X)
    vocab = vec.get_feature_names_out()
    rows = []
    for tid in range(k):
        comp = svd.components_[tid]
        top_idx = comp.argsort()[::-1][:10]
        for rank, i in enumerate(top_idx, 1):
            rows.append({"topic_id": tid, "rank": rank,
                         "word": vocab[i],
                         "weight": round(float(comp[i]), 4)})
    return pd.DataFrame(rows), doc_topic


# --- Optional: BERTopic (only if installed) -------------------------------
def bertopic_available() -> bool:
    """True if bertopic can be imported (it is optional)."""
    try:
        import bertopic  # noqa: F401
        return True
    except Exception:
        return False


def topic_helper_bullets(k_table: pd.DataFrame, th: dict) -> list:
    """3-5 plain-English observations from the REAL k-search table."""
    if k_table is None or k_table.empty:
        return ["Run the k-search first to get coherence numbers."]
    bullets = []
    fallback = "fallback" in str(k_table.attrs.get("coherence_metric", ""))
    best = k_table.loc[k_table["coherence_cv"].idxmax()]
    cv = float(best["coherence_cv"])
    if fallback:
        bullets.append(
            f"Best k = **{int(best['k'])}** with fallback coherence "
            f"**{cv:.3f}** (mean-NPMI, no gensim here). Compare k values "
            f"against EACH OTHER only — the 0.35/0.50 C_V badges do not apply "
            f"to this scale. Also check Perplexity (lower = better fit).")
        perps = k_table["perplexity"].dropna()
        if len(perps) >= 2:
            bullets.append(
                f"Perplexity ranges {perps.min():.0f}–{perps.max():.0f} "
                f"(classic scale). Big drops then flattening = elbow k.")
        bullets.append(
            "Top-word check matters MORE than the number here: if topics are "
            "full of function words (si, yung, yan, nyo, po), add them as "
            "custom stopwords on page 3 and re-run.")
        bullets.append(
            "Next step: read the representative documents per topic. If two topics share "
            "the same top words AND the same example comments, reduce k by one.")
        return bullets[:5]
    if cv < th["coherence_red_max"]:
        level = "red — topics may be hard to interpret"
    elif cv <= th["coherence_yellow_max"]:
        level = "yellow — usable but check top words carefully"
    else:
        level = "green — topics hang together well"
    bullets.append(
        f"Best k = **{int(best['k'])}** with Coherence Cv **{cv:.3f}** ({level}). "
        f"Name each topic from its top words, not from this number alone.")
    if len(k_table) >= 3:
        bullets.append(
            "If coherence jumps up then flattens, the 'elbow' k is often the most "
            "defensible choice — a flat line means extra topics add little meaning.")
    perps = k_table["perplexity"].dropna()
    if len(perps) >= 2 and perps.iloc[-1] < perps.iloc[0]:
        bullets.append(
            "Perplexity falls as k grows (normal — bigger models fit training words "
            "better). Trust Coherence over Perplexity when they disagree, because "
            "Coherence matches human judgment better.")
    if cv < th["coherence_red_max"]:
        bullets.append(
            "Coherence is low: try cleaning harder (more stopwords), merging tiny "
            "groups, or collecting more records before concluding topics are 'bad'.")
    bullets.append(
        "Next step: read the representative documents per topic. If two topics share "
        "the same top words AND the same example comments, reduce k by one.")
    return bullets[:5]
