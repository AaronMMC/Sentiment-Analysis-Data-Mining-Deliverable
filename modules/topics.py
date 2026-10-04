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
    """Train one LDA model; return (model, dictionary, corpus)."""
    from gensim.models import LdaModel
    dictionary, corpus = build_gensim_corpus(token_lists)
    model = LdaModel(corpus=corpus, id2word=dictionary,
                     num_topics=int(num_topics), random_state=random_state,
                     passes=int(passes), alpha="auto", eta="auto")
    return model, dictionary, corpus


def lda_metrics(model, token_lists: list, dictionary, corpus) -> dict:
    """Compute Coherence (Cv) and Perplexity (log-perplexity) for a model."""
    from gensim.models.coherencemodel import CoherenceModel
    cm = CoherenceModel(model=model, texts=token_lists,
                        dictionary=dictionary, coherence="c_v")
    coherence = float(cm.get_coherence())
    # gensim log_perplexity returns negative log-likelihood bound; convert
    # with exp2 to classic perplexity. Lower = better fit.
    try:
        perplexity = float(np.exp2(-model.log_perplexity(corpus)))
        if not np.isfinite(perplexity):
            perplexity = float("nan")
    except Exception:
        perplexity = float("nan")
    return {"coherence_cv": coherence, "perplexity": perplexity}


def search_k(token_lists: list, k_min: int = 2, k_max: int = 10,
             random_state: int = 42) -> pd.DataFrame:
    """Train LDA for each k in [k_min, k_max]; return metrics table.

    The row with the highest Coherence is the recommended k (highlighted
    in the app). k_max is capped by the number of documents.
    """
    k_max = max(k_min, min(int(k_max), len(token_lists)))
    rows = []
    for k in range(int(k_min), k_max + 1):
        model, dictionary, corpus = train_lda(token_lists, num_topics=k,
                                              random_state=random_state)
        m = lda_metrics(model, token_lists, dictionary, corpus)
        rows.append({"k": k, "coherence_cv": round(m["coherence_cv"], 4),
                     "perplexity": (round(m["perplexity"], 3)
                                    if np.isfinite(m["perplexity"])
                                    else None)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["is_best"] = out["coherence_cv"] == out["coherence_cv"].max()
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
    best = k_table.loc[k_table["coherence_cv"].idxmax()]
    cv = float(best["coherence_cv"])
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
