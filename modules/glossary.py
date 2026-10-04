"""
glossary.py — Persistent glossary + sidebar search data.

Each entry has: plain definition, simple example, formula (where relevant),
and "how to read it in this project". The app renders these in the Glossary
tab and injects (?) hover tooltips next to technical terms on every page.

For the defense: if you can explain each entry's "how to read it" line in
your own words, you can answer most panel questions about methods.
"""

GLOSSARY = {
    "Accuracy": {
        "definition": "Share of all predictions the classifier got right.",
        "example": "90 correct out of 100 comments = 0.90 accuracy.",
        "formula": "Accuracy = (TP + TN) / (all predictions)",
        "how_to_read": "High accuracy with imbalanced classes can mislead — always check per-class recall too (Page 7).",
    },
    "Precision": {
        "definition": "Of everything the classifier CALLED positive, how much really was positive. Punishes false alarms.",
        "example": "Called 10 comments positive, 8 truly were → precision 0.80.",
        "formula": "Precision = TP / (TP + FP)",
        "how_to_read": "Low precision for Negative means many neutral comments are mislabeled negative — your negative lexicon is too trigger-happy.",
    },
    "Recall": {
        "definition": "Of everything TRULY positive, how much did the classifier catch. Punishes misses.",
        "example": "10 truly negative comments, caught 6 → recall 0.60.",
        "formula": "Recall = TP / (TP + FN)",
        "how_to_read": "Low Negative recall = the classifier misses many negative comments; add negative words to the lexicon.",
    },
    "F1-Score": {
        "definition": "Harmonic mean of precision and recall — a single balance number per class.",
        "example": "P=0.8, R=0.6 → F1 ≈ 0.69.",
        "formula": "F1 = 2·P·R / (P + R)",
        "how_to_read": "Compare F1 across classes on Page 7 to find your weakest sentiment.",
    },
    "Macro vs Weighted average": {
        "definition": "Macro = plain mean over classes (rare classes count equally). Weighted = mean weighted by class size.",
        "example": "Macro-F1 0.60 but weighted 0.75 → the small class is weak.",
        "formula": "macro = mean(F1 per class); weighted = Σ F1·support / total",
        "how_to_read": "For this project, MACRO-F1 is the headline: it stops a big Neutral class from hiding a weak Negative class.",
    },
    "Confusion Matrix (TP/FP/TN/FN)": {
        "definition": "Table of true label (rows) vs predicted label (columns). TP=true positives, FP=false alarms, FN=misses, TN=true negatives.",
        "example": "Row Negative, column Neutral = 12 → 12 negative comments mislabeled neutral.",
        "formula": "rows = gold label, columns = prediction",
        "how_to_read": "Read Page 7's heatmap row by row: each row shows where that true class 'leaks' to.",
    },
    "Train/Test Split": {
        "definition": "Splitting labeled data so the model is tuned on one part and honestly tested on another part it never saw.",
        "example": "90% dev for tuning, 10% holdout for the final grade.",
        "formula": "—",
        "how_to_read": "Our split happens BEFORE tuning (Page 7) so threshold choices can't peek at the final test.",
    },
    "Holdout/Validation Set": {
        "definition": "The locked 10% slice used ONCE at the very end. Its score is your honest final grade.",
        "example": "20 of 200 comments locked away, evaluated once.",
        "formula": "—",
        "how_to_read": "If you re-tune after seeing holdout errors, the grade is no longer honest — the app locks the split to prevent this.",
    },
    "Cross-Validation (10-fold)": {
        "definition": "Split the dev set into 10 parts; train-tune on 9, test on 1, rotate. Average the 10 scores; SD shows stability.",
        "example": "Mean macro-F1 0.68 ± 0.04.",
        "formula": "report mean ± SD across folds",
        "how_to_read": "High SD (> 0.05) means results swing with the split — collect more data, especially for rare classes.",
    },
    "Stratified sampling": {
        "definition": "Splitting that preserves each class's share in every fold/split, so rare sentiments aren't lost.",
        "example": "30% negative overall → ~30% negative in each fold.",
        "formula": "—",
        "how_to_read": "Both our holdout and CV are stratified — required because sentiment classes are rarely balanced.",
    },
    "Data Leakage": {
        "definition": "When test information sneaks into training/tuning (e.g., tuning thresholds on the holdout), inflating scores dishonestly.",
        "example": "Picking lexicon words from holdout errors, then reporting holdout accuracy.",
        "formula": "—",
        "how_to_read": "Our protocol (split first, lock holdout, tune on dev only) exists to prevent exactly this.",
    },
    "Overfitting": {
        "definition": "Tuning so hard to the dev set that general quality drops — CV looks good, holdout looks bad.",
        "example": "CV 0.85 but holdout 0.65 → gap 0.20, flagged.",
        "formula": "flag if (CV − holdout) > 0.10",
        "how_to_read": "Page 7's overfit badge computes this gap automatically from your real scores.",
    },
    "Cosine Similarity": {
        "definition": "Angle between two TF-IDF word vectors: 0 = nothing alike, 1 = same direction. Ignores group size differences.",
        "example": "YouTube vs Reddit = 0.45 → moderate overlap.",
        "formula": "cos(A,B) = A·B / (||A||·||B||)",
        "how_to_read": "Low (< 0.3) = camps use different language; high (> 0.6) = they argue in the same terms.",
    },
    "Jaccard Similarity": {
        "definition": "Word-set overlap: shared words divided by all words used by either group. Stricter than cosine.",
        "example": "40 shared of 200 total unique = 0.20 → moderate.",
        "formula": "J(A,B) = |A ∩ B| / |A ∪ B|",
        "how_to_read": "Use alongside cosine on Page 4; Jaccard < 0.1 means almost no shared vocabulary.",
    },
    "TF-IDF": {
        "definition": "Term Frequency × Inverse Document Frequency: up-weights words frequent in one group but rare overall.",
        "example": "'confidential' in many YouTube comments but few Reddit ones → high TF-IDF for YouTube.",
        "formula": "tfidf(t,d) = tf(t,d) × log(N / df(t))",
        "how_to_read": "Page 4's vectors and distinctive-terms tables both come from these weights.",
    },
    "Tokenization": {
        "definition": "Splitting text into units (tokens/words) for counting. Our regex keeps letters, digits, ñ and n't.",
        "example": "'Wag magpadala!' → ['wag', 'magpadala']",
        "formula": "—",
        "how_to_read": "Check the before/after table on Page 3 — merged or split Filipino words show up there.",
    },
    "Stopwords": {
        "definition": "Ultra-common function words ('ang', 'the', 'sa') removed before modeling because they carry no topic signal.",
        "example": "Removing 'ang/ng/sa/at' leaves content words like 'impeachment'.",
        "formula": "—",
        "how_to_read": "The editable stopword box on Page 3 lets you remove rally slogans that flood every topic.",
    },
    "Lemmatization/Stemming": {
        "definition": "Reducing word variants to a base form ('running'→'run'). We use light normalization (slang map + repeats) instead of full lemmatization for Taglish transparency.",
        "example": "'graaabe' → 'grabe' via repeat-collapsing.",
        "formula": "—",
        "how_to_read": "Mention in your defense why full English lemmatizers underperform on Taglish.",
    },
    "Taglish/Code-switching": {
        "definition": "Mixing Tagalog and English in one sentence — the normal register of PH social media.",
        "example": "'Accountability is not political persecution, basahin muna.'",
        "formula": "—",
        "how_to_read": "Code-switching is why we need BOTH English and Filipino stopwords plus a custom lexicon.",
    },
    "Lexicon": {
        "definition": "A word list with sentiment scores (e.g., 'kurakot' → −1.8) that the rule-based classifier looks up.",
        "example": "See the editable table on Page 6.",
        "formula": "score range [−2, +2] per word",
        "how_to_read": "Every lexicon edit re-runs classification — cite which words you added and why in your defense.",
    },
    "VADER": {
        "definition": "Valence Aware Dictionary and sEntiment Reasoner: a fast English-tuned rule-based scorer (−1..+1) handling punctuation, caps, emojis.",
        "example": "'This is AMAZING!!!' → strongly positive.",
        "formula": "compound in [−1, +1]",
        "how_to_read": "Our baseline on Page 6; weak on Tagalog, which motivates the custom lexicon blend.",
    },
    "Negation": {
        "definition": "Words like 'hindi/not/never' that flip the next sentiment word ('hindi corrupt' ≠ corrupt).",
        "example": "'hindi maganda' → negative despite 'maganda'.",
        "formula": "flip sign of next sentiment word within 3 tokens",
        "how_to_read": "The score-explanation column shows (negated) tags so you can audit flips.",
    },
    "Net Sentiment": {
        "definition": "Positive% minus Negative% per group, from −100 (all negative) to +100 (all positive).",
        "example": "60% pos − 25% neg = +35.",
        "formula": "net = %positive − %negative",
        "how_to_read": "Bars on Page 6; values inside ±10 sit in the neutral zone.",
    },
    "Topic Modeling": {
        "definition": "Unsupervised discovery of themes: groups of words that co-occur across comments.",
        "example": "Topic A: funds, confidential, coa, audit → 'funds controversy'.",
        "formula": "—",
        "how_to_read": "You NAME the topics from top words + representative docs (Page 5) — the algorithm only proposes them.",
    },
    "LDA": {
        "definition": "Latent Dirichlet Allocation: each comment mixes topics; each topic mixes words. Learned unsupervised.",
        "example": "Comment = 70% Topic 1 + 30% Topic 3.",
        "formula": "P(word|doc) = Σ P(word|topic)·P(topic|doc)",
        "how_to_read": "Our default model (Page 5); needs ≥5 docs and benefits from clean stopwords.",
    },
    "BERTopic": {
        "definition": "Modern pipeline: transformer embeddings → clustering → topic words. Best on large data; heavy install, hence optional here.",
        "example": "—",
        "formula": "—",
        "how_to_read": "Mention as future work if your dataset grows past ~1k comments.",
    },
    "LSA": {
        "definition": "Latent Semantic Analysis: TF-IDF matrix compressed with SVD. Fast deterministic alternative to LDA.",
        "example": "Toggle on Page 5 to compare topic words vs LDA.",
        "formula": "X ≈ U·Σ·Vᵀ (truncated SVD)",
        "how_to_read": "Expect less interpretable topics on short comments — say so in your analysis.",
    },
    "Coherence Score (Cv)": {
        "definition": "Do a topic's top words actually co-occur in real comments? Higher = more interpretable. We maximize it to pick k.",
        "example": "k=4 → Cv 0.52 (green) beats k=7 → 0.38.",
        "formula": "Cv via sliding-window NPMI + cosine (gensim)",
        "how_to_read": "> 0.50 green, 0.35–0.50 yellow, < 0.35 red (Page 5).",
    },
    "Perplexity": {
        "definition": "How 'surprised' the topic model is by held-out words. Lower = better statistical fit, but often disagrees with humans.",
        "example": "Falls as k grows even when topics get mushy.",
        "formula": "perplexity = exp2(−log-likelihood per word)",
        "how_to_read": "Shown next to Coherence; trust Coherence when they disagree.",
    },
    "Cohen's Kappa": {
        "definition": "Agreement between two annotators beyond chance: < 0.4 poor, 0.4–0.6 fair, > 0.6 good.",
        "example": "Two labelers agree on 80/100 with chance 50% → κ ≈ 0.6.",
        "formula": "κ = (Po − Pe) / (1 − Pe)",
        "how_to_read": "Optional second-annotator mode on Page 2 computes this from real double-labels.",
    },
    "Class Imbalance": {
        "definition": "When one sentiment dominates (e.g., 70% neutral), metrics and models skew toward it.",
        "example": "Warn if any class < 15% of data.",
        "formula": "share = class count / total",
        "how_to_read": "Check the balance badge on Page 1/2 before trusting accuracy.",
    },
    "Gold Label": {
        "definition": "The trusted human label (manual_sentiment) that evaluation treats as ground truth.",
        "example": "You reading a comment and marking it Negative.",
        "formula": "—",
        "how_to_read": "Quality in = quality out: sloppy gold labels cap every metric on Page 7.",
    },
    "Annotation": {
        "definition": "The human labeling process itself (sentiment + political bloc per comment).",
        "example": "One-comment-at-a-time buttons on Page 2.",
        "formula": "—",
        "how_to_read": "Follow the labeling guide + stance-vs-tone caution; log disagreements for your appendix.",
    },
}


def search_glossary(query: str) -> dict:
    """Case-insensitive substring search over glossary terms."""
    q = (query or "").strip().lower()
    if not q:
        return GLOSSARY
    return {k: v for k, v in GLOSSARY.items() if q in k.lower()}
