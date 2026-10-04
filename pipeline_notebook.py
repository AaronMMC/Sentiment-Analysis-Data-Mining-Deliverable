"""
pipeline_notebook.py — Notebook-style single-file version of the CORE pipeline.

Course requirement: ".ipynb-compatible script version of the core pipeline
(a single notebook-style .py with # %% cells)".

HOW TO USE:
  Option A (VS Code): open this file, click "Run Cell" on each `# %%` block
      (with the Jupyter extension installed) — it behaves like a notebook.
  Option B (Jupyter):  jupytext --to ipynb pipeline_notebook.py
      (pip install jupytext) then open the .ipynb in Jupyter/VS Code.
  Option C (plain):   py pipeline_notebook.py   (runs end-to-end on demo data)

Everything here REUSES the same modules as the Streamlit app so results
match the GUI. No numbers are invented: change INPUT_CSV to your real
exported dataset to reproduce your project results.
"""

# %% [markdown]
# # Impeachment Discourse Pipeline (notebook version)
# Steps: load → preprocess (Taglish) → similarity → topics (LDA) → sentiment → evaluation.
# Replace INPUT_CSV with your real `impeachment_dataset.csv` export from the app.

# %% Imports
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import DEFAULT_THRESHOLDS as TH
from modules import data_collection as DC
from modules import preprocessing as PP
from modules import similarity as SIM
from modules import topics as TOP
from modules import sentiment as SEN
from modules import evaluation as EV

INPUT_CSV = "data/demo_sample.csv"   # <-- point at your real export for the defense
print("Python pipeline — thresholds:", {k: TH[k] for k in
      ["records_red_max", "coherence_yellow_max", "acc_yellow_max"]})

# %% 1. Load data
df_raw = DC.load_demo_dataframe(INPUT_CSV)
print(f"Loaded {len(df_raw)} rows from {INPUT_CSV}")
print(df_raw["platform"].value_counts())
print(df_raw["manual_sentiment"].value_counts(dropna=False))

# %% 2. Preprocess (Taglish)
df_clean, log, freq = PP.preprocess_dataframe(df_raw, custom_stopwords=[],
                                              map_emojis=True, min_tokens=3)
for step, n in log:
    print(f"{step}: {n}")
print("\nBefore/after (first 5):")
print(df_clean[["text", "clean_text"]].head())
print("\nTop 15 words:")
print(freq.head(15))

# %% 3. Similarity (TF-IDF cosine + Jaccard by platform)
groups, vec, mat, cos_df = SIM.tfidf_cosine_matrix(df_clean, "platform")
jac_df = SIM.jaccard_matrix(df_clean, "platform")
print("Cosine:\n", cos_df)
print("\nJaccard:\n", jac_df)
shared, distinctive = SIM.top_shared_and_distinctive(df_clean, "platform")
print("\nShared words:\n", shared.head(10))
for g, d in distinctive.items():
    print(f"\nDistinctive [{g}]:\n", d.head(8))
print("\nHelper notes:")
for b in SIM.similarity_helper_bullets(cos_df, jac_df, TH):
    print("-", b)

# %% 4. Topic modeling (LDA k-search 2..10, then best-k model)
token_lists = TOP._get_tokens(df_clean)
k_table = TOP.search_k(token_lists, 2, min(10, len(token_lists)))
print(k_table)
best_k = int(k_table.loc[k_table["coherence_cv"].idxmax(), "k"])
print(f"Best k = {best_k}")
model, dictionary, corpus = TOP.train_lda(token_lists, num_topics=best_k)
mets = TOP.lda_metrics(model, token_lists, dictionary, corpus)
print("Coherence Cv:", round(mets["coherence_cv"], 4),
      "| Perplexity:", mets["perplexity"])
print(TOP.top_words_per_topic(model))
assign = TOP.assign_dominant_topics(model, corpus)
print(TOP.topic_share_per_platform(df_clean, assign))
print("\nHelper notes:")
for b in TOP.topic_helper_bullets(k_table, TH):
    print("-", b)

# %% 5. Sentiment (VADER + custom lexicon)
lex = dict(SEN.DEFAULT_LEXICON)
sdf = SEN.classify_dataframe(df_clean, lex, pos_thresh=0.15, neg_thresh=-0.15)
print(sdf["predicted"].value_counts())
print("Overall net:", SEN.net_sentiment(sdf["predicted"]))
print("\nNet by platform:\n", SEN.net_by_group(sdf, "platform"))
print("\nHelper notes:")
for b in SEN.sentiment_helper_bullets(sdf, TH):
    print("-", b)

# %% 6. Validation: holdout → 10-fold CV → final eval
lab = EV.labeled_only(sdf)
dev, hold = EV.make_holdout(lab, holdout_frac=TH["holdout_frac"],
                            random_state=TH["random_state"])
print(f"dev={len(dev)} holdout={len(hold)}")


def classify_fn(fold_df):
    return SEN.classify_dataframe(
        fold_df, lex, pos_thresh=0.15, neg_thresh=-0.15)["predicted"].tolist()


cv_df = EV.cross_validate(dev, classify_fn, n_splits=TH["cv_folds"])
print(cv_df)
print(EV.cv_summary(cv_df))
final = EV.final_metrics(hold["manual_sentiment"].tolist(),
                         hold["predicted"].tolist())
print("Accuracy:", final["accuracy"], "| Macro-F1:", final["macro_f1"])
print(final["report_df"])
print("\nConfusion matrix (rows=gold, cols=pred [Pos, Neg, Neu]):\n",
      final["confusion_matrix"])
print("\nErrors:\n", EV.error_table(hold).head())
print("\nHelper notes:")
for b in EV.evaluation_helper_bullets(final, EV.cv_summary(cv_df), cv_df,
                                      len(hold), mets["coherence_cv"], TH):
    print("-", b)

# %% Done
print("\nDone. To reproduce with real data, set INPUT_CSV to your exported "
      "impeachment_dataset.csv and re-run all cells.")
