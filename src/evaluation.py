import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import spearmanr
from itertools import combinations
import os 

MODE = os.environ.get("DATA_MODE", "top4")
def pkl(stem):  return f"{stem}_{MODE}.pkl"
def jsn(stem):  return f"{stem}_{MODE}.json"


#eval function
def evaluate(results_df, label=""):
    gold  = results_df["majority_frame_name"]
    predictions = results_df["predicted_frame"]
    print(f"\n=== {label} ===")
    print("Accuracy:", accuracy_score(gold, predictions))
    print("Macro F1:", f1_score(gold, predictions, average="macro", zero_division=0))
    print(classification_report(gold, predictions, zero_division=0))

#kl divergence between the soft labels (annotator distribution) and the saved predicted_probs
#only the bert models have predicted_probs so kl only applies to them
#in nats (natural log), same as the training scripts
def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    p = np.clip(p, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

#per-row kls (keep the individual values so we can do mean/median/histogram/correlations later)
def compute_kls(results_df):
    return [kl_divergence(row["soft_label"], row["predicted_probs"])
            for _, row in results_df.iterrows()]

#mean kl over all rows, needs soft_label and predicted_probs columns
#so "mean" here = averaged over the test headlines, for one run
def mean_kl(results_df, label=""):
    mean = float(np.mean(compute_kls(results_df)))
    if label:
        print(f"Mean KL (vs soft labels, per-headline mean, nats) [{label}]: {mean:.4f}")
    return mean

#entropy of the soft label per row, keep it continuous so we can split however later
def soft_entropy(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    return float(-np.sum(p * np.log(p)))

"""============Dummies + Perceptron=========================="""

#Majority Dummy eval
results_dummy_majority_dev_df  = pd.read_pickle(pkl("results_dummy_majority_dev"))
results_dummy_majority_test_df = pd.read_pickle(pkl("results_dummy_majority_test"))
evaluate(results_dummy_majority_dev_df,  "Dummy Majority method Dev")
evaluate(results_dummy_majority_test_df, "Dummy Majority method Test")


#Random Frame Dummy eval 
results_dummy_random_dev_df  = pd.read_pickle(pkl("results_dummy_random_dev"))
results_dummy_random_test_df = pd.read_pickle(pkl("results_dummy_random_test"))
evaluate(results_dummy_random_dev_df,  "Dummy Random method Dev")
evaluate(results_dummy_random_test_df, "Dummy Random method Test")

#Perceptron on Unigrams eval 
results_perceptron_unigram_dev_df = pd.read_pickle(pkl("results_perceptron_unigrams_dev"))
evaluate(results_perceptron_unigram_dev_df, "Perceptron on Unigrams Dev Results")
results_perceptron_unigram_test_df = pd.read_pickle(pkl("results_perceptron_unigrams_test"))
evaluate(results_perceptron_unigram_test_df, "Perceptron on Unigrams Test Results")

#bigram
results_perceptron_bigram_dev_df = pd.read_pickle(pkl("results_perceptron_bigrams_dev"))
evaluate(results_perceptron_bigram_dev_df, "Perceptron on Birgam Dev Results")
results_perceptron_bigram_test_df = pd.read_pickle(pkl("results_perceptron_bigrams_test"))
evaluate(results_perceptron_bigram_test_df, "Perceptrpn on Bigram Test Results")


#tf-idf dev and test
results_perceptron_tfidf_dev_df = pd.read_pickle(pkl("results_perceptron_tfidf_dev"))
evaluate(results_perceptron_tfidf_dev_df, "Perceptron on TF-IDF Dev Results")
results_perceptron_tfidf_test_df = pd.read_pickle(pkl("results_perceptron_tfidf_test"))
evaluate(results_perceptron_tfidf_test_df, "Perceptron on TF-IDF Test Results")

#combined dev and test
results_perceptron_combined_dev_df = pd.read_pickle(pkl("results_perceptron_combined_dev"))
evaluate(results_perceptron_combined_dev_df, "Perceptron on Unigram/Bigram combined Dev Results")
results_perceptron_combined_test_df = pd.read_pickle(pkl("results_perceptron_combined_test"))
evaluate(results_perceptron_combined_test_df, "Perceptron on Unigram/Bigram Combined Results")

"""============Big Comparison Table (still called Final Table in Result Log)==========================""" 
def get_f1(pkl_path):
    df = pd.read_pickle(pkl_path)
    return f1_score(df["majority_frame_name"], df["predicted_frame"], average="macro", zero_division=0)

with open(jsn("all_results")) as f:
    all_results = json.load(f)
with open(jsn("bert_results")) as f:
    bert_hard_runs = json.load(f)["BERT_hard"]
with open(jsn("bert_results_soft")) as f:
    bert_soft_runs = json.load(f)["BERT_soft"]

ordered_results = {
    "Dummy Majority": {"dev_f1": [get_f1(pkl("results_dummy_majority_dev"))],
                   "test_f1": [get_f1(pkl("results_dummy_majority_test"))]},
    "Dummy Random":   {"dev_f1": [get_f1(pkl("results_dummy_random_dev"))],
                   "test_f1": [get_f1(pkl("results_dummy_random_test"))]},
    "P: Unigrams":    all_results["unigrams"],
    "P: Bigrams":     all_results["bigrams"],
    "P: TF-IDF":      all_results["tfidf"],
    "P: Combined":    all_results["combined"],
    "BERT hard":      bert_hard_runs,
    "BERT soft":      bert_soft_runs,
}

rows = []
for method, scores in ordered_results.items():
    rows.append({
        "Method": method,
        "Dev":    f"{np.mean(scores['dev_f1']):.3f} ± {np.std(scores['dev_f1']):.3f}",
        "Test":   f"{np.mean(scores['test_f1']):.3f} ± {np.std(scores['test_f1']):.3f}",
    })
df_table = pd.DataFrame(rows)
print("\n\n=== FINAL TABLE ===")
print(df_table.to_string(index=False))
print("\n--- LaTeX ---\n")
print("\\begin{tabular}{lcc}")
print("\\hline")
print("Method & Dev & Test \\\\")
print("\\hline")
for _, row in df_table.iterrows():
    print(f"{row['Method']} & {row['Dev']} & {row['Test']} \\\\")
print("\\hline")
print("\\end{tabular}")


from scipy.stats import ttest_rel
from itertools import combinations

# ttest for all pairs of perceptron models on the test set 
with open(jsn("all_results")) as f:
    all_results = json.load(f)

feature_names = {
    "unigrams":  "Unigrams",
    "bigrams":  "Bigrams",
    "tfidf":    "TF-IDF",
    "combined": "Combined",
}

print("\n=== PAIRWISE PAIRED T-TESTS (Test set) ===")
print(f"{'Model A':<20} {'Model B':<20} {'t-stat':>8} {'p-value':>10} {'Sig':>6}")
print("-" * 68)

keys = list(feature_names.keys())
for a, b in combinations(keys, 2):
    scores_a = all_results[a]["test"]
    scores_b = all_results[b]["test"]
    t_stat, p_val = ttest_rel(scores_a, scores_b)
    sig = "*" if p_val < 0.05 else ""
    print(f"{feature_names[a]:<20} {feature_names[b]:<20} {t_stat:>8.3f} {p_val:>10.4f} {sig:>6}")

#______BERT (hard + soft)_______________________________________________________________

#hard bert dev and test (F1 vs hard label, KL vs soft label)
results_bert_hard_dev_df  = pd.read_pickle(pkl("results_bert_hard_dev"))
results_bert_hard_test_df = pd.read_pickle(pkl("results_bert_hard_test"))
evaluate(results_bert_hard_dev_df,  "Hard BERT Dev Results")
mean_kl(results_bert_hard_dev_df,   "Hard BERT Dev")
evaluate(results_bert_hard_test_df, "Hard BERT Test Results")
mean_kl(results_bert_hard_test_df,  "Hard BERT Test")

#soft bert dev and test (F1 vs hard label, KL vs soft label)
results_bert_soft_dev_df  = pd.read_pickle(pkl("results_bert_soft_dev"))
results_bert_soft_test_df = pd.read_pickle(pkl("results_bert_soft_test"))
evaluate(results_bert_soft_dev_df,  "Soft BERT Dev Results")
mean_kl(results_bert_soft_dev_df,   "Soft BERT Dev")
evaluate(results_bert_soft_test_df, "Soft BERT Test Results")
mean_kl(results_bert_soft_test_df,  "Soft BERT Test")

#stable label order, same convention as prep_data_final.py and bert_train.py:
#index i of soft_label / predicted_probs  <->  sorted(frame names)[i]
#(has to live here, after the bert test df is loaded)
FRAMES = sorted(results_bert_hard_test_df["majority_frame_name"].unique())

# sanity: soft labels, predicted probs, and FRAMES must share the same (sorted) label space
_n_soft  = len(results_bert_hard_test_df["soft_label"].iloc[0])
_n_probs = len(results_bert_hard_test_df["predicted_probs"].iloc[0])
assert len(FRAMES) == _n_soft == _n_probs, \
    f"label space mismatch: {len(FRAMES)} frames vs soft_label dim {_n_soft} vs probs dim {_n_probs}"

# ========================================================================================
# H1: Does BERT improve classification (F1 vs hard) and distribution quality (KL vs soft)?
# ========================================================================================

# attach per-row kl and entropy to the bert test frames so we can reuse them below
#(both bert frames share the same rows/order, so columns line up across them)
results_bert_hard_test_df["kl"]      = compute_kls(results_bert_hard_test_df)
results_bert_soft_test_df["kl"]      = compute_kls(results_bert_soft_test_df)
results_bert_hard_test_df["entropy"] = results_bert_hard_test_df["soft_label"].apply(soft_entropy)
results_bert_soft_test_df["entropy"] = results_bert_soft_test_df["soft_label"].apply(soft_entropy)

#H1 table: F1 (vs hard label) and mean KL (vs soft labels) for both BERT models on test
#representative run only, mean kl = averaged over the 628 test headlines, in nats
#soft bert is expected to have lower KL by construction, bit circular(?)
print("\n\n=== H1: BERT F1 (vs hard) and KL (vs soft, per-headline mean, nats), Test, representative run ===")
h1_rows = []
for name, df in [("BERT hard", results_bert_hard_test_df),
                 ("BERT soft", results_bert_soft_test_df)]:
    h1_rows.append({
        "Model":    name,
        "Macro F1": f1_score(df["majority_frame_name"], df["predicted_frame"],
                             average="macro", zero_division=0),
        "Mean KL":  df["kl"].mean(),
    })
h1_table = pd.DataFrame(h1_rows)
print(h1_table.to_string(index=False))

with open(jsn("bert_results")) as f:
    bert_hard_runs = json.load(f)["BERT_hard"]
with open(jsn("bert_results_soft")) as f:
    bert_soft_runs = json.load(f)["BERT_soft"]

#each seed first gets its per-headline mean kl, then mean and std over the 6 seeds
print("\n=== H1 across seeds: BERT F1 (vs hard) and KL (vs soft), Test, mean ± std over 6 seeds, nats ===")
bert_seed_rows = []
for name, runs in [("BERT hard", bert_hard_runs), ("BERT soft", bert_soft_runs)]:
    f1s = np.array(runs["test_f1"])
    kls = np.array(runs["test_kl"])
    bert_seed_rows.append({
        "Model":    name,
        "Test F1":  f"{f1s.mean():.3f} ± {f1s.std():.3f}",
        "Test KL":  f"{kls.mean():.3f} ± {kls.std():.3f}",
    })
print(pd.DataFrame(bert_seed_rows).to_string(index=False))

print("\n=== HARD vs SOFT BERT: PAIRED T-TESTS (across seeds, Test set) ===")
print(f"{'Metric':<12} {'t-stat':>8} {'p-value':>10} {'Sig':>6}")
print("-" * 38)
for metric in ["test_f1", "test_kl"]:
    a = bert_hard_runs[metric]
    b = bert_soft_runs[metric]
    t_stat, p_val = ttest_rel(a, b)
    sig = "*" if p_val < 0.05 else ""
    print(f"{metric:<12} {t_stat:>8.3f} {p_val:>10.4f} {sig:>6}")

#per topic table, uses the per_topic lists the training scripts now write into the jsons
#so this is mean and std over the 6 seeds per topic, not just the representative run
print("\n=== PER TOPIC across seeds: F1 and KL, Test (mean ± std over 6 seeds, KL in nats) ===")
pt_rows = []
for name, runs in [("BERT hard", bert_hard_runs), ("BERT soft", bert_soft_runs)]:
    for topic in runs["per_topic"][0]:
        f1s = np.array([s[topic]["test_f1"] for s in runs["per_topic"]])
        kls = np.array([s[topic]["test_kl"] for s in runs["per_topic"]])
        pt_rows.append({
            "Model": name,
            "Topic": topic,
            "Test F1": f"{f1s.mean():.3f} ± {f1s.std():.3f}",
            "Test KL": f"{kls.mean():.3f} ± {kls.std():.3f}",
        })
pt_table = pd.DataFrame(pt_rows)
print(pt_table.to_string(index=False))
print("\n--- LaTeX ---\n")
print(pt_table.to_latex(index=False))

#-------H2: per-headline delta, split by annotator disagreement-------------------------"""

#generic bucket report: across headlines + entropy buckets + NEW frame buckets + NEW domain buckets
#each bucket gets n, mean/median delta and a paired t-test hard_kl vs soft_kl
def h2_bucket_report(df, label):
    """
    df needs columns: hard_kl, soft_kl, delta, entropy, majority_frame_name, topic
    """
    def line(name, sub):
        if len(sub) < 2:
            print(f"{name}: n={len(sub)} (too small for t-test)")
            return
        t, p = ttest_rel(sub["hard_kl"], sub["soft_kl"])
        print(f"{name}: n={len(sub)}  mean delta={sub['delta'].mean():.4f}  "
              f"median={sub['delta'].median():.4f}  t={t:.3f} p={p:.4f}"
              + ("  *" if p < 0.05 else ""))

    print(f"\n\n=== H2 BUCKETS [{label}] (delta = hard_kl - soft_kl, nats, >0: soft closer) ===")

    # --- one across headlines ---
    print("\n--- across all headlines ---")
    line("all", df)

    # --- entropy buckets (agreed vs disagreed, 1e-6 because 0 is sometimes not exactly 0
    #     due to clipping because of log(0)) ---
    print("\n--- by disagreement bucket ---")
    line("agreed (entropy==0)",           df[df["entropy"] < 1e-6])
    line("high disagreement (entropy>0)", df[df["entropy"] >= 1e-6])

    # --- buckets across frames ---
    print("\n--- by frame ---")
    for frame in FRAMES:
        line(frame, df[df["majority_frame_name"] == frame])

    # --- buckets across domains ---
    print("\n--- by domain ---")
    for topic, sub in df.groupby("topic"):
        line(topic, sub)


#bucketing by entropy (annotator disagreement) and computing delta (hard_kl - soft_kl)
h2 = pd.DataFrame({#building pandas dataframe with the relevant cols for H2 analysis
    "title":                results_bert_hard_test_df["title"].values,
    "majority_frame_name":  results_bert_hard_test_df["majority_frame_name"].values,
    "topic":                results_bert_hard_test_df["topic"].values,
    "entropy":              results_bert_hard_test_df["entropy"].values,
    "hard_kl":              results_bert_hard_test_df["kl"].values,
    "soft_kl":              results_bert_soft_test_df["kl"].values,
})


h2["delta"] = h2["hard_kl"] - h2["soft_kl"] #computing delta (hard_kl - soft_kl) per headline, 
# >0 means soft is closer to annotators

#correlation between entropy and delta (does soft help more where annotators disagreed? 
corr = h2[["entropy", "delta"]].corr().iloc[0, 1]
print(f"\n  corr(entropy, delta) = {corr:.4f}")

#buckets: across headlines + entropy + frames + domains, representative (median-F1 seed) run
h2_bucket_report(h2, "representative run (median seed)")


#scatter plot of delta vs entropy, one point per headline, color by frame 
fig, ax = plt.subplots(figsize=(8, 6))
for frame, sub in h2.groupby("majority_frame_name"):
    ax.scatter(sub["entropy"], sub["delta"], label=frame[:20], alpha=0.6, s=25)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Annotator disagreement (entropy of soft label)")
ax.set_ylabel("delta = hard_kl - soft_kl in nats  (>0: soft closer)")
ax.set_title(f"Per-headline soft-label advantage vs disagreement\ncorr = {corr:.3f}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"h2_delta_vs_entropy_{MODE}.png", dpi=150, bbox_inches="tight")
print("\nSaved: h2_delta_vs_entropy.png")
plt.close()


#-------H2 again but over all seeds----------------------------------------------------
#the block above only uses the representative run of each model, which is just one seed pair
#and can be a lucky or unlucky draw. here we average the per-headline kl over all 6 seeds per
#model first (using the all_seeds pickles) and then do the same bucket analysis on that.
#same kl_divergence as above so everything in this section stays in the same units

hard_all_seeds = pd.read_pickle(pkl("bert_hard_all_seeds"))
soft_all_seeds = pd.read_pickle(pkl("bert_soft_all_seeds"))
soft_label_arr = results_bert_hard_test_df["soft_label"].tolist()

def kl_per_row_for_run(run):
    return np.array([kl_divergence(soft_label_arr[i], run["test_probs"][i])
                     for i in range(len(soft_label_arr))])

hard_kl_all = np.mean([kl_per_row_for_run(r) for r in hard_all_seeds], axis=0)
soft_kl_all = np.mean([kl_per_row_for_run(r) for r in soft_all_seeds], axis=0)

h2_seeds = pd.DataFrame({
    "majority_frame_name": results_bert_hard_test_df["majority_frame_name"].values,
    "topic":               results_bert_hard_test_df["topic"].values,
    "entropy":             results_bert_hard_test_df["entropy"].values,
    "hard_kl":             hard_kl_all,
    "soft_kl":             soft_kl_all,
})
h2_seeds["delta"] = h2_seeds["hard_kl"] - h2_seeds["soft_kl"]

corr_s = h2_seeds[["entropy", "delta"]].corr().iloc[0, 1]
print(f"\n  corr(entropy, delta), across seeds = {corr_s:.4f}")

#same buckets as above but on the seed-averaged kls
h2_bucket_report(h2_seeds, "across seeds (per-headline KL averaged over 6 seeds)")

#corr(entropy, delta) within each domain, since disagreement rates differ between the
#domains and could drive the frame effects
print("\n--- corr(entropy, delta) within each domain (across seeds) ---")
for topic, sub in h2_seeds.groupby("topic"):
    c = sub[["entropy", "delta"]].corr().iloc[0, 1]
    print(f"{topic}: corr={c:.4f} (n={len(sub)})")

# ========================================================================================
# QUANT ERROR ANALYSIS: model confusion (all frames) vs annotator disagreement
# run with DATA_MODE=full so FRAMES is the full 15-label space
# ========================================================================================

SHORT_F = [f.replace(" headline", "") for f in FRAMES]

def plot_model_confusion(df, label):
    cm_ = confusion_matrix(df["majority_frame_name"], df["predicted_frame"],
                           labels=FRAMES, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10) if len(FRAMES) > 4 else (6.5, 5.5))
    mask = np.eye(len(FRAMES), dtype=bool)
    sns.heatmap(cm_, annot=True, fmt=".2f", cmap="Blues", mask=mask,
                xticklabels=SHORT_F, yticklabels=SHORT_F,
                linewidths=0.5, linecolor="white", vmin=0, ax=ax,
                cbar_kws={"label": "Row-normalised rate"})
    for i in range(len(FRAMES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, color="#d9d9d9", lw=0))
    ax.set_title(f"Model confusion ({label}, {MODE})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    fname = f"confusion_model_{label.replace(' ', '_')}_{MODE}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
    return cm_

def annotator_confusion(df):
    """
    Annotator disagreement matrix from headline_annotations: for every headline
    and every unordered pair of annotators, count the label pair symmetrically.
    Row-normalised so it is directly comparable to the model confusion matrix.
    """
    with open("media_frames_corpus/annotations/codes.json") as f:
        codes = json.load(f)
    idx = {f_: i for i, f_ in enumerate(FRAMES)}
    M = np.zeros((len(FRAMES), len(FRAMES)), dtype=float)
    for ann in df["headline_annotations"]:
        names = [codes.get(str(c), "Unknown") for c in ann["frames"].values()]
        names = [n for n in names if n in idx]   # drop labels outside current space
        for a, b in combinations(names, 2):
            M[idx[a], idx[b]] += 1
            M[idx[b], idx[a]] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    M_norm = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(12, 10) if len(FRAMES) > 4 else (6.5, 5.5))
    mask = np.eye(len(FRAMES), dtype=bool)
    sns.heatmap(M_norm, annot=True, fmt=".2f", cmap="Oranges", mask=mask,
                xticklabels=SHORT_F, yticklabels=SHORT_F,
                linewidths=0.5, linecolor="white", vmin=0, ax=ax,
                cbar_kws={"label": "Row-normalised co-occurrence"})
    for i in range(len(FRAMES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, color="#d9d9d9", lw=0))
    ax.set_title(f"Annotator label co-occurrence ({MODE})")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    fname = f"confusion_annotators_{MODE}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
    return M_norm

def compare_confusions(cm_model, cm_annot, label):
    """Is the model confused about the same pairs the annotators disagree on?
    Spearman rank corr over off-diagonal cells + side-by-side top pairs."""
    off = ~np.eye(len(FRAMES), dtype=bool)
    rho, p = spearmanr(cm_model[off], cm_annot[off])
    print(f"\n[{label}] Spearman corr, off-diagonal model confusion vs annotator "
          f"disagreement: rho={rho:.3f} p={p:.4f}")

    def top_pairs(cm_, k=8):
        pairs = [(SHORT_F[i], SHORT_F[j], cm_[i, j])
                 for i in range(len(FRAMES)) for j in range(len(FRAMES)) if i != j]
        return sorted(pairs, key=lambda x: x[2], reverse=True)[:k]

    print(f"\nTop confused pairs, MODEL ({label}):")
    for a, b, v in top_pairs(cm_model):
        print(f"  {a:28s} -> {b:28s} {v:.2%}")
    print("\nTop disagreement pairs, ANNOTATORS:")
    for a, b, v in top_pairs(cm_annot):
        print(f"  {a:28s} <-> {b:28s} {v:.2%}")
    return rho, p

cm_annot = annotator_confusion(results_bert_hard_test_df)
for name, df in [("Perceptron Unigrams", results_perceptron_unigram_test_df),
                 ("BERT hard",           results_bert_hard_test_df),
                 ("BERT soft",           results_bert_soft_test_df)]:
    cm_model = plot_model_confusion(df, name)
    compare_confusions(cm_model, cm_annot, name)