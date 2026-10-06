import json
import numpy as np
import pandas as pd



# ── Load splits ───────────────────────────────────────────────────────────────
train_df = pd.read_pickle("train.pkl")
dev_df   = pd.read_pickle("dev.pkl")
test_df  = pd.read_pickle("test.pkl")

# ── Load codes.json to map frame codes (e.g. 3.1) → frame names ──────────────
with open("media_frames_corpus/annotations/codes.json") as f:
    codes = json.load(f)

# ── Use the exact same label_dict as perceptron.py and bert_train.py ──────────
label_dict         = {key: value for value, key in enumerate(sorted(train_df["majority_frame_name"].unique()))}
reverse_label_dict = {v: k for k, v in label_dict.items()}
print("Label dict:", label_dict)
NUM_CLASSES = len(label_dict)
top4_names  = set(label_dict.keys())


def build_soft_label(headline_annotation):
    """
    For each article, count annotator votes that fall within our top-4 frames,
    then normalise to a probability distribution.

    If ALL annotators voted outside the top-4 (very rare after our filtering),
    return None so the row gets dropped.
    """
    frame_votes = headline_annotation["frames"]   # {annotator_id: frame_code}

    counts = np.zeros(NUM_CLASSES, dtype=float)
    for frame_code in frame_votes.values():
        frame_name = codes.get(str(frame_code), None)
        if frame_name in top4_names:
            counts[label_dict[frame_name]] += 1

    total = counts.sum()
    if total == 0:
        return None          # all votes were outside top-4; row will be dropped
    return counts / total    # normalise → probability distribution


def make_soft_labels(df):
    soft = df["headline_annotations"].apply(build_soft_label)

    n_missing = soft.isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} rows had zero top-4 votes — dropping them")
        mask = soft.notna()
        df   = df[mask].copy()
        soft = soft[mask]

    df = df.copy()
    df["soft_label"] = list(soft)
    return df


print("\nBuilding soft labels for train...")
train_df = make_soft_labels(train_df)
print(f"  Train rows: {len(train_df)}")

print("Building soft labels for dev...")
dev_df = make_soft_labels(dev_df)
print(f"  Dev rows: {len(dev_df)}")

print("Building soft labels for test...")
test_df = make_soft_labels(test_df)
print(f"  Test rows: {len(test_df)}")


# ── Sanity checks ─────────────────────────────────────────────────────────────
print("\n── Sanity checks ──")

# 1. All soft labels sum to 1
for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
    sums = np.array([s.sum() for s in df["soft_label"]])
    assert np.allclose(sums, 1.0), f"{name}: soft labels don't sum to 1!"
    print(f"  {name}: all soft labels sum to 1 ✓")

# 2. Corrected check: rows where ALL annotators voted for the same top-4 frame
#    → soft label must be one-hot matching majority_frame_name.
#    (The original `disagree` column was computed on ALL 15 frames before filtering,
#     so we recompute "top-4 unanimous" here.)
def is_top4_unanimous(row):
    votes = [codes.get(str(c), None) for c in row["headline_annotations"]["frames"].values()]
    top4_votes = [v for v in votes if v in top4_names]
    return len(set(top4_votes)) == 1 and len(top4_votes) > 0

top4_unanimous = train_df[train_df.apply(is_top4_unanimous, axis=1)]
mismatches = 0
for _, row in top4_unanimous.iterrows():
    hard_idx    = label_dict[row["majority_frame_name"]]
    soft_argmax = int(np.argmax(row["soft_label"]))
    if hard_idx != soft_argmax:
        mismatches += 1

print(f"  Top-4-unanimous rows: {len(top4_unanimous)}")
print(f"  Mismatches (soft argmax ≠ hard label): {mismatches}  ← should be 0")

# 3. Show examples where annotators actually disagreed on top-4 frames
#    (these are the interesting cases for the paper)
print("\n── Example soft labels where annotators split across top-4 frames ──")
def has_top4_split(row):
    votes = [codes.get(str(c), None) for c in row["headline_annotations"]["frames"].values()]
    top4_votes = [v for v in votes if v in top4_names]
    return len(set(top4_votes)) > 1

split_examples = train_df[train_df.apply(has_top4_split, axis=1)].head(5)
label_names = [reverse_label_dict[i] for i in range(NUM_CLASSES)]
for _, row in split_examples.iterrows():
    dist_str = ", ".join(f"{label_names[i][:12]}: {row['soft_label'][i]:.2f}"
                         for i in range(NUM_CLASSES) if row["soft_label"][i] > 0)
    print(f"  Title: {row['title'][:65]}")
    print(f"  Hard label: {row['majority_frame_name']}")
    print(f"  Soft label: {dist_str}\n")

print(f"Total examples with genuine top-4 split: {split_examples.shape[0]} shown / "
      f"{train_df[train_df.apply(has_top4_split, axis=1)].shape[0]} in train")


# ── Save ──────────────────────────────────────────────────────────────────────
train_df.to_pickle("train.pkl")
dev_df.to_pickle("dev.pkl")
test_df.to_pickle("test.pkl")

print("\nSaved soft_label column to train.pkl, dev.pkl, test.pkl ✓")
print(f"Vector order: {label_names}")