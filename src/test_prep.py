import json
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("prep.log"),
        logging.StreamHandler(),
    ],
)

'------Load codes-------------------------------------------------------------------'
with open("media_frames_corpus/annotations/codes.json") as f:
    codes = json.load(f)


# checks one dataset family (either the top4 files or the full ones)
# builds label_dict from that datasets own classes so it works for both 4 and 15 class
def check_dataset(name, train_path, dev_path, test_path):
    logging.info(f"══════ checking {name} ══════")
    train_df = pd.read_pickle(train_path)
    dev_df   = pd.read_pickle(dev_path)
    test_df  = pd.read_pickle(test_path)

    label_dict = {k: v for v, k in enumerate(sorted(train_df["majority_frame_name"].unique()))}
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    class_names = set(label_dict.keys())
    NUM_CLASSES = len(label_dict)
    logging.info(f"  {NUM_CLASSES} classes: {list(label_dict.keys())}")

    '------Check soft labels sum to 1---------------------------------------------------'
    for split_name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        sums = np.array([s.sum() for s in df["soft_label"]])
        assert np.allclose(sums, 1.0), f"{name}/{split_name}: soft labels dont sum to 1!"
        logging.info(f"  {split_name}: all soft labels sum to 1 ✓")

    '------Check unanimous rows are one-hot---------------------------------------------'
    # if every annotator (within this datasets classes) voted the same frame,
    # the soft label should put all its mass on the hard label
    def is_unanimous(row):
        votes = [codes.get(str(c), None) for c in row["headline_annotations"]["frames"].values()]
        in_class = [v for v in votes if v in class_names]
        return len(set(in_class)) == 1 and len(in_class) > 0

    unanimous = train_df[train_df.apply(is_unanimous, axis=1)]
    mismatches = 0
    for _, row in unanimous.iterrows():
        hard_idx    = label_dict[row["majority_frame_name"]]
        soft_argmax = int(np.argmax(row["soft_label"]))
        if hard_idx != soft_argmax:
            mismatches += 1
    logging.info(f"  unanimous rows: {len(unanimous)}, mismatches: {mismatches}  ← should be 0")
    assert mismatches == 0, f"{name}: some unanimous rows arent one-hot!"

    '------Peek at genuine disagreement cases------------------------------------------'
    logging.info("  ── example soft labels where annotators split across classes ──")
    def has_split(row):
        votes = [codes.get(str(c), None) for c in row["headline_annotations"]["frames"].values()]
        in_class = [v for v in votes if v in class_names]
        return len(set(in_class)) > 1

    label_names = [reverse_label_dict[i] for i in range(NUM_CLASSES)]
    split_rows = train_df[train_df.apply(has_split, axis=1)]
    for _, row in split_rows.head(5).iterrows():
        dist = ", ".join(f"{label_names[i][:12]}: {row['soft_label'][i]:.2f}"
                         for i in range(NUM_CLASSES) if row["soft_label"][i] > 0)
        logging.info(f"  title: {row['title'][:65]}")
        logging.info(f"  hard: {row['majority_frame_name']}")
        logging.info(f"  soft: {dist}")
    logging.info(f"  total rows with a genuine split: {len(split_rows)}")


'------Run the checks on both dataset families--------------------------------------'
check_dataset("top4", "train.pkl", "dev.pkl", "test.pkl")
check_dataset("full", "train_full.pkl", "dev_full.pkl", "test_full.pkl")

logging.info("✓ all prep checks passed")