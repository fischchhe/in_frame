import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import torch.nn.functional as F
import random
import logging
from datetime import datetime
import os

MODE = os.environ.get("DATA_MODE", "top4")


# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cpu")  # using distilbert rn so macbook can deal
print(f"Using device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────
# same exact data and data split for perceptron, hard BERT, and soft BERT
train_df = pd.read_pickle("train.pkl")
dev_df   = pd.read_pickle("dev.pkl")
test_df  = pd.read_pickle("test.pkl")

# ── Labels: same dicts as perceptron.py ─────────────────────────────────
label_dict         = {key: value for value, key in enumerate(sorted(train_df["majority_frame_name"].unique()))}
reverse_label_dict = {v: k for k, v in label_dict.items()}
label_names        = [reverse_label_dict[i] for i in range(len(label_dict))]
assert len(label_dict) == len(train_df["soft_label"].iloc[0]), "label/soft mismatch"

# ── Tokenizer ─────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#tok_lens = [len(tokenizer.encode(t)) for t in pd.concat([train_df, dev_df, test_df])["title"]]
#print(f"longest headline by token: {max(tok_lens)}")

MAX_LEN   = 48 #longest healdine after tokenization in data is 46

# ── Dataset ───────────────────────────────────────────────────────────────────────────
# create a class out of the data to be able to use dataset in torch (dataloaders)
# carries soft_label too (NOT used for training here; hard BERT trains on the hard
# majority label but needed so we can compute KL against the soft annotator
# distribution at eval time, symmetric with soft BERT)
class HeadlineDataset(Dataset):
    def __init__(self, df, label_dict, tokenizer, max_len):
        self.texts       = df["title"].tolist()
        self.labels      = df["majority_frame_name"].map(label_dict).tolist()
        self.soft_labels = [torch.tensor(s, dtype=torch.float32)
                            for s in df["soft_label"]]
        self.tok         = tokenizer
        self.max_len     = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
            "soft_label":     self.soft_labels[idx],
        }


# ── DataLoaders ───────────────────────────────────────────────────────────────
BATCH_SIZE = 8  # on cpu, and have time so lets keep it small to avoid memory issues

train_loader = DataLoader(HeadlineDataset(train_df, label_dict, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(HeadlineDataset(dev_df,   label_dict, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(HeadlineDataset(test_df,  label_dict, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=False)


# ── Evaluate (F1 against the hard majority label) ──────────────────────────────
def evaluate(model, loader, split_name=""):
    model.eval()  # no dropout etc, just evaluation
    all_preds, all_labels, all_probs = [], [], []  # store predictions, labels,
    # and probabilities (for KL divergence, so its comparable with soft bert)
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1)
            probs          = torch.softmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    if split_name:
        print(f"\n=== {split_name} ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Macro F1 : {macro_f1:.4f}")
        print(classification_report(all_labels, all_preds,
                                    target_names=label_names, zero_division=0))
    return acc, macro_f1, all_preds, all_probs


## KL divergence against the SOFT labels (annotator distribution).
# Symmetric with soft BERT: both models are measured by how close their predicted
# distribution sits to the true annotator distribution. Hard BERT is expected to
# do worse here by construction having been trained on the majority label but we need
# to compare with the correct metrics for each 
def compute_dev_kl(model, loader):
    model.eval()
    total_kl = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_labels    = batch["soft_label"].to(device)   # ← soft target, not one-hot
            logits         = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs      = F.log_softmax(logits, dim=1)
            kl             = F.kl_div(log_probs, soft_labels, reduction="batchmean")
            total_kl      += kl.item()
            n             += 1
    return total_kl / n


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_once(seed):
    set_seed(seed)

    # ── Training ──────────────────────────────────────────────────────────────────
    # doing the loop manually and not with Trainer here because I have to do that for soft
    # bert anyway and I want this script to generalize well so i dont have to adapt as much
    N_EPOCHS   = 6 
    LR         = 1e-5  # product of experiments too, just really small seems to work best
    NUM_LABELS = len(label_dict)

    # using distilbert because im running it on my macbook
    # distilbert-base-uncased are the weights
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS
    ).to(device)

    # AdamW standard optimizer according to docs
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # creating vars to keep track of best model state and best macro f1 score
    best_macro_f1 = 0.0
    best_state    = None

    # training loop: train for N_EPOCHS, evaluate on dev after each epoch, save best model state
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:  # getting the batches from train_df via the dataloader
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            optimizer.zero_grad()  # dont want to accumulate gradients across batches
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # apply model (with labels → cross-entropy loss)
            outputs.loss.backward()  # compute gradients of the loss wrt model params
            optimizer.step()         # update model params now
            total_loss += outputs.loss.item()  # update tally on total loss for this epoch

        avg_loss = total_loss / len(train_loader)  # calc average loss for this epoch
        # hard bert selected on macro f1:
        _, macro_f1, _, _ = evaluate(model, dev_loader, split_name=f"Dev (epoch {epoch+1})")
        dev_kl = compute_dev_kl(model, dev_loader)  # KL vs soft labels, monitoring only
        print(f"Avg train loss: {avg_loss:.4f}")
        print(f"Dev KL (vs soft): {dev_kl:.4f}")
        logging.info(f"seed={seed} epoch={epoch+1} train_loss={avg_loss:.4f} dev_f1={macro_f1:.4f} dev_kl={dev_kl:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ↑ New best dev Macro F1: {best_macro_f1:.4f}")

    # ── Reload best (by macro F1) and final eval ────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)
    dev_acc,  dev_f1,  dev_preds,  dev_probs  = evaluate(model, dev_loader,  "FINAL Dev")
    test_acc, test_f1, test_preds, test_probs = evaluate(model, test_loader, "FINAL Test")

    test_kl = compute_dev_kl(model, test_loader)

    # per-topic metrics (test)
    soft   = np.stack(test_df["soft_label"].values)
    probs  = np.array(test_probs)
    kl_row = np.sum(np.where(soft > 0, soft * (np.log(soft + 1e-12) - np.log(probs + 1e-12)), 0.0), axis=1)
    topics = test_df["topic"].values
    preds  = np.array(test_preds)
    gold   = test_df["majority_frame_name"].map(label_dict).values
    per_topic = {}
    for t in np.unique(topics):
        m = topics == t
        per_topic[t] = {
            "test_f1": float(f1_score(gold[m], preds[m], average="macro", zero_division=0)),
            "test_kl": float(kl_row[m].mean()),
        }

    logging.info(f"seed={seed} FINAL dev_f1={dev_f1:.4f} test_f1={test_f1:.4f} test_kl={test_kl:.4f}")

    return {
        "seed": seed, "dev_acc": dev_acc, "dev_f1": dev_f1,
        "test_acc": test_acc, "test_f1": test_f1, "test_kl": test_kl,
        "per_topic": per_topic,
        "dev_preds": dev_preds, "dev_probs": dev_probs,
        "test_preds": test_preds, "test_probs": test_probs,
    }

SEEDS  = [0, 1, 2, 3, 4, 5]
LOGNAME = f"hard_bert_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    handlers=[logging.FileHandler(LOGNAME), logging.StreamHandler()])
logging.info(f"CONFIG model=distilbert-base-uncased MAX_LEN={MAX_LEN} BATCH_SIZE={BATCH_SIZE} "
             f"N_EPOCHS=6 LR=1e-5 NUM_LABELS={len(label_dict)} seeds={SEEDS} n_frames={len(label_dict)}")

results = [run_once(s) for s in SEEDS]

test_f1s = np.array([r["test_f1"] for r in results])
test_kls = np.array([r["test_kl"] for r in results])
logging.info(f"SUMMARY test_f1 mean={test_f1s.mean():.4f} std={test_f1s.std():.4f} | "
             f"test_kl mean={test_kls.mean():.4f} std={test_kls.std():.4f}")
print(f"\n=== SEED SUMMARY ===")
print(f"Test Macro F1: {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")
print(f"Test KL:       {test_kls.mean():.4f} ± {test_kls.std():.4f}")

dev_f1s = np.array([r["dev_f1"] for r in results])
with open(f"bert_results_{MODE}.json", "w") as f:
    json.dump({"BERT_hard": {"dev_f1": list(map(float, dev_f1s)),
                             "test_f1": list(map(float, test_f1s)),
                             "test_kl": list(map(float, test_kls)),
                             "per_topic": [r["per_topic"] for r in results]}}, f)   # ← added

pd.to_pickle(results, f"bert_hard_all_seeds_{MODE}.pkl")

# pick the median-F1 seed as the representative run whose predictions feed evaluation.py
median_idx = int(np.argsort(dev_f1s)[len(dev_f1s) // 2])
rep = results[median_idx]
logging.info(f"REPRESENTATIVE seed={rep['seed']} dev_f1={rep['dev_f1']:.4f} (median) -> saving pickles")
print(f"Representative run: seed={rep['seed']} (median dev F1)")

# ── Save predictions (same format as perceptron.py) ─────────────────────────────
dev_out  = dev_df.copy()
test_out = test_df.copy()
dev_out["predicted_frame"]  = [reverse_label_dict[p] for p in rep["dev_preds"]]
test_out["predicted_frame"] = [reverse_label_dict[p] for p in rep["test_preds"]]
dev_out["predicted_probs"]  = list(rep["dev_probs"])
test_out["predicted_probs"] = list(rep["test_probs"])
dev_out.to_pickle(f"results_bert_hard_dev_{MODE}.pkl")
test_out.to_pickle(f"results_bert_hard_test_{MODE}.pkl")
