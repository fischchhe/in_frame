# In-Frame: Automatic Frame Classification of News Headlines

This repository contains my part of a group project for the Team Lab (Computational Linguistics, University of Stuttgart, summer semester 2026). The project was originally developed in a private repository under our university GitHub accounts. What you find here is the part I designed and implemented myself: the complete experimental pipeline, from data preparation and soft label construction over the baselines and the perceptron to both BERT models and the evaluation. The exploratory and qualitative analyses were done by my teammate and are not included.

## What the project is about

We classify news headlines from the [Media Frames Corpus](https://github.com/dallascard/media_frames_corpus) (Card et al., 2015) by their dominant framing dimension, i.e. the lens through which an issue is presented (e.g. political, legal, crime-related, policy-focused).

Usually, annotator disagreement is collapsed into a single majority label and thrown away. I wanted to treat it as information instead and see whether a model can learn from it, which led to the question:

> Does incorporating annotator disagreement via soft labels into BERT-based frame classification affect model performance, and what factors account for any observed gains?

- **H1:** Soft-label training improves overall classification performance compared to hard-label training, because it captures the ambiguity of overlapping frames.
- **H2:** The gains are largest on headlines where annotators disagree.

## Data

We use the headline-level frame annotations of the `immigration` and `samesex` topics. The Media Frames Corpus goes beyond headlines: articles are also annotated at the document and span level. The full article texts are not publicly available, though, so we restrict ourselves to the headlines and their annotations, which are publicly available. The corpus itself is **not** included here (see How to Run).

| Setting | Frames | Headlines | Train / Dev / Test |
|---|---|---|---|
| `top4` | Political, Legality, Crime, Policy Prescription | 4,147 | 2,902 / 622 / 623 |
| `full` | all 15 headline frame labels (incl. "Other") | 8,260 | 5,782 / 1,239 / 1,239 |

Before splitting, I remove irrelevant articles, articles without a headline annotation and headlines longer than 30 tokens. Both settings use a stratified 70/15/15 split.

For the hard label I take the majority vote, breaking ties randomly (fixed seed). For the soft label I keep the disagreement and use the normalised distribution of the annotators' votes.

## Models

| Model | Details |
|---|---|
| Dummy baselines | Majority class and uniform random |
| Multiclass perceptron | Unigram, bigram, TF-IDF and combined features; 20 epochs, learning rate 1, best epoch chosen by dev accuracy; 6 runs |
| Hard-BERT | `distilbert-base-uncased` + linear head, cross-entropy on majority labels |
| Soft-BERT | Same architecture, KL-divergence loss against the soft label distributions |

Both BERT models use max. length 48, batch size 8, learning rate 1e-5 and 6 epochs (keeping the best dev epoch), trained on CPU with 6 seeds (0–5). I report mean ± std over the seeds; for per-headline analyses I additionally use the seed with the median dev macro-F1.

## Results

Test macro-F1, mean ± std over 6 seeds:

| Method | Top-4 | Full |
|---|---|---|
| Dummy Majority | 0.229 | 0.036 |
| Dummy Random | 0.245 | 0.050 |
| Perceptron (unigrams) | 0.586 ± 0.017 | 0.383 ± 0.013 |
| Perceptron (combined) | 0.592 ± 0.013 | 0.384 ± 0.017 |
| Hard-BERT | 0.715 ± 0.006 | 0.550 ± 0.008 |
| Soft-BERT | **0.718 ± 0.010** | **0.555 ± 0.011** |

In terms of F1, Hard- and Soft-BERT are not significantly different (paired t-test across seeds). Soft-BERT is, however, closer to how the annotators actually voted: the test KL divergence drops from 1.308 to 1.194 nats in the full setting (p = 0.001) and from 0.651 to 0.614 nats in the top-4 setting (p = 0.053).

## Repository Structure

```
in_frame/
├── src/
│   ├── runner.py                 # Pipeline entry point (set MODE here)
│   ├── prep_data_final.py        # Data preparation, label construction, train/dev/test split
│   ├── test_prep.py              # Sanity checks on the prepared data
│   ├── dummy_method.py           # Majority and random baselines
│   ├── feature_extraction.py     # Unigram, bigram, TF-IDF and combined features
│   ├── soft_labels_create.py     # Soft labels from annotator votes
│   ├── perceptron.py             # Multiclass perceptron
│   ├── bert_train.py             # Hard-BERT
│   ├── soft_bert.py              # Soft-BERT
│   └── evaluation.py             # Metrics, significance tests, plots
├── results/                      # Final result JSONs and figures
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

All commands are run from the repository root.

**1. Install dependencies** (tested with Python 3.12)

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**2. Get the corpus** and place it in the repository root:

```bash
git clone https://github.com/dallascard/media_frames_corpus
```

**3. Choose the setting** by setting `MODE = "top4"` or `MODE = "full"` at the top of `src/runner.py`.

**4. Run the pipeline**

```bash
python src/runner.py
```

This runs data preparation, all models and the evaluation in order. Intermediate data (`train.pkl`, `dev.pkl`, `test.pkl`), prediction pickles, result JSONs and plots are written to the repository root, with `_top4` / `_full` suffixes where they differ between settings. Training logs are written as `hard_bert_<timestamp>.log` and `soft_bert_<timestamp>.log`.

**Runtime:** on my laptop CPU, each BERT model takes about 70 minutes in the top-4 setting and about 2.3 hours in the full setting (6 seeds each). Everything else finishes within a few minutes.

## Reference

Dallas Card, Amber E. Boydstun, Justin H. Gross, Philip Resnik, and Noah A. Smith. 2015. The Media Frames Corpus: Annotations of Frames Across Issues. In *Proceedings of ACL-IJCNLP 2015*, pages 438–444.
