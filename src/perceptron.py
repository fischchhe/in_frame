import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score
import os

MODE = os.environ.get("DATA_MODE", "top4")

'''
classes are frames:
1. Political
2. Legality, Constitutionality and Jurisprudence
3. Crime and Punishment
4. Policy Prescription
features: unigrams, bigrams, (tfidf)
'''
FEATURE_COL = {
    "unigrams": "vectors",
    "bigrams":  "bigram_vectors",
    "tfidf": "tfidf_vectors",
    "combined": "combined_vectors"
}
N_RUNS = 6

#load the data
train_df = pd.read_pickle("train.pkl")
dev_df   = pd.read_pickle("dev.pkl")
test_df  = pd.read_pickle("test.pkl")

"""==================Labels because they need to go from string to int (0-3) and back later================================"""
label_dict = {key: value for value, key in enumerate(sorted(train_df["majority_frame_name"].unique()))}
reverse_label_dict = {v: k for k, v in label_dict.items()}
y_train = train_df["majority_frame_name"].map(label_dict).values
y_dev   = dev_df["majority_frame_name"].map(label_dict).values
y_test  = test_df["majority_frame_name"].map(label_dict).values

"""=================Perceptron Functions========================================================================="""
def predict(X, W, b):
    return np.argmax(X @ W.T + b, axis=1)

def evaluate(X, y, W, b):
    return np.mean(y == predict(X, W, b))

def evaluate_f1(X, y, W, b):
    return f1_score(y, predict(X, W, b), average="macro", zero_division=0)

def train(X_train, y_train, X_dev, y_dev, W, b, learning_rate, n_epochs):
    best_acc, best_W, best_b = 0, W.copy(), b.copy()
    for epoch in range(n_epochs):
        indices = np.random.permutation(len(X_train))
        for i in indices:
            x = X_train[i]
            y_true = y_train[i]
            scores = W @ x + b
            y_pred = np.argmax(scores)
            if y_pred != y_true:
                W[y_true] += learning_rate * x
                W[y_pred] -= learning_rate * x
                b[y_true] += learning_rate
                b[y_pred] -= learning_rate
        acc = evaluate(X_dev, y_dev, W, b)
        print(f"Epoch {epoch+1}: dev accuracy = {acc:.3f}")
        if acc > best_acc:
            best_acc, best_W, best_b = acc, W.copy(), b.copy()
    print(f"Best dev accuracy: {best_acc:.3f}")
    return best_W, best_b

"""===========Run the perceptron for each feature set==================================================================="""
all_results = {FEATURE: {"dev": [], "test": [], "dev_f1": [], "test_f1": []} for FEATURE in FEATURE_COL}


for FEATURE, col in FEATURE_COL.items():
    print(f"\n=== Running perceptron with {FEATURE} ===")
    X_train = np.vstack(train_df[col].values)
    X_dev   = np.vstack(dev_df[col].values)
    X_test  = np.vstack(test_df[col].values)
    NUM_CLASSES = len(label_dict)
    for run in range(N_RUNS):
        print(f"\n-- Run {run+1}/{N_RUNS} --")
        np.random.seed(run)
        W = np.zeros((NUM_CLASSES, X_train.shape[1]))
        b = np.zeros(NUM_CLASSES)
        W, b = train(X_train, y_train, X_dev, y_dev, W, b, learning_rate=1, n_epochs=20)
        all_results[FEATURE]["dev"].append(evaluate(X_dev, y_dev, W, b))
        all_results[FEATURE]["test"].append(evaluate(X_test, y_test, W, b))
        all_results[FEATURE]["dev_f1"].append(evaluate_f1(X_dev, y_dev, W, b))
        all_results[FEATURE]["test_f1"].append(evaluate_f1(X_test, y_test, W, b))



    # save the predictions of the last run for classification report
    test_df["predicted_frame"] = [reverse_label_dict[i] for i in predict(X_test, W, b)]
    dev_df["predicted_frame"]  = [reverse_label_dict[i] for i in predict(X_dev, W, b)]
    test_df.to_pickle(f"results_perceptron_{FEATURE}_test_{MODE}.pkl")
    dev_df.to_pickle(f"results_perceptron_{FEATURE}_dev_{MODE}.pkl")

with open(f"all_results_{MODE}.json", "w") as f:
    json.dump(all_results, f)
"""==== Print mean ,std etc================================"""
print("\n\n=== SUMMARY ===")
for FEATURE in FEATURE_COL:
    dev_mean  = np.mean(all_results[FEATURE]["dev"])
    dev_std   = np.std(all_results[FEATURE]["dev"])
    test_mean = np.mean(all_results[FEATURE]["test"])
    test_std  = np.std(all_results[FEATURE]["test"])
    print(f"{FEATURE}: dev={dev_mean:.3f}±{dev_std:.3f}  test={test_mean:.3f}±{test_std:.3f}")