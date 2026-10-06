import pandas as pd
import numpy as np
import os 

MODE = os.environ.get("DATA_MODE", "top4")

#read in the cleaned four_frames splits
train_df = pd.read_pickle("train.pkl")
dev_df   = pd.read_pickle("dev.pkl")
test_df  = pd.read_pickle("test.pkl")

#Majority Frame Dummy Method: For each topic, predict the most common frame in the training set as 
#the predicted frame for all instances of that topic in the test set.
def dummy_method_most_common(train_df, eval_df):
    results = eval_df.copy()
    for topic in train_df["topic"].unique():
        topic_rows = train_df[train_df["topic"] == topic]
        majority_class = topic_rows["majority_frame_name"].value_counts().idxmax()
        results.loc[results["topic"] == topic, "predicted_frame"] = majority_class
    return results

results_dummy_majority_test_df = dummy_method_most_common(train_df, test_df)
results_dummy_majority_dev_df  = dummy_method_most_common(train_df, dev_df)
results_dummy_majority_test_df.to_pickle(f"results_dummy_majority_test_{MODE}.pkl")
results_dummy_majority_dev_df.to_pickle(f"results_dummy_majority_dev_{MODE}.pkl")



# Random Frame Dummy Method
def dummy_method_random(train_df, eval_df):
    results = eval_df.copy()
    all_frames = train_df["majority_frame_name"].unique()
    results["predicted_frame"] = np.random.choice(all_frames, size=len(eval_df))
    return results

results_dummy_random_test_df = dummy_method_random(train_df, test_df)
results_dummy_random_dev_df  = dummy_method_random(train_df, dev_df)
results_dummy_random_test_df.to_pickle(f"results_dummy_random_test_{MODE}.pkl")
results_dummy_random_dev_df.to_pickle(f"results_dummy_random_dev_{MODE}.pkl")

print(f"Test shape: {results_dummy_majority_test_df.shape}")
print(f"Dev shape: {results_dummy_majority_dev_df.shape}")