import pandas as pd
import json
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("prep.log"),   
        logging.StreamHandler(),   
    ],
)

MODE = os.environ.get("DATA_MODE", "top4")

''' Content: Prepping the JSONS from the media frames corpus to be used for our approaches
1. Loading the data (mediaframes corpus, immigration + samesex from the second annotation round for the 
    MODE = top4 which was our original apporach. Mode = full includes samesex, immigration and all frames)
2. Cleaning: removing irrelevant articles, keeping only headline-level frame annotations (codes ending in .1)
3. Majority rule to get a single frame label per article i.e. hard labels then Mapping majority frame codes to frame labels via codes.json
4. Extracting soft labels for the advance approach from the annoations 
5. Building a top4 subset (without "Other") next to the full set
6. stratified train/dev/test splits for both top4 and full, saved as pickles
'''

'1.------Loading data-----------------------------------------------------------------'
topics = ["immigration", "samesex"] #using immigration + samesex only in the beginning bc theyre from round 2  more experienced annotators and ~2.2 per article instead of 1.2
# then expanded to whole dataset to really be able to 
dfs = []

for topic in topics:
    with open(f"media_frames_corpus/annotations/{topic}.json") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient="index")
    df["topic"] = topic
    dfs.append(df)

df = pd.concat(dfs)

#look at and log general outline of df 
logging.info(f"Shape: {df.shape}")
logging.info(f"Columns: {df.columns}")
logging.info(f"First annotation: {df['annotations'].iloc[0]}")

'2.------------Cleaning--------------------------------------------------'

# Remove irrelavant: true articles (marked so in the head, not in annotations)
df = df[df["irrelevant"] == False]

#check how many were removed
logging.info(f"Shape after removing irrelevant articles: {df.shape}")

#keep only headline-level annotations of frames (the articles were annotated at the
#document level as well as span level: we do not have access to the ful dataset
#and thus have to restrict ourselves to the headline annotations which are freely
#available)
def extract_headline_annotations(annotations):
    framing = annotations.get("framing", {})
    headline_frames = {}

    for annotator, spans in framing.items():
        headline_span = next((s for s in spans if str(s["code"]).endswith(".1")), None)
        
        if headline_span:
            headline_frames[annotator] = headline_span["code"]

    if not headline_frames:
        return None
    return {"frames": headline_frames}

df["headline_annotations"] = df["annotations"].apply(extract_headline_annotations)

# drop rows that have no headline annotions (i.e. articles where the healdines
# were not annotated)
df = df[df["headline_annotations"].notna()]

#check that we didnt loose an exessive amount of headlines here
logging.info(f"Shape after extracting headline annotations: {df.shape}")

#take care of title length outliers
df["title_length"] = df["title"].str.split().str.len()
df = df[df["title_length"] > 0]
df = df[df["title_length"] <= 30]
logging.info(f"Shape after title-length filter: {df.shape}")

'3.------Creating A Majority Label----------------------------------------------------'
# this becomes the hard label later: one frame gets 1, the rest 0.
# most disagreements are ties, so we break them by picking randomly from the
# top-voted labels instead of always taking annotator nr.1

#for reproducibility
random.seed(42)

def majority_vote(annotations):
    frames = annotations["frames"]
    frame_counts = pd.Series(list(frames.values())).value_counts()
    max_count = frame_counts.max()
    top_labels = frame_counts[frame_counts == max_count].index.tolist()
    return random.choice(top_labels)

df["majority_frame"] = df["headline_annotations"].apply(majority_vote)

n_missing = df["majority_frame"].isna().sum()
logging.info(f"Articles with no majority frame: {n_missing}")  # should be 0

#Map majority frame codes to frame labels'
with open("media_frames_corpus/annotations/codes.json") as f:
    codes = json.load(f)

def get_frame_name(x):
    return codes.get(str(x), "Unknown")

df["majority_frame_name"] = df["majority_frame"].apply(get_frame_name)

logging.info(f"Shape after mapping codes to names: {df.shape}") 

'4.---Extract Soft Labels-----------------------------------------------------------------'
# full label space: all frames that appear, sorted for a stable column order

all_frame_names = sorted(
    codes.get(str(c), "Unknown")
    for ann in df["headline_annotations"]
    for c in ann["frames"].values()
)

all_frame_names = sorted(set(all_frame_names))
full_label_dict = {name: i for i, name in enumerate(all_frame_names)}
NUM_FULL = len(full_label_dict)
print("Full label dict:", full_label_dict)


def build_soft_label_full(headline_annotation):
    counts = np.zeros(NUM_FULL, dtype=float)
    for frame_code in headline_annotation["frames"].values():
        name = codes.get(str(frame_code), "Unknown")
        counts[full_label_dict[name]] += 1
    total = counts.sum()
    if total == 0:
        return None 
    return counts / total

df["soft_label_full"] = df["headline_annotations"].apply(build_soft_label_full)

'5.-------Build top4 df-----------------------------------------------------------------------'

# Drop class "Other headlines" to then assemble a top4 that has top4 without that
# class because it would be in the top4 otherwise. 
df_no_other = df[df["majority_frame_name"] != "Other headline"]
top4_names = df_no_other["majority_frame_name"].value_counts().nlargest(4).index
logging.info(f"Four most common frames: {list(top4_names)}")

top4 = df[df["majority_frame_name"].isin(top4_names)].copy()
logging.info(f"Shape after keeping only top 4 frames: {top4.shape}")

# top4 soft-label view: slice the 4 columns, renormalize, drop no-mass rows
def to_top4_view(soft_full):
    idx = [full_label_dict[n] for n in sorted(top4_names)]
    sliced = np.asarray(soft_full)[idx]
    total = sliced.sum()
    return None if total == 0 else sliced / total

top4["soft_label"] = top4["soft_label_full"].apply(to_top4_view)
n_drop = top4["soft_label"].isna().sum()
if n_drop:
    logging.info(f"dropping {n_drop} top4 rows with no top4 vote mass")
    top4 = top4[top4["soft_label"].notna()].copy()

'6.----------Save the df and build the stratified splits-----------------------'

if MODE == "top4":
    data = top4
else:
    df["soft_label"] = df["soft_label_full"]
    data = df

train, temp = train_test_split(
    data, test_size=0.3, random_state=42,
    stratify=data["majority_frame_name"])
dev, test = train_test_split(
    temp, test_size=0.5, random_state=42,
    stratify=temp["majority_frame_name"])

train.to_pickle("train.pkl")
dev.to_pickle("dev.pkl")
test.to_pickle("test.pkl")
logging.info(f"[{MODE}] Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")