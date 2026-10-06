import pandas as pd
import numpy as np
import sklearn as sklearn
from nltk import bigrams as nltk_bigrams
from nltk.tokenize import word_tokenize
import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# read the splits
train = pd.read_pickle("train.pkl")
dev = pd.read_pickle("dev.pkl")
test = pd.read_pickle("test.pkl")

"""==============UNIGRAMS===================================================================================="""
# extract unigrams from train only to build vocab
# us/USA as the same word might be an issue but we will think about it later
train['unigrams'] = train['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.split()
# build the vocab dict from train only
unigrams_dict = {word: i for i, word in enumerate(train["unigrams"].explode().unique())}

# build the vector for each headline
def headline_to_vector(features, vocab):
    vector = [0] * len(vocab)
    for word in features:
        if word in vocab:
            vector[vocab[word]] = 1
    return vector

# apply unigram extraction and vectorization to all three splits
# dev and test use the same vocab (unigrams_dict) built from train
train['vectors'] = train['unigrams'].apply(lambda x: headline_to_vector(x, unigrams_dict))
dev['unigrams'] = dev['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.split()
dev['vectors'] = dev['unigrams'].apply(lambda x: headline_to_vector(x, unigrams_dict))
test['unigrams'] = test['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.split()
test['vectors'] = test['unigrams'].apply(lambda x: headline_to_vector(x, unigrams_dict))

"""===============BIGRAMS======================================================================="""
#bigram function

def get_bigrams(text):
    # lowercase, strip punctuation, tokenize
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(cleaned)
    return list(nltk_bigrams(tokens))

train['bigrams'] = train['title'].apply(get_bigrams)
dev['bigrams'] = dev['title'].apply(get_bigrams)
test['bigrams'] = test['title'].apply(get_bigrams)
# build the vocab dict from train only
bigrams_dict = {bigram: i for i, bigram in enumerate(train['bigrams'].explode().unique())}
#vectorize
train['bigram_vectors'] = train['bigrams'].apply(lambda x: headline_to_vector(x, bigrams_dict))
dev['bigram_vectors']   = dev['bigrams'].apply(lambda x: headline_to_vector(x, bigrams_dict))
test['bigram_vectors']  = test['bigrams'].apply(lambda x: headline_to_vector(x, bigrams_dict))


#actually combine bigram and unigram because bigram is performing terribly alone
def combine_features(row):
    return row["unigrams"] + row["bigrams"]

train["combined"] = train.apply(combine_features, axis=1)
dev["combined"]   = dev.apply(combine_features, axis=1)
test["combined"]  = test.apply(combine_features, axis=1)

# build vocab
combined_dict = {feat: i for i, feat in enumerate(train["combined"].explode().unique())}

# vectorize
train['combined_vectors'] = train['combined'].apply(lambda x: headline_to_vector(x, combined_dict))
dev['combined_vectors']   = dev['combined'].apply(lambda x: headline_to_vector(x, combined_dict))
test['combined_vectors']  = test['combined'].apply(lambda x: headline_to_vector(x, combined_dict))

"""=============TF-IDF==============================================================================="""
#term frequency
def create_freq_vector(features):
    freq_vector = [0] * len(unigrams_dict)
    for word in features:
        if word in unigrams_dict:
            freq_vector[unigrams_dict[word]] += 1
    return freq_vector

#vectorize
train['tf_vectors'] = train['unigrams'].apply(create_freq_vector)
dev['tf_vectors']   = dev['unigrams'].apply(create_freq_vector)
test['tf_vectors']  = test['unigrams'].apply(create_freq_vector) 

def compute_idf(train_df, vocab):
    N = len(train_df)
    idf = [0] * len(vocab)
    for freq_vector in train_df["tf_vectors"]:
        for idx, count in enumerate(freq_vector):
            if count > 0:
                idf[idx] += 1
    for idx in range(len(idf)):
        idf[idx] = np.log10((N + 1) / (idf[idx] + 1))
    return idf

# compute idf from train only
idf = compute_idf(train, unigrams_dict)

# multiply TF x IDF to get TF-IDF vectors
train['tfidf_vectors'] = train['tf_vectors'].apply(lambda tf: [tf[i] * idf[i] for i in range(len(idf))])
dev['tfidf_vectors']   = dev['tf_vectors'].apply(lambda tf: [tf[i] * idf[i] for i in range(len(idf))])
test['tfidf_vectors']  = test['tf_vectors'].apply(lambda tf: [tf[i] * idf[i] for i in range(len(idf))])

"""========Save em all ============="""
train.to_pickle("train.pkl")
dev.to_pickle("dev.pkl")
test.to_pickle("test.pkl")
