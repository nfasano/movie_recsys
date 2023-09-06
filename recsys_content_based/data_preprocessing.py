"""
Preprocess film script dataset for LDA model, EDA on transformed data

Description: This notebook reads in the cleaned dataset from data_cleaning_and_synthesis.ipynb notebook and returns a bag-of-words representation, X, that will be used for model training. 

The script data is first removed of stop words and, optionally, lemmatization is performed. The corpus is then represented using a bag-of-words model. 
"""

import pickle
import json
from scipy import sparse
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


def get_wordnet_pos(word):
    """
    input: word, a str
    ouput: part of speech for that word, used in lemmatization function
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def lemma(text):
    """
    input: text, a str
    ouput: lemmatized text, a str
    """
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text.split()])


def stem(text):
    """
    input: text, a str
    ouput: stemmed text, a str
    """
    ps = PorterStemmer()
    return " ".join([ps.stem(w) for w in text.split()])


def list_duplicates(word_list):
    """
    description: after lemmatization is performed, find all words
                 in a list that are duplicated and return the duplicates
                 as a dictionary with values of the index where the duplicated word occurs

    input:
            - word_list, a list of lemmatized words
    output:
            - a tuple of (key, locs) where key is the lemmatized word
              and locs are the indice locations of duplicated entries
    """
    tally = defaultdict(list)
    for i, item in enumerate(word_list):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def df_preprocessing(df, word_map, stop_words):
    """
    description: function for preprocesing the entire corpus
                 methods include lowercase, punctuation removal, word mappings, and removing stop words
    input:
            - df, a pd.DataFrame with columns ['script_text']
            - word_map, a dictionary for mapping one word to another (e.g. ok maps to okay)
            - stop_words, a list of strings to be removed from corpus
    output:
            - df_processed, a pd.DataFrame
    """

    # remove punctuation
    df["script_text"] = df["script_text"].str.replace(r"[^\w\s]+", " ", regex=True)

    # make entire corpus lowercase
    df["script_text"] = df["script_text"].str.lower()

    # map specific words
    if len(word_map) > 0:
        for jstr in word_map:
            df["script_text"] = df["script_text"].str.replace(
                rf"\b{jstr}\b", word_map[jstr], regex=True
            )

    # remove initial set of stop words
    if len(stop_words):
        df["script_text"] = df["script_text"].apply(
            [lambda x: " ".join([word for word in x.split() if word not in stop_words])]
        )

    return df


def construct_BoW(df, n_features=10000, max_df=0.8, min_df=1, lemmatization=False):
    """
    inputs:
            - df, a pd.DataFrame with columns ['script_text'] that will be used for Bag of words (BoW) construction
            - n_features, an int, maximum number of columns in BoW matrx, X
            - max_df, max word counts (if int) or doc frequency (if float between 0 and 1)
            - min_df, min word counts (if int) or doc frequency (if float between 0 and 1)
            - lemmatization, a bool, whether or not to apply lemmatization (default is False)
    ouputs:
            - X_out, a sparse matrix of size (num_documents x n_features)
            - word_key_out, a list of strings for the features used to create X_out, sorted by increasing columns of X_out
    """
    if not lemmatization:
        # use scikit-learn's CountVectorizer to create BoW representation
        vectorizer = CountVectorizer(
            max_df=max_df, min_df=min_df, max_features=n_features, stop_words="english"
        )
        X_out = vectorizer.fit_transform(df["script_text"])

        # get word features in increasing item number
        word_key_out = vectorizer.get_feature_names_out()

        return X_out, word_key_out
    else:
        # first create bag of words on corpus then perform lemmatization
        vectorizer = CountVectorizer(max_df=0.7, min_df=5)
        X_csr = vectorizer.fit_transform(df["script_text"])
        X_lil = X_csr.tolil()  # used for fast adjustments to matrix sparsity
        X_csc = X_lil.tocsc()  # used for fast column-wise math operations

        # get vocab and sort it in increasing item number
        vocab = vectorizer.vocabulary_
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=False))
        word_key = list(vocab.keys())
        word_idx = list(vocab.values())

        # perform lemmatization/stemming of word_key list
        word_key_text = " ".join([wk for wk in word_key])
        word_key_text = lemma(word_key_text)
        # word_key_text = stem(word_key_text)

        # find the duplicate words after lemmatization
        dup_all = []
        for dup in sorted(list_duplicates(word_key_text.split())):
            dup_all.append(dup)

        # sum coulmns of X for duplicate words after lemmatization
        words_to_remove = []
        for i, dup in enumerate(dup_all):
            words_to_remove = words_to_remove + dup[1][1:]
            X_lil[:, dup[1][0]] = np.sum(X_csc[:, dup[1][:]], axis=1)

        # construt output matrix and drop the duplicated columns, update word_key
        X_out = X_lil.tocsr()
        words_to_keep = np.delete(word_idx, words_to_remove)
        X_out = X_out[:, words_to_keep]
        word_key = [word_key[jword] for jword in words_to_keep]

        # keep words that only appear in more than df_min documents but less than df_max documents
        X_bin = X_out.copy()
        X_bin.data = np.ones(shape=X_bin.data.shape)

        min_df = min_df * len(df) if type(min_df) == float else min_df
        max_df = max_df * len(df) if type(max_df) == float else max_df

        word_idx = np.intersect1d(
            np.argwhere(np.sum(X_bin, axis=0) > min_df)[:, 1],
            np.argwhere(np.sum(X_bin, axis=0) < max_df)[:, 1],
        )
        X_out = X_out[:, word_idx]
        word_key = [word_key[jword] for jword in word_idx]

        # # drop rows that have no words
        # non_blank_movie_scripts = np.argwhere(np.sum(Xbin, axis=1) > 0)[:, 0]
        # X_out = X_out[non_blank_movie_scripts, :]

        # finally retain only top n_features, update word_key accordingly
        wordcount_ordered = np.flip(
            np.argsort(np.array(np.sum(X_out, axis=0)).reshape(-1))
        )
        X_out = X_out[:, wordcount_ordered[0:n_features]]
        word_key = [word_key[jword] for jword in wordcount_ordered[0:n_features]]

        # return lemmatized word_key
        word_key_out = lemma(" ".join([wk for wk in word_key])).split()

        return X_out, word_key_out


if __name__ == "__main__":
    # import cleaned text data from data_cleaning_and_synthesis.ipynb notebook
    path_to_csv = (
        "data_cleaning_and_synthesis_out\springfield_movie_scripts_2023_01_13_clean.csv"
    )
    df = pd.read_csv(path_to_csv, index_col=[0])
    df = df[["movie_title", "movie_year", "script_text"]]

    # Define stop words from nltk library
    nltk.download("stopwords")
    stop_words = stopwords.words("english")
    stop_words = [sw.replace("'", "") for sw in stop_words]

    # read in word map from .txt file as a dictionary
    # map common misspellings or alternate spellings (e.g. ok -> okay)
    with open("word_map.txt") as f:
        word_map = f.read()
    word_map = json.loads(word_map)

    # preprocess the script text and create BoW representation
    df = df_preprocessing(df, word_map, stop_words)

    X, word_key = construct_BoW(
        df, n_features=10000, max_df=0.8, min_df=3, lemmatization=False
    )

    # save bag-of-words matrix and corresponding word_key
    sparse.save_npz("data_preprocessing_out\\X.npz", X)

    with open("data_preprocessing_out\\word_key.txt", "wb") as f:
        pickle.dump(word_key, f)
