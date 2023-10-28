"""
01_data_preprocessing_LDA.py

Description:
    Run this python file to create bag of words of film script dataset 
    which will be fed into LDA model. Options: remove stopwords, lemmatization/stemming, word mapping

External Dependencies:
    springfield_movie_scripts.csv: Read in as a Pd.DataFrame. Must contain the column 'script_text'
    word_map.txt: Read in as a dictionary, contains key:value pairs for word mappings

returns:
    bag_of_words (scipy sparse matrix): saved to data_preprocessing_out\\bag_of_words.npz
    word_text (list of final vocabulary): saved to data_preprocessing_out\\word_text.txt

"""

import pickle
import json

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import defaultdict


def get_wordnet_pos(word):
    """
    Function to get the part of speech of a word. The part of speech
    will eventually be used into the lemmatization method.

    Args:
        word (str): a single string

    Returns:
        pos (str): part of speech of word
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    pos = tag_dict.get(tag, wordnet.NOUN)
    return pos


def lemma(text):
    """
    Function to transform all words in a string to their lemmatized form.
    e.g. "I am caring" -> "I be care"

    Args:
        text (str)
    Returns:
        lemmatized text (str)
    """

    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text.split()])


def stem(text):
    """
    Function to transform all words in a string to their stemmed form.
    e.g. "I am caring" -> "I am car"

    Args:
        text (str)
    Returns:
        stemmed text (str)
    """
    ps = PorterStemmer()
    return " ".join([ps.stem(w) for w in text.split()])


def list_duplicates(word_list):
    """
    Function to be called after lemmatization is performed. It finds all words
    in a list that are duplicated and returns the duplicates

    Args:
            - word_list (list): a list of lemmatized words

    Returns:
            - (key, locs): a list of tuples where the key is the lemmatized word
              and locs are the indice locations of duplicated entries
    """
    tally = defaultdict(list)
    for i, item in enumerate(word_list):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def df_preprocessing(df, word_map, stop_words):
    """
    Function for preprocesing the entire corpus methods include lowercase,
    punctuation removal, word mappings, and removing stop words

    Args:
        df (pd.DataFrame): DataFrame with columns ['script_text']
        word_map (dictionary): key-value pairs for mapping one word to another (e.g. ok:okay)
        stop_words (list of str): List of words to be removed from corpus

    Returns:
        df_processed (pd.DataFrame)
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
    Function to crete a bag of words for script text

    Args:
            - df (pd.DataFrame): DataFrame with columns ['script_text']
            - n_features (int): maximum number of words for vocabulary
            - max_df (int or float): max word counts (if int) or max doc frequency (if float between 0 and 1)
            - min_df (int or float): min word counts (if int) or min doc frequency (if float between 0 and 1)
            - lemmatization (bool): whether or not to apply lemmatization (default is False)
    Returns:
            - bag_of_words (scipy sparse matrix): a sparse matrix of size (num_documents x n_features)
            - word_key (list of str): features used to create bag_of_words
    """
    if not lemmatization:
        # use scikit-learn's CountVectorizer to create BoW representation
        vectorizer = CountVectorizer(
            max_df=max_df, min_df=min_df, max_features=n_features, stop_words="english"
        )
        bag_of_words = vectorizer.fit_transform(df["script_text"])

        # get word features in increasing item number
        word_key = vectorizer.get_feature_names_out()

        return bag_of_words, word_key
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
        bag_of_words = X_lil.tocsr()
        words_to_keep = np.delete(word_idx, words_to_remove)
        bag_of_words = bag_of_words[:, words_to_keep]
        word_key = [word_key[jword] for jword in words_to_keep]

        # keep words that only appear in more than df_min documents but less than df_max documents
        X_bin = bag_of_words.copy()
        X_bin.data = np.ones(shape=X_bin.data.shape)

        min_df = min_df * len(df) if type(min_df) == float else min_df
        max_df = max_df * len(df) if type(max_df) == float else max_df

        word_idx = np.intersect1d(
            np.argwhere(np.sum(X_bin, axis=0) > min_df)[:, 1],
            np.argwhere(np.sum(X_bin, axis=0) < max_df)[:, 1],
        )
        bag_of_words = bag_of_words[:, word_idx]
        word_key = [word_key[jword] for jword in word_idx]

        # # drop rows that have no words
        # non_blank_movie_scripts = np.argwhere(np.sum(X_bin, axis=1) > 0)[:, 0]
        # bag_of_words = bag_of_words[non_blank_movie_scripts, :]

        # finally retain only top n_features, update word_key accordingly
        wordcount_ordered = np.flip(
            np.argsort(np.array(np.sum(bag_of_words, axis=0)).reshape(-1))
        )
        bag_of_words = bag_of_words[:, wordcount_ordered[0:n_features]]
        word_key = [word_key[jword] for jword in wordcount_ordered[0:n_features]]

        # return lemmatized word_key
        word_key = lemma(" ".join([wk for wk in word_key])).split()

        return bag_of_words, word_key


if __name__ == "__main__":
    # import cleaned text data from data_cleaning_and_synthesis.ipynb notebook
    path_to_csv = "..\\database\\dataset_film_scripts\\springfield_movie_scripts.csv"

    print(f"Loading in cleaned movie script dataset...")
    df = pd.read_csv(path_to_csv, index_col=[0])
    df = df[["movie_title", "movie_year", "script_text"]]
    print("Done!")

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
    print(f"Begin data preprocessing...")
    df = df_preprocessing(df, word_map, stop_words)
    print("Done!")

    print(f"Create bag of words representation...")
    bag_of_words, word_key = construct_BoW(
        df, n_features=10000, max_df=0.8, min_df=3, lemmatization=False
    )
    print("Done!")

    # save bag-of-words matrix and corresponding word_key
    sparse.save_npz("data_preprocessing_out\\bag_of_words.npz", bag_of_words)

    with open("data_preprocessing_out\\word_key.txt", "wb") as f:
        pickle.dump(word_key, f)
