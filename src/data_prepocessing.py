# src/data_preprocessing.py

import pandas as pd
import numpy as np
import spacy
import torch

def load_data(train_path: str, test_path: str):
    """
    Loads train and test CSV files.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_text(text: str, nlp):
    """
    Uses spaCy to process the text.
    Filters out stop words and returns a list of lemmas.
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

def build_vocab(texts: list, nlp):
    """
    Builds a vocabulary dictionary from a list of texts.
    Each unique token (lemma) is assigned a unique index.
    """
    vocab = {}
    for text in texts:
        tokens = preprocess_text(text, nlp)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def vectorize(text: str, vocab: dict):
    """
    Converts a text into a numerical vector based on the vocabulary.
    The vector counts the occurrence of each word.
    """
    vector = np.zeros(len(vocab))
    for word in text.lower().split():
        if word in vocab:
            vector[vocab[word]] += 1
    return vector

def vectorize_dataset(texts: list, vocab: dict):
    """
    Converts a list of texts into a torch tensor of vectors.
    """
    vectors = [vectorize(text, vocab) for text in texts]
    return torch.tensor(np.array(vectors), dtype=torch.float32)
