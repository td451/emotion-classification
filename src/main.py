# main.py

from src.data_preprocessing import load_data, build_vocab, vectorize_dataset
from src.bow_model import BoWModel, train_model
from src.generate_submission import generate_submission_torch
import torch
import spacy
import pandas as pd

# Load train and test data
train_df, test_df = load_data("train.csv", "test.csv")

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Build vocabulary from training text
vocab = build_vocab(train_df["text"].tolist(), nlp)

# Vectorize training and test datasets
train_vectors = vectorize_dataset(train_df["text"].tolist(), vocab)
test_vectors = vectorize_dataset(test_df["text"].tolist(), vocab)
labels = torch.tensor(train_df["label"].tolist(), dtype=torch.float32)

# Split training data into train and validation sets (80/20 split)
split = int(len(train_vectors) * 0.8)
train_vec = train_vectors[:split]
train_lab = labels[:split]
valid_vec = train_vectors[split:]
valid_lab = labels[split:]

# Create the BoW model, loss function, and optimizer
model = BoWModel(len(vocab), 100, 28)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
train_model(model, optimizer, loss_fn, train_vec, train_lab, valid_vec, valid_lab, num_epochs=200)

# Generate submission CSV for the BoW model
generate_submission_torch(model, test_vectors, test_df["id"], filename="submission_bow.csv")
