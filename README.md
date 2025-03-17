# Emotion Classification of Natural Language

This repository is part of the CS 3780/5780 Creative Project. It contains a complete machine learning workflow that classifies text into one of 28 emotion categories using several models, including:

- A Bag-of-Words based feed-forward neural network (implemented in PyTorch)
- A gradient-boosted decision tree classifier (using XGBoost)
- A fine-tuned DistilBERT model (using Hugging Face Transformers)

## Project Overview

The task is to classify text into human emotion categories. This project includes:
- **Data Preprocessing:** Loading CSV data, cleaning text using spaCy, building a vocabulary, and vectorizing texts.
- **Modeling:** Implementing and training multiple models:
  - A neural network using a Bag-of-Words (BoW) representation.
  - An XGBoost classifier.
  - A DistilBERT model fine-tuned for sequence classification.
- **Evaluation & Submission:** Splitting data into training and validation sets, evaluating performance using accuracy and F1 score, and generating CSV submission files for Kaggle.

## Repository Structure
emotion-classification/ ├── README.md # This file: project overview and instructions ├── requirements.txt # List of Python dependencies ├── .gitignore # Files and directories to ignore in Git ├── notebooks/ # Jupyter Notebooks (exploratory work and reports) ├── src/ # Python modules containing project code │ ├── data_preprocessing.py # Data loading, cleaning, vocabulary, and vectorization │ ├── bow_model.py # Bag-of-Words model and training loop │ ├── xgb_model.py # XGBoost training and evaluation functions │ ├── distilbert_model.py # DistilBERT fine-tuning using Hugging Face Trainer API │ └── generate_submission.py # Functions to generate CSV submission files └── docs/ # Additional documentation └── explanation.md # Detailed explanation of project choices and results


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/emotion-classification.git
   cd emotion-classification

   Set up a virtual environment (recommended):
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

   Install the dependencies:
    pip install -r requirements.txt

## Usage

1. **Run the Main Script:**

Execute main.py (if available) or run the Jupyter Notebooks in the notebooks/ directory to reproduce the training, evaluation, and submission generation.

python main.py

## Models Overview
  Bag-of-Words Neural Network:
  Defined in src/bow_model.py, this model uses a Bag-of-Words representation to convert text into vectors and trains a feed-forward neural network with batch normalization and dropout to mitigate overfitting.

  XGBoost Classifier:
  Implemented in src/xgb_model.py, this model applies gradient boosting on the BoW features. It includes training, evaluation (using accuracy), and prediction functions.

  DistilBERT Model:
  Fine-tuned for sequence classification in src/distilbert_model.py using the Hugging Face Transformers library. It uses a custom dataset class and the Trainer API to perform training and evaluation.

Acknowledgments
  [spaCy Documentation](https://spacy.io/usage)
  [PyTorch Documentation](https://pytorch.org/)
  [XGBoost Documentation](https://xgboost.readthedocs.io/)
  [transformers](https://huggingface.co/docs/transformers/index)
  [scikit-learn Documentation](https://scikit-learn.org/stable/)