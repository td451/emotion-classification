# Project Explanation: Emotion Classification of Natural Language

## Overview

This project develops machine learning models to classify text into one of 28 distinct emotion categories. By analyzing natural language, the system predicts sentiments such as joy, sadness, anger, and love, enabling a deeper understanding of emotional expression.

## Data Preprocessing and Feature Extraction

### Data Loading
The dataset consists of training and test CSV files. The training data is used to build a vocabulary and generate feature vectors, while the test data is utilized for generating final predictions.

### Text Cleaning and Vocabulary Building
- **Text Preprocessing:**  
  Text is cleaned using spaCy, which removes stop words and lemmatizes tokens to reduce redundancy and noise.
- **Vocabulary Construction:**  
  A vocabulary is built from the cleaned training text, assigning each unique token a numerical index for subsequent vectorization.

### Vectorization
Each text document is converted into a Bag-of-Words (BoW) vector, representing the frequency of each token from the vocabulary.

## Modeling Approaches

### Neural Network with Bag-of-Words (BoW NN)
- **Architecture:**  
  A feed-forward neural network implemented in PyTorch processes the BoW vectors. The model incorporates hidden layers with batch normalization, dropout for regularization, and LeakyReLU activations.
- **Training:**  
  The network is trained using cross-entropy loss. Hyperparameters such as the number of hidden layers, dropout rate, and learning rate are tuned to optimize performance and reduce overfitting.

### Gradient-Boosted Decision Trees (XGBoost)
- **Approach:**  
  An XGBoost classifier is used on the BoW feature set. Experiments include feature reduction via PCA and tuning of tree depth and learning rates.
- **Evaluation:**  
  Performance is measured using accuracy, providing a robust comparison against the neural network model.

### Transformer-Based Model (DistilBERT)
- **Fine-Tuning:**  
  A pre-trained DistilBERT model from Hugging Face is fine-tuned for sequence classification. Text is tokenized using DistilBERT’s tokenizer, and the model is trained using the Trainer API.
- **Metrics:**  
  The fine-tuned model is evaluated using accuracy and weighted F1 scores to ensure balanced performance across classes.

## Model Evaluation and Selection

The training data is split (80/20) into training and validation sets. Each model’s performance is carefully evaluated based on:
- **Accuracy and F1 Score:**  
  These metrics guide the selection of hyperparameters and the overall model selection process.
- **Hyperparameter Tuning:**  
  Techniques such as dropout, weight decay, and adjusting tree depth (for XGBoost) are applied to improve generalization and mitigate overfitting.

## Deployment and Submission

For deployment:
- **Prediction Generation:**  
  Final models generate predictions on the test dataset.
- **CSV Submission:**  
  A CSV file is produced containing the required fields (`id` and `label`), formatted for external evaluation (e.g., Kaggle submissions).

## Conclusion and Future Directions

This project demonstrates multiple approaches to classifying natural language emotions. Comparing a neural network, XGBoost, and a transformer-based model highlights the strengths and limitations of each approach. Future work could focus on:
- Further fine-tuning of the transformer model.
- Exploring ensemble techniques.
- Deploying the models as part of an interactive web application.

## References

- [spaCy Documentation](https://spacy.io/usage)
- [PyTorch Documentation](https://pytorch.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Transformers by Hugging Face](https://huggingface.co/transformers/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
