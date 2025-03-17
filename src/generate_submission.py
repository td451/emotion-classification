# src/generate_submission.py

import pandas as pd
import torch

def generate_submission_torch(model, test_vectors, test_ids, filename='submission.csv'):
    """
    Generates a submission CSV using a PyTorch model.
    """
    model.eval()
    with torch.no_grad():
        test_preds = model(test_vectors)
        test_preds_arg = torch.argmax(test_preds, dim=1).numpy()
    predictions_df = pd.DataFrame({'id': test_ids, 'label': test_preds_arg})
    predictions_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")

def generate_submission_xgb(model, test_vectors, test_ids, filename='submission2.csv'):
    """
    Generates a submission CSV using an XGBoost model.
    """
    test_predictions = model.predict(test_vectors.numpy())
    predictions_df = pd.DataFrame({'id': test_ids, 'label': test_predictions})
    predictions_df.to_csv(filename, index=False)
    print(f"XGB Submission saved to {filename}")

def generate_submission_bert(predicted_classes, test_ids, filename='submission3.csv'):
    """
    Generates a submission CSV for the DistilBERT model.
    """
    predictions_df = pd.DataFrame({'id': test_ids, 'predictions': predicted_classes})
    predictions_df.to_csv(filename, index=False)
    print(f"BERT Submission saved to {filename}")
