# src/xgb_model.py

import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_xgb_model(train_vectors, train_labels, valid_vectors, valid_labels, num_class=28):
    """
    Trains an XGBoost classifier.
    """
    # Convert tensors to numpy arrays
    train_vectors_np = train_vectors.numpy()
    valid_vectors_np = valid_vectors.numpy()
    train_labels_np = train_labels.numpy()
    valid_labels_np = valid_labels.numpy()
    
    eval_set = [(train_vectors_np, train_labels_np), (valid_vectors_np, valid_labels_np)]
    
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=100,
        objective="multi:softmax",
        num_class=num_class
    )
    model.fit(train_vectors_np, train_labels_np, eval_set=eval_set, verbose=True)
    return model

def evaluate_xgb_model(model, valid_vectors, valid_labels):
    """
    Evaluates the XGBoost model.
    """
    valid_vectors_np = valid_vectors.numpy()
    valid_labels_np = valid_labels.numpy()
    val_predictions = model.predict(valid_vectors_np)
    val_accuracy = accuracy_score(valid_labels_np, val_predictions)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    return val_accuracy

def predict_xgb_model(model, test_vectors):
    """
    Predicts labels for the test set using the XGBoost model.
    """
    return model.predict(test_vectors.numpy())
