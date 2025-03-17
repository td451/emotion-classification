# src/bow_model.py

import torch
import torch.nn as nn

class BoWModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int):
        super(BoWModel, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self, x):
        x = self.leaky(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, optimizer, loss_fn, train_vectors, train_labels, valid_vectors, valid_labels, num_epochs=200):
    """
    Trains the BoW model and prints training and validation metrics.
    """
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_preds = model(train_vectors)
        train_loss = loss_fn(train_preds, train_labels.long())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_preds = model(valid_vectors)
            valid_loss = loss_fn(valid_preds, valid_labels.long())
            valid_preds_arg = torch.argmax(valid_preds, dim=1)
            valid_accuracy = (valid_preds_arg == valid_labels.long()).sum().item() / valid_labels.size(0)
        
        if epoch % 25 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss.item():.4f}")
            print(f"  Validation Loss: {valid_loss.item():.4f}")
            print(f"  Validation Accuracy: {valid_accuracy * 100:.2f}%")
