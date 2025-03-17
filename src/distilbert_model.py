# src/distilbert_model.py

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
from sklearn.metrics import f1_score, accuracy_score

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    score = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'f1': score, 'accuracy': acc}

def train_distilbert_model(train_df, test_df, num_labels=28, num_epochs=3, batch_size=16, output_dir='./results'):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).to('cpu')
    
    inputs = tokenizer(train_df["text"].tolist(), padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(train_df["label"].tolist())
    test_inputs = tokenizer(test_df["text"].tolist(), padding=True, truncation=True, return_tensors='pt')
    
    dataset = TextDataset(inputs, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=20,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='steps',
        output_dir=output_dir,
        run_name='my_experiment',
        report_to='none',
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.evaluate()
    
    return model, tokenizer, trainer, test_inputs
