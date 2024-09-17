import torch
import wandb
import json
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from dr import prepare_query

class BinaryClassifier:
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=2, max_length=512, device=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)

    def train(self, train_docs, train_labels, val_docs, val_labels, epochs=3, batch_size=64, lr=2e-5):
        train_dataset = BinaryClassificationDataset(train_docs, train_labels, self.tokenizer, self.max_length)
        val_dataset = BinaryClassificationDataset(val_docs, val_labels, self.tokenizer, self.max_length)

        class_counts = torch.bincount(torch.tensor(train_labels))
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        best_val_accuracy = 0.0
        for epoch in range(epochs):
            self.model.train()
            it = tqdm(enumerate(train_dataloader, start=1))
            for step, batch in it:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                loss = loss_fn(outputs.logits, batch['label'])
                loss.backward()
                optimizer.step()

                it.set_postfix({'loss': loss.item()})

                if step % 1000 == 0:
                    self.model.eval()
                    val_accuracy = self.evaluate(val_dataloader)
                    wandb.log({'val_accuracy': val_accuracy}, step=step)

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        torch.save(self.model.state_dict(), 'best_model.pth')
                
                if step % 10 == 0:
                    wandb.log({'train_loss': loss.item()}, step=step)

    def evaluate(self, dataloader):
        self.model.eval()
        val_accuracy = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch in dataloader:
                val_batch = {k: v.to(self.device) for k, v in val_batch.items()}
                outputs = self.model(input_ids=val_batch['input_ids'], attention_mask=val_batch['attention_mask'])
                predictions = torch.argmax(outputs.logits, dim=1)
                val_accuracy += (predictions == val_batch['label']).sum().item()
                total_samples += val_batch['label'].size(0)
        val_accuracy /= total_samples
        return val_accuracy

    def predict(self, doc):
        encoding = self.tokenizer(doc, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            pred_label = torch.argmax(outputs.logits, dim=1)[0]
        return pred_label.item()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

class BinaryClassificationDataset(Dataset):
    def __init__(self, docs, labels, tokenizer, max_length):
        self.docs = docs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(doc, truncation=True, padding='max_length', max_length=self.max_length)
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'label': torch.tensor(label)
        }

if __name__ == '__main__':
    # Initialize wandb
    wandb.init(project="binary-classifier-deberta")

    # Prepare the data
    print("Loading data...")
    docs = []
    labels = []
    label_0_count = 0
    label_1_count = 0
    with open('train.jsonl') as f:
        for line in tqdm(f):
            d = json.loads(line)
            for i in range(len(d['thread'])):
                d_sub = d.copy()
                d_sub['thread'] = d['thread'][:i]
                doc = prepare_query(d_sub, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
                label = 1 if len(d['thread'][i]['wiki_links']) > 0 else 0
                docs.append(doc)
                labels.append(label)
                if label == 0:
                    label_0_count += 1
                else:
                    label_1_count += 1

    print(f"Label 0 count: {label_0_count}, Label 1 count: {label_1_count}")

    # Split the data into train and validation sets
    train_docs, val_docs, train_labels, val_labels = train_test_split(docs, labels, test_size=1000, stratify=labels)

    # Create an instance of the binary classifier
    classifier = BinaryClassifier()
    classifier.load_model('best_model.pth')

    # Train the classifier
    classifier.train(train_docs, train_labels, val_docs, val_labels)

    # Load the best model
    #classifier.load_model('best_model.pth')

    # Load some examples from the train set
    docs = []
    labels = []
    limit = 100
    with open('train.jsonl') as f:
        for line in f:
            d = json.loads(line)
            for i in range(len(d['thread'])):
                d_sub = d.copy()
                d_sub['thread'] = d_sub['thread'][:i]  # Exclude item at index i
                doc = prepare_query(d_sub, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
                label = 1 if len(d['thread'][i]['wiki_links']) > 0 else 0  # Label based on next item
                docs.append(doc)
                labels.append(label)
            limit -= 1
            if limit == 0:
                break
    # example inference
    correct = 0
    total = 0
    for doc, label in zip(docs, labels):
        pred_label = classifier.predict(doc)
        print(f"Predicted label: {pred_label}, True label: {label}")
        if pred_label == label:
            correct += 1
        total += 1
    print(f"Accuracy: {correct / total}")
