import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import json

from sentence_transformers import SentenceTransformer

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
df = pd.read_csv("air_quality_dataset.csv")

label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

X = df["text"]
y = df["label_id"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------------------------------------
# SENTENCE EMBEDDINGS
# ----------------------------------------------------------
print("\nGenerating sentence embeddings...\n")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

X_train_vec = embedder.encode(list(X_train), show_progress_bar=True)
X_test_vec  = embedder.encode(list(X_test), show_progress_bar=True)

# ----------------------------------------------------------
# HIGH-ACCURACY CLASSICAL MODELS
# ----------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=4000, C=3.0),
    "svm": SVC(kernel="linear", C=2.0, probability=True),
    "random_forest": RandomForestClassifier(
        n_estimators=350,
        max_depth=50,
        min_samples_split=2
    ),
    "knn": KNeighborsClassifier(n_neighbors=4, weights="distance"),
    "decision_tree": DecisionTreeClassifier(max_depth=25),
    "naive_bayes": GaussianNB(),
}

results = {}

print("\nTraining classical ML models...\n")

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name} accuracy: {acc:.4f}")

# Save classical ML components
joblib.dump(embedder, "models/embedder.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

for name, model in models.items():
    joblib.dump(model, f"models/{name}.pkl")

print("\nClassical ML models saved!")


# ----------------------------------------------------------
# BERT-BASE TRANSFORMER
# ----------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

class AirDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(texts.tolist(), truncation=True, padding=True)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AirDataset(X_train, y_train)
test_dataset  = AirDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(df["label_id"].unique())
)

# ---------------------------
# CLASS-WEIGHTED LOSS
# ---------------------------
class_counts = df["label_id"].value_counts().sort_index().values
weights = torch.tensor(1.0 / class_counts, dtype=torch.float)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

total_steps = len(train_loader) * 12
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# ----------------------------------------------------------
# TRAIN BERT (12 EPOCHS)
# ----------------------------------------------------------
print("\nFine-tuning BERT...\n")

model.train()
for epoch in range(12):
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        loss = loss_fn(out.logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print(f"Epoch {epoch+1} completed!")

# Save BERT safely on Windows
model.save_pretrained("models/transformer_model/", safe_serialization=False)
tokenizer.save_pretrained("models/transformer_model/")

print("\nBERT model saved successfully!")


# ----------------------------------------------------------
# EVALUATE BERT
# ----------------------------------------------------------
print("\nEvaluating BERT...\n")

model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        ).logits
        preds.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(batch["labels"].tolist())

hf_acc = accuracy_score(true_labels, preds)
results["bert_transformer"] = hf_acc

print(f"\nBERT Accuracy: {hf_acc:.4f}")

# ----------------------------------------------------------
# SAVE FINAL RESULTS FOR GRAPH
# ----------------------------------------------------------
with open("models/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved all accuracies to models/results.json\n")
