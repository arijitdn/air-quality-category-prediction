# predict.py
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load classical components
vec = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

models = {
    "knn": joblib.load("models/knn.pkl"),
    "decision_tree": joblib.load("models/decision_tree.pkl"),
    "random_forest": joblib.load("models/random_forest.pkl"),
    "naive_bayes": joblib.load("models/naive_bayes.pkl"),
    "svm": joblib.load("models/svm.pkl"),
}

# Load HF transformer model
tokenizer = DistilBertTokenizerFast.from_pretrained("models/transformer_model/")
hf_model = DistilBertForSequenceClassification.from_pretrained("models/transformer_model/")

# ---------------------------
# Prediction functions
# ---------------------------
def predict_classical(model_name, text):
    vector = vec.transform([text]).toarray()
    pred_id = models[model_name].predict(vector)[0]
    return label_encoder.inverse_transform([pred_id])[0]

def predict_transformer(text):
    enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = hf_model(**enc).logits
    pred_id = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

# ---------------------------
# Ask user for prediction
# ---------------------------
while True:
    text = input("\nEnter air quality description (or 'exit'): ")

    if text.lower() == "exit":
        break

    print("\n=== Predictions ===")
    print("KNN:", predict_classical("knn", text))
    print("Decision Tree:", predict_classical("decision_tree", text))
    print("Random Forest:", predict_classical("random_forest", text))
    print("Naive Bayes:", predict_classical("naive_bayes", text))
    print("SVM:", predict_classical("svm", text))
    print("Transformer:", predict_transformer(text))
