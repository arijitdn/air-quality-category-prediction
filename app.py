# app.py (BEST VERSION)
import json
import joblib
import torch
import pandas as pd
import streamlit as st

from transformers import BertTokenizerFast, BertForSequenceClassification


# ====================================================
#  LOAD MODELS WITH CACHE
# ====================================================

@st.cache_resource
def load_classical_models():
    """Loads sentence embedder + classical ML models."""
    embedder = joblib.load("models/embedder.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "SVM": joblib.load("models/svm.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl")
    }

    return embedder, label_encoder, models


@st.cache_resource
def load_transformer_model():
    """Loads BERT tokenizer + fine-tuned model."""
    tokenizer = BertTokenizerFast.from_pretrained("models/transformer_model/")
    model = BertForSequenceClassification.from_pretrained("models/transformer_model/")
    model.eval()
    return tokenizer, model


@st.cache_data
def load_results():
    """Loads accuracy chart data."""
    try:
        with open("models/results.json", "r") as f:
            results = json.load(f)
        df = pd.DataFrame({"model": list(results.keys()), "accuracy": list(results.values())})
        return df
    except:
        return None


# ====================================================
#  PREDICTION FUNCTIONS
# ====================================================

def predict_classical(model, embedder, label_encoder, text: str):
    """Predict using a classical ML model + sentence embeddings."""
    vector = embedder.encode([text])
    pred_id = model.predict(vector)[0]
    return label_encoder.inverse_transform([pred_id])[0]


def predict_transformer(model, tokenizer, label_encoder, text: str):
    """Predict using fine-tuned BERT-base transformer."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id = torch.argmax(logits).item()
    return label_encoder.inverse_transform([pred_id])[0]


# ====================================================
#  STREAMLIT UI
# ====================================================

def main():
    st.set_page_config(page_title="Air Quality Classifier AI", layout="wide")

    st.title("🌫️ AI-Powered Air Quality Category Classifier")
    st.write("Enter a description of the air/environment to classify it into AQI categories.")

    # Load models
    with st.spinner("Loading models..."):
        embedder, label_encoder, classical_models = load_classical_models()
        tokenizer, bert_model = load_transformer_model()
        results_df = load_results()

    # Sidebar: Accuracy Graph
    st.sidebar.header("📊 Model Accuracy Chart")
    if results_df is not None:
        st.sidebar.bar_chart(results_df.set_index("model")["accuracy"])
    else:
        st.sidebar.info("Run train.py once to generate accuracies.")

    # User Input
    default_text = "Smoggy weather with burning eyes, coughing blood and breathing difficulty"
    text = st.text_area("📝 Enter any air quality description below:", height=110, value=default_text)

    selected_models = st.multiselect(
        "Choose the models to evaluate:",
        list(classical_models.keys()) + ["Transformer (BERT)"],
        default=["Logistic Regression", "SVM", "Random Forest", "Transformer (BERT)"]
    )

    if st.button("🔮 Predict Category"):
        if not text.strip():
            st.warning("Please enter a description.")
            return

        st.subheader("🔍 Predictions")
        predictions = []

        # Classical model predictions
        for name, model in classical_models.items():
            if name in selected_models:
                pred = predict_classical(model, embedder, label_encoder, text)
                predictions.append({"Model": name, "Prediction": pred})

        # BERT-base transformer prediction
        if "Transformer (BERT)" in selected_models:
            pred = predict_transformer(bert_model, tokenizer, label_encoder, text)
            predictions.append({"Model": "Transformer (BERT)", "Prediction": pred})

        # Show predictions table
        st.table(pd.DataFrame(predictions))

        # Majority vote (ensemble)
        final = (
            pd.Series([p["Prediction"] for p in predictions])
            .value_counts()
            .idxmax()
        )

        st.success(f"🏆 Final Majority Category: **{final}**")


if __name__ == "__main__":
    main()
