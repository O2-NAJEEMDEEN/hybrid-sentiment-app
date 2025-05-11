
import streamlit as st
import os
import requests
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Hybrid Sentiment App", layout="centered")
st.title("🧠 Hybrid Sentiment Analysis Web App")

# ---------- CONFIG ----------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

GDRIVE_MODELS = {
    "nb_model": "1xiYaLo_N3xWzFhM8bMmUgCvJuvEN6yLQ",
    "tfidf_vectorizer": "1viJXIIothivQinOeB6FvEYsKozlDPHlU",
    "lstm_model": "1VRwXFYM8bhf-pkWBNy9HogpInXiXbUKG",
    "lstm_tokenizer": "1bxq1inqDyHxxLJ-R3nxJwW9WyXKsknwQ",
    "bert_model": "1-U-GEnkYHyG3GZ9el57S0JSwOL7gNR_O",
    "bert_tokenizer_config": "1TbMfCBqNmNSy6Pz7bb8UinZCfj4Z6EHW",
    "bert_vocab": "1n4B4uPyDKVQMY1R-S-6yVkLcF-TEq7LV",
    "bert_special_tokens": "11UA19xJ5mEqo-I4RG5z7oEP6x2n017ZG",
    "bert_config": "1X77Q_sa4hTstHIj_PYTfEydGaBNiGLKt"
}

# ---------- UTILS ----------
def download_from_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    with open(dest_path, 'wb') as f:
        f.write(response.content)

def ensure_file_downloaded(name, ext):
    path = os.path.join(MODEL_DIR, f"{name}.{ext}")
    if not os.path.exists(path):
        st.info(f"Downloading {name}.{ext} from Google Drive...")
        download_from_drive(GDRIVE_MODELS[name], path)
    return path

def download_bert_tokenizer():
    tokenizer_dir = os.path.join(MODEL_DIR, "bert-tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    mapping = {
        "tokenizer_config.json": GDRIVE_MODELS["bert_tokenizer_config"],
        "vocab.txt": GDRIVE_MODELS["bert_vocab"],
        "special_tokens_map.json": GDRIVE_MODELS["bert_special_tokens"],
        "config.json": GDRIVE_MODELS["bert_config"]
    }
    for filename, file_id in mapping.items():
        path = os.path.join(tokenizer_dir, filename)
        if not os.path.exists(path):
            st.info(f"Downloading BERT tokenizer file: {filename}")
            download_from_drive(file_id, path)
    return tokenizer_dir

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    nb_model = joblib.load(ensure_file_downloaded("nb_model", "pkl"))
    vectorizer = joblib.load(ensure_file_downloaded("tfidf_vectorizer", "pkl"))
    lstm_model = tf.keras.models.load_model('models/lstm_model_v3')
    lstm_tokenizer = joblib.load(ensure_file_downloaded("lstm_tokenizer", "pkl"))
    bert_model = TFBertForSequenceClassification.from_pretrained("models/bert-base-uncased", local_files_only=True)
    bert_tokenizer = BertTokenizer.from_pretrained(download_bert_tokenizer())
    return nb_model, vectorizer, lstm_model, lstm_tokenizer, bert_model, bert_tokenizer

nb_model, vectorizer, lstm_model, lstm_tokenizer, bert_model, bert_tokenizer = load_models()
stop_words = ENGLISH_STOP_WORDS

# ---------- PREPROCESS ----------
def preprocess_text(text):
    import re
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# ---------- PREDICT ----------
def predict_nb(text):
    cleaned = preprocess_text(text)
    tfidf_input = vectorizer.transform([cleaned])
    return int(nb_model.predict(tfidf_input)[0])

def predict_lstm(text):
    cleaned = preprocess_text(text)
    seq = lstm_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    pred = lstm_model.predict(padded)[0][0]
    return int(pred > 0.5)

def predict_bert(text):
    encoding = bert_tokenizer(text, truncation=True, padding=True, max_length=200, return_tensors="tf")
    logits = bert_model(encoding)[0]
    return int(tf.argmax(logits, axis=1)[0].numpy())

def predict_hybrid(text):
    votes = [predict_nb(text), predict_lstm(text), predict_bert(text)]
    return int(np.round(np.mean(votes)))

# ---------- UI ----------

review = st.text_area("Enter a product review:", height=150)
if st.button("Predict Sentiment"):
    if review.strip():
        prediction = predict_hybrid(review)
        sentiment = "🟢 Positive" if prediction == 1 else "🔴 Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review before predicting.")
st.markdown("---")
st.caption("Built with Naive Bayes, LSTM, BERT & Ensemble Learning ✨")
