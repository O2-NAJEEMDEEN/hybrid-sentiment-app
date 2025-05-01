# HybridApp.py (Streamlit Cloud Ready)
import streamlit as st
import re
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Hybrid Sentiment App", layout="centered")
# ---------------------- Load Models ---------------------- #
@st.cache_resource
def load_all_models():
    nb_model = joblib.load('models/nb_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    lstm_tokenizer = joblib.load('models/lstm_tokenizer.pkl')
    bert_model = TFBertForSequenceClassification.from_pretrained('models/bert-base-uncased')
    bert_tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
    return nb_model, vectorizer, lstm_model, lstm_tokenizer, bert_model, bert_tokenizer

nb_model, vectorizer, lstm_model, lstm_tokenizer, bert_model, bert_tokenizer = load_all_models()
stop_words = ENGLISH_STOP_WORDS

# ---------------------- Preprocessing ---------------------- #
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ---------------------- Prediction Functions ---------------------- #
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

# ---------------------- Streamlit UI ---------------------- #
st.title("🧠 Hybrid Sentiment Analysis Web App")

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
