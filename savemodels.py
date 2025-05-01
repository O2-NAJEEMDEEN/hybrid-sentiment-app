# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 08:36:06 2025

@author: DELL
"""

# save_all_models.py
import os
import joblib
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

# Ensure the models/ directory exists
os.makedirs("models", exist_ok=True)

# Save Naive Bayes model and TF-IDF vectorizer
print("Saving Naive Bayes and TF-IDF vectorizer...")
joblib.dump(nb_model, 'models/nb_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Save LSTM model and tokenizer
print("Saving LSTM model and tokenizer...")
lstm_model.save('models/lstm_model.h5')
joblib.dump(tokenizer, 'models/lstm_tokenizer.pkl')  # Tokenizer used for LSTM

# Save BERT model and tokenizer
print("Saving BERT model and tokenizer...")
bert_model.save_pretrained('models/bert-base-uncased')
bert_tokenizer.save_pretrained('models/bert-base-uncased')

print("All models and tokenizers saved successfully.")
