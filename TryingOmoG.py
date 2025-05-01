# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:37:11 2025

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import swifter


train_data = pd.read_csv('C:/Users/DELL/Documents/Writing Folder_2/Omogbolahan/DSP/AmazonReviews/amazon_review_polarity_csv/train.csv', header=None, names=['class', 'title', 'review'])
test_data = pd.read_csv('C:/Users/DELL/Documents/Writing Folder_2/Omogbolahan/DSP/AmazonReviews/amazon_review_polarity_csv/test.csv', header=None, names=['class', 'title', 'review'])

print(train_data.head())
print(test_data.head())

print(train_data.isnull().sum())
print(train_data['class'].value_counts())

plt.figure(figsize=(10, 6))  # Set the figure size
train_data['text_length'] = train_data['review'].apply(len)
plt.hist(train_data['text_length'], bins=50, color='skyblue', edgecolor='black')  # Create the histogram
# Add axis labels and a title
plt.xlabel('Length of Review Text', fontsize=14)  # X-axis label
plt.ylabel('Number of Reviews', fontsize=14)  # Y-axis label
plt.title('Distribution of Review Text Lengths', fontsize=16)  # Title

# Show the plot
plt.show()

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = ENGLISH_STOP_WORDS

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

train_data['cleaned_review'] = train_data['review'].swifter.apply(preprocess_text)
test_data['cleaned_review'] = test_data['review'].swifter.apply(preprocess_text)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(train_data['cleaned_review'])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['cleaned_review']), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['cleaned_review']), maxlen=200)

y_train = train_data['class'] - 1  # Convert to 0 (negative) and 1 (positive)
y_test = test_data['class'] - 1

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

vectorizer = TfidfVectorizer(max_features=50000)
X_train_tfidf = vectorizer.fit_transform(train_data['cleaned_review'])
X_test_tfidf = vectorizer.transform(test_data['cleaned_review'])

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
predictions = nb_model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=50000, output_dim=128, input_length=200),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

#%%
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Get predicted probabilities
lstm_probs = model.predict(X_test)

# Convert probabilities to binary predictions (0 or 1)
lstm_preds = (lstm_probs > 0.5).astype(int)

# Flatten the predictions and ground truth
lstm_preds_flat = lstm_preds.flatten()
y_test_flat = y_test.values

# Print metrics
print("LSTM Accuracy:", accuracy_score(y_test_flat, lstm_preds_flat))
print("LSTM Classification Report:\n", classification_report(y_test_flat, lstm_preds_flat))
print("LSTM Confusion Matrix:\n", confusion_matrix(y_test_flat, lstm_preds_flat))
cm = confusion_matrix(y_test_flat, lstm_preds_flat)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LSTM Confusion Matrix')
plt.show()

#%%
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def plot_model_comparison(results_dict):
    """
    Plots a side-by-side bar chart of Accuracy, Precision, Recall, and F1-score
    for multiple models based on classification_report and accuracy_score.

    Parameters:
    - results_dict (dict): Format {
          'Model Name': {
              'y_true': true_labels,
              'y_pred': predicted_labels
          },
          ...
      }
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    scores = {metric: [] for metric in metrics}
    model_names = []

    for model_name, data in results_dict.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        report = classification_report(y_true, y_pred, output_dict=True)
        acc = accuracy_score(y_true, y_pred)
        scores['Accuracy'].append(acc)
        scores['Precision'].append(report['weighted avg']['precision'])
        scores['Recall'].append(report['weighted avg']['recall'])
        scores['F1-score'].append(report['weighted avg']['f1-score'])
        model_names.append(model_name)

    x = np.arange(len(metrics))
    total_models = len(model_names)
    width = 0.8 / total_models  # Adjust bar width to fit all models

    plt.figure(figsize=(10, 6))

    for idx, model_name in enumerate(model_names):
        offsets = x - 0.4 + width/2 + idx * width
        values = [scores[m][idx] for m in metrics]
        plt.bar(offsets, values, width=width, label=model_name)
        for i, v in enumerate(values):
            plt.text(offsets[i], v + 0.005, f'{v:.2f}', ha='center', fontsize=9)

    plt.xticks(x, metrics, fontsize=12)
    plt.ylim(0.75, 1.0)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Prepare your results
results = {
    'Naive Bayes': {
        'y_true': y_test,
        'y_pred': predictions
    },
    'LSTM': {
        'y_true': y_test_flat,
        'y_pred': lstm_preds_flat
    }
}

# Call the comparison plot
plot_model_comparison(results)


#%%
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

# Limit to a subset for performance (adjust as needed)
sample_size = 10000
bert_train_texts = train_data['cleaned_review'][:sample_size].tolist()
bert_test_texts = test_data['cleaned_review'][:sample_size].tolist()
bert_y_train = y_train[:sample_size]
bert_y_test = y_test[:sample_size]

# Tokenization
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = bert_tokenizer(bert_train_texts, truncation=True, padding=True, max_length=200, return_tensors="tf")
test_encodings = bert_tokenizer(bert_test_texts, truncation=True, padding=True, max_length=200, return_tensors="tf")

# Load BERT model for binary classification
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Use Hugging Face-compatible optimizer
batch_size = 16
epochs = 2
num_train_steps = (len(bert_y_train) // batch_size) * epochs
optimizer, _ = create_optimizer(init_lr=2e-5, num_train_steps=num_train_steps, num_warmup_steps=0)

# Compile model
bert_model.compile(optimizer=optimizer,
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# Train the model
bert_model.fit(
    x={'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    y=bert_y_train,
    validation_data=(
        {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},
        bert_y_test
    ),
    epochs=epochs,
    batch_size=batch_size
)

# Make predictions
bert_logits = bert_model.predict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']}).logits
bert_preds_flat = tf.argmax(bert_logits, axis=1).numpy()

#%%
# Prepare results dictionary for dynamic plotting
results = {
    'Naive Bayes': {
        'y_true': y_test,
        'y_pred': predictions
    },
    'LSTM': {
        'y_true': y_test_flat,
        'y_pred': lstm_preds_flat
    },
    'BERT': {
        'y_true': bert_y_test,
        'y_pred': bert_preds_flat
    }
}

plot_model_comparison(results)

#%%
#hybrid model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import mode

# Align to 10k sample subset used by BERT
nb_preds_small = predictions[:10000]
lstm_preds_small = lstm_preds_flat[:10000]
bert_preds_small = bert_preds_flat
y_true_small = y_test[:10000]


# Stack model predictions: shape (3, 10000)
stacked_preds = np.vstack([nb_preds_small, lstm_preds_small, bert_preds_small])

# Apply majority voting along axis=0
hybrid_preds = mode(stacked_preds, axis=0).mode.flatten()

print("Hybrid Model Accuracy:", accuracy_score(y_true_small, hybrid_preds))
print("Hybrid Model Classification Report:\n", classification_report(y_true_small, hybrid_preds))
print("Hybrid Model Confusion Matrix:\n", confusion_matrix(y_true_small, hybrid_preds))

results['Hybrid'] = {
    'y_true': y_true_small,
    'y_pred': hybrid_preds
}

plot_model_comparison(results)

#%%
import os
import joblib

# Create models/ folder
os.makedirs("models", exist_ok=True)

# Save Naive Bayes + Vectorizer
joblib.dump(nb_model, 'models/nb_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Save LSTM + Tokenizer
model.save('models/lstm_model.h5')
joblib.dump(tokenizer, 'models/lstm_tokenizer.pkl')

# Save BERT + Tokenizer
bert_model.save_pretrained('models/bert-base-uncased')
bert_tokenizer.save_pretrained('models/bert-base-uncased')

print("✅ All models saved in /models folder.")

