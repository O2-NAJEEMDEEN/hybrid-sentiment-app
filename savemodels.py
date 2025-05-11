from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import joblib

# Load original tokenizer saved using joblib (locally!)
tokenizer = joblib.load("models/lstm_tokenizer.pkl")

# Convert to JSON and save
with open("models/lstm_tokenizer.json", "w") as f:

    f.write(tokenizer.to_json())
    