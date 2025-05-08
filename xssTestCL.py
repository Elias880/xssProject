import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
df = pd.read_csv("test.csv")
texts = df["payload"].astype(str).tolist()
true_labels = df["label"].astype(int).tolist()

# Load tokenizer and model
with open("xss_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("xss_bilstm_model.keras")

# Tokenize and pad
sequences = tokenizer.texts_to_sequences(texts)
maxlen = model.input_shape[1]
X_test = pad_sequences(sequences, maxlen=maxlen, padding="post")

# Predict
pred_probs = model.predict(X_test)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, digits=4))

# Save predictions
df["predicted_label"] = pred_labels
df["probability"] = pred_probs

df.to_csv("test_predictions.csv", index=False)
print("\nPredictions saved to 'test_predictions.csv'")
