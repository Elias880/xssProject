import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load and prepare data
df = pd.read_csv("train.csv")

texts = df["payload"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

# Char level tokenization
tokenizer = Tokenizer(char_level=True, lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
maxlen = 500
X = pad_sequences(sequences, maxlen=maxlen, padding="post")
y = np.array(labels)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Build BiLSTM model
vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Save model and tokenizer
model.save("xss_bilstm_model.keras")

with open("xss_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)