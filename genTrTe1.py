import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv("XSS_dataset1.csv", encoding="latin1")

# Rename and clean
df = df.rename(columns={"Payloads": "payload", "Class": "label"})

# Map labels to binary
df["label"] = df["label"].map({"Malicious": 1, "Benign": 0})

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Save to CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
