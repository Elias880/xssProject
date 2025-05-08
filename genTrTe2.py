import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("XSS_dataset2.csv") 
df = df.rename(columns={"Sentence": "payload", "Label": "label"})

# Clean payloads
df = df.dropna(subset=["payload", "label"])
df["payload"] = df["payload"].astype(str)
df["label"] = df["label"].astype(int)

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Save to CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)