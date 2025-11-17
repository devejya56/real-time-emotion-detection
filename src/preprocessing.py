"""Data preprocessing utilities."""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config

def preprocess_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    return text.strip()

def load_and_preprocess_data(file_path=None):
    """Load and preprocess the emotion dataset."""
    config = Config()
    file_path = file_path or config.DATA_PATH
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    df['content'] = df['content'].apply(preprocess_text)
    df['label'] = df['sentiment'].map(config.LABEL_TO_ID)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"Dataset loaded: {len(df)} samples")
    return df

def split_data(df, test_size=0.1, random_state=42):
    """Split data into train and validation sets."""
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
