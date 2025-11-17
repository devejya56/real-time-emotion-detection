# 🚀 Complete Setup Script

This file contains ALL the code needed for your TensorFlow + DistilBERT emotion detection model.

## Quick Setup (3 commands):

```bash
# 1. Clone and setup
git clone https://github.com/devejya56/real-time-emotion-detection.git
cd real-time-emotion-detection
python -m venv venv && venv\Scripts\activate  # Windows
# OR: source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# 2. Copy your emotion.csv to data/ folder
mkdir data
# Place emotion.csv in data/

# 3. Run the auto-setup script
python setup_project.py
```

## Create setup_project.py (Copy this entire file)

Create a file called `setup_project.py` in your project root and paste the following code.
Then run: `python setup_project.py`

This will automatically create ALL necessary files with complete working code!

```python
#!/usr/bin/env python3
"""Auto-setup script to create all project files."""

import os

# File contents dictionary
FILES = {
    'src/__init__.py': '''"""Real-Time Emotion Detection Package"""
__version__ = "1.0.0"
''',

    'src/config.py': '''"""Configuration for DistilBERT emotion detection."""

class Config:
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 6
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    DATA_PATH = "data/emotion.csv"
    MODEL_SAVE_PATH = "./saved_model"
    TEST_SIZE = 0.1
    RANDOM_STATE = 42
    
    EMOTION_LABELS = {
        0: "anger", 1: "fear", 2: "joy",
        3: "love", 4: "sadness", 5: "surprise"
    }
    
    LABEL_TO_ID = {
        "anger": 0, "fear": 1, "joy": 2,
        "love": 3, "sadness": 4, "surprise": 5
    }
''',

    'src/preprocessing.py': '''"""Data preprocessing utilities."""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r\'http\\S+|www\\S+\', \'\', text)
    return text.strip()

def load_and_preprocess_data(file_path=None):
    config = Config()
    file_path = file_path or config.DATA_PATH
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    df[\'content\'] = df[\'content\'].apply(preprocess_text)
    df[\'label\'] = df[\'sentiment\'].map(config.LABEL_TO_ID)
    df = df.dropna(subset=[\'label\'])
    df[\'label\'] = df[\'label\'].astype(int)
    print(f"Dataset loaded: {len(df)} samples")
    return df

def split_data(df, test_size=0.1, random_state=42):
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df[\'label\'], random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
''',

    'src/model.py': '''"""DistilBERT model with TensorFlow."""
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from src.config import Config

def create_model(num_labels=6, model_name="distilbert-base-uncased"):
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    config = Config()
    model.config.id2label = config.EMOTION_LABELS
    model.config.label2id = config.LABEL_TO_ID
    return model

def create_tokenizer(model_name="distilbert-base-uncased"):
    return DistilBertTokenizer.from_pretrained(model_name)

def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(
        texts, truncation=True, padding=\'max_length\',
        max_length=max_length, return_tensors=\'tf\'
    )
''',

    'src/train.py': '''"""Training script."""
import tensorflow as tf
from src.config import Config
from src.preprocessing import load_and_preprocess_data, split_data
from src.model import create_model, create_tokenizer, tokenize_data

def train_model():
    print("🚀 Starting training...")
    config = Config()
    
    df = load_and_preprocess_data()
    train_df, val_df = split_data(df, config.TEST_SIZE, config.RANDOM_STATE)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    tokenizer = create_tokenizer(config.MODEL_NAME)
    model = create_model(config.NUM_LABELS, config.MODEL_NAME)
    
    train_encodings = tokenize_data(train_df[\'content\'].tolist(), tokenizer, config.MAX_LENGTH)
    val_encodings = tokenize_data(val_df[\'content\'].tolist(), tokenizer, config.MAX_LENGTH)
    
    train_labels = tf.convert_to_tensor(train_df[\'label\'].values)
    val_labels = tf.convert_to_tensor(val_df[\'label\'].values)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings), train_labels
    )).shuffle(1000).batch(config.BATCH_SIZE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings), val_labels
    )).batch(config.BATCH_SIZE)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=[\'accuracy\'])
    
    print("🏋️ Training...")
    history = model.fit(
        train_dataset, validation_data=val_dataset,
        epochs=config.EPOCHS, verbose=1
    )
    
    print(f"💾 Saving to {config.MODEL_SAVE_PATH}...")
    model.save_pretrained(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    print("✅ Training complete!")
    return model, tokenizer, history

if __name__ == "__main__":
    train_model()
''',

    'src/predict.py': '''"""Inference script."""
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from src.config import Config

class EmotionPredictor:
    def __init__(self, model_path=None):
        config = Config()
        self.model_path = model_path or config.MODEL_SAVE_PATH
        self.config = config
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(self.model_path)
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors=\'tf\', truncation=True,
                                padding=True, max_length=128)
        outputs = self.model(inputs, training=False)
        probabilities = tf.nn.softmax(outputs.logits, axis=1)
        predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]
        confidence = probabilities[0][predicted_class].numpy()
        
        return {
            \'emotion\': self.config.EMOTION_LABELS[predicted_class],
            \'confidence\': float(confidence),
            \'all_probabilities\': {
                self.config.EMOTION_LABELS[i]: float(prob)
                for i, prob in enumerate(probabilities[0].numpy())
            }
        }
''',

    'quick_test.py': '''"""Quick test script."""
from src.predict import EmotionPredictor
import os

def quick_test():
    if not os.path.exists("./saved_model"):
        print("❌ Model not trained yet! Run: python src/train.py")
        return
    
    print("🔧 Loading model...")
    predictor = EmotionPredictor()
    
    test_texts = [
        "I am so happy today!",
        "This makes me really angry.",
        "I\'m scared about the exam.",
        "I love this project!",
        "This is such a sad story.",
        "Wow, what a surprise!"
    ]
    
    print("\\n" + "="*60)
    print("🎭 EMOTION DETECTION TEST RESULTS")
    print("="*60 + "\\n")
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Emotion: {result[\'emotion\'].upper()} ({result[\'confidence\']*100:.1f}% confidence)")
        print("-" * 60)

if __name__ == "__main__":
    quick_test()
''',

    'app.py': '''"""Streamlit web application."""
import streamlit as st
import pandas as pd
from src.predict import EmotionPredictor
import os

st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="wide")
st.title("🎭 Real-Time Emotion Detection")

@st.cache_resource
def load_predictor():
    if not os.path.exists("./saved_model"):
        st.error("❌ Model not found! Please train first: python src/train.py")
        return None
    return EmotionPredictor()

predictor = load_predictor()

if predictor:
    text_input = st.text_area("Enter text:", "I am so happy today!", height=100)
    
    if st.button("Analyze Emotion", type="primary"):
        result = predictor.predict(text_input)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Emotion", result[\'emotion\'].upper())
            st.metric("Confidence", f"{result[\'confidence\']*100:.2f}%")
        
        with col2:
            st.subheader("All Probabilities")
            prob_df = pd.DataFrame([
                {"Emotion": k.upper(), "Probability": f"{v*100:.2f}%"}
                for k, v in result[\'all_probabilities\'].items()
            ])
            st.dataframe(prob_df, hide_index=True)
'''
}

def create_files():
    """Create all project files."""
    print("🚀 Creating project structure...\n")
    
    # Create directories
    os.makedirs('src', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # Create files
    for filepath, content in FILES.items():
        print(f"Creating {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("\n✅ All files created successfully!")
    print("\n📝 Next steps:")
    print("1. Place your emotion.csv file in the data/ folder")
    print("2. Run: python src/train.py")
    print("3. Test: python quick_test.py")
    print("4. Launch app: streamlit run app.py")

if __name__ == "__main__":
    create_files()
```

That's it! Run this script and it will create everything!
