"""DistilBERT model with TensorFlow."""
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from src.config import Config

def create_model(num_labels=6, model_name="distilbert-base-uncased"):
    """Create and return the DistilBERT model."""
    model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    config = Config()
    model.config.id2label = config.EMOTION_LABELS
    model.config.label2id = config.LABEL_TO_ID
    return model

def create_tokenizer(model_name="distilbert-base-uncased"):
    """Create and return the tokenizer."""
    return DistilBertTokenizer.from_pretrained(model_name)

def tokenize_data(texts, tokenizer, max_length=128):
    """Tokenize text data for model input."""
    return tokenizer(
        texts, truncation=True, padding='max_length',
        max_length=max_length, return_tensors='tf'
    )
