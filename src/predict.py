"""Prediction module for emotion detection."""
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from src.config import Config

class EmotionPredictor:
    """Class for emotion prediction."""
    
    def __init__(self, model_path=None):
        config = Config()
        self.model_path = model_path or config.MODEL_SAVE_PATH
        self.config = config
        
        print("\u23f3 Loading model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(self.model_path)
        print("✅ Model loaded successfully!\n")
    
    def predict(self, text):
        """Predict emotion for a single text."""
        inputs = self.tokenizer(text, return_tensors='tf', truncation=True,
                                padding=True, max_length=128)
        
        outputs = self.model(inputs, training=False)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]
        confidence = probabilities[0][predicted_class].numpy()
        
        return {
            'emotion': self.config.EMOTION_LABELS[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                self.config.EMOTION_LABELS[i]: float(prob)
                for i, prob in enumerate(probabilities[0].numpy())
            }
        }
    
    def predict_batch(self, texts):
        """Predict emotions for multiple texts."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
