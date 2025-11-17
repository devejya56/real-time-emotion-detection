"""Configuration settings for DistilBERT emotion detection model."""

class Config:
    """Model configuration class."""
    
    # Model settings
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 6
    MAX_LENGTH = 128
    
    # Training settings
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    
    # Data settings
    DATA_PATH = "data/emotion.csv"
    MODEL_SAVE_PATH = "./saved_model"
    TEST_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Emotion labels mapping
    EMOTION_LABELS = {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "love",
        4: "sadness",
        5: "surprise"
    }
    
    LABEL_TO_ID = {
        "anger": 0,
        "fear": 1,
        "joy": 2,
        "love": 3,
        "sadness": 4,
        "surprise": 5
    }
