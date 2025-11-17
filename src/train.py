"""Training script for DistilBERT emotion detection."""
import tensorflow as tf
from src.config import Config
from src.preprocessing import load_and_preprocess_data, split_data
from src.model import create_model, create_tokenizer, tokenize_data

def train_model():
    """Main training function."""
    print("🚀 Starting training...")
    config = Config()
    
    df = load_and_preprocess_data()
    train_df, val_df = split_data(df, config.TEST_SIZE, config.RANDOM_STATE)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    tokenizer = create_tokenizer(config.MODEL_NAME)
    model = create_model(config.NUM_LABELS, config.MODEL_NAME)
    
    train_encodings = tokenize_data(train_df['content'].tolist(), tokenizer, config.MAX_LENGTH)
    val_encodings = tokenize_data(val_df['content'].tolist(), tokenizer, config.MAX_LENGTH)
    
    train_labels = tf.convert_to_tensor(train_df['label'].values)
    val_labels = tf.convert_to_tensor(val_df['label'].values)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings), train_labels
    )).shuffle(1000).batch(config.BATCH_SIZE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings), val_labels
    )).batch(config.BATCH_SIZE)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
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
