"""Advanced training script with optimizations for RoBERTa-large emotion detection."""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import create_optimizer
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbCallback
import logging
from datetime import datetime

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from preprocessing import load_and_preprocess_data
from augmentation import augment_dataset, TextAugmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """Advanced trainer with learning rate scheduling and callbacks."""
    
    def __init__(self, config=None, use_wandb=True):
        """Initialize advanced trainer.
        
        Args:
            config (Config): Configuration object
            use_wandb (bool): Whether to use Weights & Biases logging
        """
        self.config = config if config else Config()
        self.use_wandb = use_wandb
        
        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project="emotion-detection-roberta",
                config={
                    "model": self.config.MODEL_NAME,
                    "batch_size": self.config.BATCH_SIZE,
                    "learning_rate": self.config.LEARNING_RATE,
                    "epochs": self.config.EPOCHS,
                    "max_length": self.config.MAX_LENGTH
                }
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        logger.info(f"Initialized AdvancedTrainer with {self.config.MODEL_NAME}")
    
    def create_lr_scheduler(self, num_train_steps):
        """Create learning rate scheduler with warmup.
        
        Args:
            num_train_steps (int): Total number of training steps
            
        Returns:
            Learning rate schedule
        """
        # Warmup for 10% of training
        num_warmup_steps = int(0.1 * num_train_steps)
        
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.config.LEARNING_RATE,
            decay_steps=num_train_steps - num_warmup_steps,
            end_learning_rate=self.config.LEARNING_RATE * 0.1,
            power=1.0
        )
        
        # Add warmup
        warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0,
            decay_steps=num_warmup_steps,
            end_learning_rate=self.config.LEARNING_RATE,
            power=1.0
        )
        
        logger.info(f"Created LR scheduler: warmup={num_warmup_steps}, decay={num_train_steps}")
        
        return lr_schedule
    
    def create_callbacks(self):
        """Create training callbacks.
        
        Returns:
            list: List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_dir = './checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'model_{{epoch:02d}}_{{val_accuracy:.4f}}.h5'
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Learning rate logging
        lr_logger = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr,
            verbose=0
        )
        callbacks.append(lr_logger)
        
        # Reduce LR on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # W&B callback
        if self.use_wandb:
            callbacks.append(WandbCallback(
                save_model=False,
                monitor='val_accuracy'
            ))
        
        # TensorBoard
        log_dir = os.path.join(
            'logs',
            datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        logger.info(f"Created {len(callbacks)} callbacks")
        
        return callbacks
    
    def create_model(self):
        """Create and compile RoBERTa model.
        
        Returns:
            Compiled TensorFlow model
        """
        logger.info(f"Loading model: {self.config.MODEL_NAME}")
        
        model = TFAutoModelForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.config.NUM_LABELS
        )
        
        # Create optimizer with weight decay
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=0.01,
            epsilon=1e-8
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        logger.info("Model compiled successfully")
        
        return model
    
    def prepare_dataset(self, texts, labels, augment=True):
        """Prepare and augment dataset.
        
        Args:
            texts (list): List of text samples
            labels (list): List of labels
            augment (bool): Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset: Prepared dataset
        """
        # Apply data augmentation
        if augment:
            logger.info("Applying data augmentation...")
            augmenter = TextAugmenter()
            texts, labels = augment_dataset(
                texts, labels,
                augmentation_factor=2,
                augmenter=augmenter
            )
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='tf'
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            np.array(labels)
        ))
        
        # Shuffle and batch
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train(self, train_texts, train_labels, val_texts, val_labels):
        """Train model with advanced optimizations.
        
        Args:
            train_texts (list): Training texts
            train_labels (list): Training labels
            val_texts (list): Validation texts
            val_labels (list): Validation labels
            
        Returns:
            Training history
        """
        logger.info("Starting advanced training...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels, augment=True)
        val_dataset = self.prepare_dataset(val_texts, val_labels, augment=False)
        
        # Create model
        model = self.create_model()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        save_path = self.config.MODEL_SAVE_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
        
        # Log final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
        logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
        if self.use_wandb:
            wandb.log({
                "final_train_accuracy": final_train_acc,
                "final_val_accuracy": final_val_acc
            })
            wandb.finish()
        
        return history



        # Save training history and generate visualization
        logger.info("Saving training history and generating plots...")
        import json
        history_dict = {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        # Save history to JSON
        os.makedirs('training_logs', exist_ok=True)
        with open('training_logs/history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Generate training history plot
        from visualizations import ResearchVisualizer
        visualizer = ResearchVisualizer(output_dir='figures')
        visualizer.plot_training_history(history)
        logger.info("Training history saved and plotted successfully")
def main():
    """Main training function."""
    # Load configuration
    config = Config()
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df['label'].tolist()
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Create trainer
    trainer = AdvancedTrainer(config=config, use_wandb=True)
    
    # Train model
    history = trainer.train(
        train_texts, train_labels,
        val_texts, val_labels
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
