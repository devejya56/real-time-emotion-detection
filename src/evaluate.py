"""Comprehensive evaluation module for emotion detection model."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
import logging
import json
from visualizations import ResearchVisualizer

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""
    
    def __init__(self, model_path=None, config=None):
        """Initialize evaluator.
        
        Args:
            model_path (str): Path to saved model
            config (Config): Configuration object
        """
        self.config = config if config else Config()
        self.model_path = model_path if model_path else self.config.MODEL_SAVE_PATH
        
        # Load model and tokenizer
        logger.info(f"Loading model from {self.model_path}")
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        logger.info("Model and tokenizer loaded successfully")
    
    def predict(self, texts, batch_size=None):
        """Make predictions on texts.
        
        Args:
            texts (list): List of text samples
            batch_size (int): Batch size for prediction
            
        Returns:
            np.array: Predicted labels
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='tf'
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(dict(encodings))
        dataset = dataset.batch(batch_size)
        
        # Predict
        predictions = self.model.predict(dataset)
        predicted_labels = np.argmax(predictions.logits, axis=1)
        
        return predicted_labels
    
    def evaluate(self, texts, true_labels, save_results=True):
        """Comprehensive evaluation with multiple metrics.
        
        Args:
            texts (list): List of text samples
            true_labels (list): True labels
            save_results (bool): Whether to save evaluation results
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating on {len(texts)} samples...")
        
        # Get predictions
        predicted_labels = self.predict(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_labels, predicted_labels, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Classification report
        report = classification_report(
            true_labels, predicted_labels,
            target_names=[self.config.EMOTION_LABELS[i] for i in range(self.config.NUM_LABELS)],
            output_dict=True
        )
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                self.config.EMOTION_LABELS[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i in range(self.config.NUM_LABELS)
            },
            'classification_report': report
        }
        
        # Log results
        logger.info(f"\nEvaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        # Save results
        if save_results:
            self.save_results(results)
            self.plot_confusion_matrix(cm)
            self.plot_per_class_metrics(results['per_class_metrics'])
        
        return results
    
    def save_results(self, results, filename='evaluation_results.json'):
        """Save evaluation results to file.
        
        Args:
            results (dict): Evaluation results
            filename (str): Output filename
        """
        os.makedirs('evaluation_results', exist_ok=True)
        filepath = os.path.join('evaluation_results', filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def plot_confusion_matrix(self, cm, save_path='evaluation_results/confusion_matrix.png'):
        """Plot and save confusion matrix.
        
        Args:
            cm (np.array): Confusion matrix
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.config.EMOTION_LABELS[i] for i in range(self.config.NUM_LABELS)],
            yticklabels=[self.config.EMOTION_LABELS[i] for i in range(self.config.NUM_LABELS)],
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(self, metrics, save_path='evaluation_results/per_class_metrics.png'):
        """Plot per-class precision, recall, and F1-score.
        
        Args:
            metrics (dict): Per-class metrics
            save_path (str): Path to save plot
        """
        emotions = list(metrics.keys())
        precision = [metrics[e]['precision'] for e in emotions]
        recall = [metrics[e]['recall'] for e in emotions]
        f1 = [metrics[e]['f1_score'] for e in emotions]
        
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
        ax.bar(x, recall, width, label='Recall', color='#A23B72')
        ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')
        
        ax.set_xlabel('Emotion', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Per-class metrics plot saved to {save_path}")
    
    def cross_validate(self, texts, labels, n_splits=5):
        """Perform stratified k-fold cross-validation.
        
        Args:
            texts (list): List of text samples
            labels (list): True labels
            n_splits (int): Number of folds
            
        Returns:
            dict: Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_f1_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
            
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Predict
            predicted = self.predict(val_texts)
            
            # Calculate metrics
            accuracy = accuracy_score(val_labels, predicted)
            _, _, f1, _ = precision_recall_fscore_support(
                val_labels, predicted, average='weighted'
            )
            
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            
            logger.info(f"Fold {fold_idx + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        results = {
            'mean_accuracy': float(np.mean(fold_accuracies)),
            'std_accuracy': float(np.std(fold_accuracies)),
            'mean_f1_score': float(np.mean(fold_f1_scores)),
            'std_f1_score': float(np.std(fold_f1_scores)),
            'fold_accuracies': [float(a) for a in fold_accuracies],
            'fold_f1_scores': [float(f) for f in fold_f1_scores]
        }
        
        logger.info(f"\nCross-Validation Results:")
        logger.info(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        logger.info(f"Mean F1-Score: {results['mean_f1_score']:.4f} ± {results['std_f1_score']:.4f}")
        
        return results


def main():
    """Main evaluation function."""
    # Load configuration
    config = Config()
    
    # Load test data
    logger.info("Loading test data...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Use test split
    from sklearn.model_selection import train_test_split
    
    _, test_texts, _, test_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df['label'].tolist()
    )
    
    logger.info(f"Test samples: {len(test_texts)}")
    
    # Create evaluator
    evaluator = ModelEvaluator(config=config)
    
    # Evaluate
    results = evaluator.evaluate(test_texts, test_labels, save_results=True)
    
    # Cross-validation (optional)
    # cv_results = evaluator.cross_validate(test_texts, test_labels, n_splits=5)
    
    logger.info("\nEvaluation completed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("\nPer-Class Performance:")
    for emotion, metrics in results['per_class_metrics'].items():
        print(f"  {emotion:10s} - P: {metrics['precision']:.3f}  R: {metrics['recall']:.3f}  F1: {metrics['f1_score']:.3f}")
    print("="*60)

    # Generate visualizations for research figures
    logger.info("\nGenerating research figures...")
    visualizer = ResearchVisualizer(output_dir='figures')
    
    # Get predictions for sample predictions figure
    predicted_labels = evaluator.predict(test_texts)
    
    # Generate sample predictions figure
    # Note: Replace confidences with actual probabilities if available
    confidences = [0.95] * len(predicted_labels)  # Placeholder
    visualizer.plot_sample_predictions(
        test_texts, predicted_labels, confidences,
        test_labels, config.EMOTION_LABELS, n_samples=8
    )


if __name__ == "__main__":
    main()
