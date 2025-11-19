"""Visualization utilities for generating research figures."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json

# Set style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class ResearchVisualizer:
    """Generate all figures for research paper."""
    
    def __init__(self, output_dir='figures'):
        """Initialize visualizer.
        
        Args:
            output_dir (str): Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_history(self, history, save_path=None):
        """Plot training and validation metrics over epochs (Figure 3).
        
        Args:
            history: Keras history object or dict with 'accuracy', 'val_accuracy', 'loss', 'val_loss'
            save_path (str): Path to save figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_history.png')
        
        # Extract history data
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        epochs = range(1, len(hist_dict['accuracy']) + 1)
        axes[0].plot(epochs, hist_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
        axes[0].plot(epochs, hist_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
        axes[0].set_title('Training and Validation Accuracy', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0.7, 1.0])
        
        # Plot loss
        axes[1].plot(epochs, hist_dict['loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
        axes[1].plot(epochs, hist_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
        axes[1].set_title('Training and Validation Loss', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {save_path}")
    
    def plot_sample_predictions(self, texts: List[str], predictions: List[str], 
                               confidences: List[float], true_labels: List[str],
                               emotion_labels: List[str], n_samples=10, save_path=None):
        """Create sample predictions table (Figure 4).
        
        Args:
            texts: List of input texts
            predictions: List of predicted emotions
            confidences: List of confidence scores
            true_labels: List of true labels
            emotion_labels: List of emotion label names
            n_samples: Number of samples to display
            save_path: Path to save figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sample_predictions.png')
        
        # Select diverse samples
        n_samples = min(n_samples, len(texts))
        indices = np.linspace(0, len(texts)-1, n_samples, dtype=int)
        
        # Create dataframe
        data = {
            'Input Text': [texts[i][:60] + '...' if len(texts[i]) > 60 else texts[i] for i in indices],
            'True Label': [emotion_labels[true_labels[i]] for i in indices],
            'Predicted': [emotion_labels[predictions[i]] for i in indices],
            'Confidence': [f"{confidences[i]:.1%}" for i in indices],
            'Correct': ['✓' if predictions[i] == true_labels[i] else '✗' for i in indices]
        }
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, n_samples * 0.8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='left', loc='center',
                        colWidths=[0.45, 0.12, 0.12, 0.11, 0.08])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                cell = table[(i, j)]
                if j == 4:  # Correct column
                    if df.iloc[i-1]['Correct'] == '✓':
                        cell.set_facecolor('#C6EFCE')
                    else:
                        cell.set_facecolor('#FFC7CE')
                elif i % 2 == 0:
                    cell.set_facecolor('#F2F2F2')
        
        plt.title('Sample Predictions - Model Output Examples', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sample predictions table saved to {save_path}")
    
    def plot_baseline_comparison(self, models_performance: Dict[str, float], save_path=None):
        """Plot baseline model comparison (Figure 7).
        
        Args:
            models_performance: Dict with model names as keys and metrics dict as values
                               e.g., {'RoBERTa-large': {'accuracy': 0.958, 'f1': 0.957}, ...}
            save_path: Path to save figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'baseline_comparison.png')
        
        models = list(models_performance.keys())
        accuracies = [models_performance[m]['accuracy'] * 100 for m in models]
        f1_scores = [models_performance[m]['f1'] * 100 for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
                      color='#2E86AB', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score',
                      color='#A23B72', edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
        ax.set_title('Performance Comparison with Baseline Models', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(fontsize=11, loc='lower right')
        ax.set_ylim([85, 100])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best model
        best_idx = accuracies.index(max(accuracies))
        ax.axvspan(best_idx - 0.5, best_idx + 0.5, alpha=0.1, color='gold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Baseline comparison plot saved to {save_path}")
    
    def generate_all_figures(self, history, evaluation_results, 
                            test_texts, test_predictions, test_labels,
                            emotion_labels, baseline_results=None):
        """Generate all figures for the research paper.
        
        Args:
            history: Training history
            evaluation_results: Dict with evaluation metrics
            test_texts: List of test texts
            test_predictions: List of predicted labels
            test_labels: List of true labels
            emotion_labels: List of emotion names
            baseline_results: Optional dict with baseline model results
        """
        print("Generating all research figures...")
        print("=" * 50)
        
        # Figure 3: Training History
        print("\n[1/3] Generating Figure 3: Training History...")
        self.plot_training_history(history)
        
        # Figure 4: Sample Predictions
        print("\n[2/3] Generating Figure 4: Sample Predictions...")
        confidences = []
        for i, pred in enumerate(test_predictions):
            # For confidence, you can pass actual probabilities if available
            # This is a placeholder
            confidences.append(0.95)  # Replace with actual confidence scores
        
        self.plot_sample_predictions(
            test_texts, test_predictions, confidences, 
            test_labels, emotion_labels, n_samples=8
        )
        
        # Figure 7: Baseline Comparison
        if baseline_results:
            print("\n[3/3] Generating Figure 7: Baseline Comparison...")
            self.plot_baseline_comparison(baseline_results)
        else:
            print("\n[3/3] Skipping Figure 7 (no baseline data provided)")
            print("\nTo generate Figure 7, provide baseline_results dict:")
            print("Example: {")
            print("    'BERT-base': {'accuracy': 0.932, 'f1': 0.930},")
            print("    'DistilBERT': {'accuracy': 0.918, 'f1': 0.915},")
            print("    'LSTM': {'accuracy': 0.865, 'f1': 0.862},")
            print("    'RoBERTa-large (Ours)': {'accuracy': 0.958, 'f1': 0.957}")
            print("}")
        
        print("\n" + "=" * 50)
        print(f"All figures saved to '{self.output_dir}/' directory")
        print("=" * 50)


def demo_usage():
    """Demonstrate usage of visualization utilities."""
    # Example: Generate figures from saved results
    visualizer = ResearchVisualizer(output_dir='figures')
    
    # Example training history
    example_history = {
        'accuracy': [0.85, 0.91, 0.94, 0.956],
        'val_accuracy': [0.83, 0.89, 0.93, 0.958],
        'loss': [0.42, 0.28, 0.19, 0.12],
        'val_loss': [0.45, 0.31, 0.21, 0.13]
    }
    
    print("Demo: Generating Figure 3...")
    visualizer.plot_training_history(example_history)
    
    # Example baseline comparison
    baseline_results = {
        'BERT-base': {'accuracy': 0.932, 'f1': 0.930},
        'DistilBERT': {'accuracy': 0.918, 'f1': 0.915},
        'LSTM': {'accuracy': 0.865, 'f1': 0.862},
        'RoBERTa-large (Ours)': {'accuracy': 0.958, 'f1': 0.957}
    }
    
    print("\nDemo: Generating Figure 7...")
    visualizer.plot_baseline_comparison(baseline_results)
    
    print("\nDemo complete! Check 'figures/' directory.")


if __name__ == "__main__":
    demo_usage()
