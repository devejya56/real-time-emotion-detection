"""Data augmentation utilities for emotion detection model."""

import numpy as np
import nlpaug.augmenter.word as naw
from googletrans import Translator
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextAugmenter:
    """Text augmentation for improving model robustness."""
    
    def __init__(self):
        """Initialize augmenters."""
        # Contextual word embeddings augmenter using RoBERTa
        self.context_aug = naw.ContextualWordEmbsAug(
            model_path='roberta-base',
            action='substitute',
            aug_p=0.15  # Replace 15% of words
        )
        
        # Synonym replacement augmenter
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.15)
        
        # Back-translation augmenter
        self.translator = Translator()
        
        logger.info("Text augmenters initialized successfully")
    
    def contextual_augment(self, text):
        """Augment text using contextual word embeddings.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Augmented text
        """
        try:
            augmented = self.context_aug.augment(text)
            return augmented[0] if isinstance(augmented, list) else augmented
        except Exception as e:
            logger.warning(f"Contextual augmentation failed: {e}")
            return text
    
    def synonym_augment(self, text):
        """Augment text using synonym replacement.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Augmented text
        """
        try:
            augmented = self.synonym_aug.augment(text)
            return augmented[0] if isinstance(augmented, list) else augmented
        except Exception as e:
            logger.warning(f"Synonym augmentation failed: {e}")
            return text
    
    def back_translate(self, text, intermediate_lang='fr'):
        """Augment text using back-translation.
        
        Args:
            text (str): Input text
            intermediate_lang (str): Intermediate language code (default: 'fr')
            
        Returns:
            str: Back-translated text
        """
        try:
            # Translate to intermediate language
            translated = self.translator.translate(text, dest=intermediate_lang)
            # Translate back to English
            back_translated = self.translator.translate(
                translated.text, dest='en'
            )
            return back_translated.text
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text
    
    def augment_text(self, text, method='random'):
        """Apply random or specific augmentation method.
        
        Args:
            text (str): Input text
            method (str): Augmentation method - 'random', 'contextual', 
                         'synonym', or 'backtranslate'
            
        Returns:
            str: Augmented text
        """
        if method == 'random':
            method = random.choice(['contextual', 'synonym', 'backtranslate', 'none'])
        
        if method == 'contextual':
            return self.contextual_augment(text)
        elif method == 'synonym':
            return self.synonym_augment(text)
        elif method == 'backtranslate':
            return self.back_translate(text)
        else:
            return text


def augment_dataset(texts, labels, augmentation_factor=2, augmenter=None):
    """Augment entire dataset with balanced class distribution.
    
    Args:
        texts (list): List of text samples
        labels (list): List of corresponding labels
        augmentation_factor (int): Number of augmented samples per original
        augmenter (TextAugmenter): Augmenter instance (creates new if None)
        
    Returns:
        tuple: (augmented_texts, augmented_labels)
    """
    if augmenter is None:
        augmenter = TextAugmenter()
    
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    
    # Calculate class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = max(counts)
    
    logger.info(f"Original dataset size: {len(texts)}")
    logger.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Balance classes by augmenting minority classes
    for label, count in zip(unique_labels, counts):
        if count < max_count:
            # Find indices for this class
            label_indices = [i for i, l in enumerate(labels) if l == label]
            
            # Calculate how many augmentations needed
            augmentations_needed = (max_count - count) * augmentation_factor
            
            logger.info(f"Augmenting class {label}: {augmentations_needed} samples")
            
            # Generate augmentations
            for _ in range(augmentations_needed):
                # Randomly select a sample from this class
                idx = random.choice(label_indices)
                original_text = texts[idx]
                
                # Apply random augmentation
                augmented_text = augmenter.augment_text(original_text)
                
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)
    
    logger.info(f"Augmented dataset size: {len(augmented_texts)}")
    
    return augmented_texts, augmented_labels


def create_validation_augmentations(texts, labels, n_augmentations=3):
    """Create multiple augmentations for test-time augmentation.
    
    Args:
        texts (list): List of text samples
        labels (list): List of corresponding labels
        n_augmentations (int): Number of augmentations per sample
        
    Returns:
        tuple: (augmented_texts, augmented_labels, sample_ids)
    """
    augmenter = TextAugmenter()
    
    augmented_texts = []
    augmented_labels = []
    sample_ids = []
    
    for idx, (text, label) in enumerate(zip(texts, labels)):
        # Add original
        augmented_texts.append(text)
        augmented_labels.append(label)
        sample_ids.append(idx)
        
        # Add augmentations
        for _ in range(n_augmentations):
            aug_text = augmenter.augment_text(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
            sample_ids.append(idx)
    
    return augmented_texts, augmented_labels, sample_ids


if __name__ == "__main__":
    # Test augmentation
    augmenter = TextAugmenter()
    
    test_texts = [
        "I am so happy today!",
        "This makes me feel sad and lonely.",
        "I am really angry about this situation."
    ]
    
    print("Testing augmentation methods:\n")
    
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Contextual: {augmenter.contextual_augment(text)}")
        print(f"Synonym: {augmenter.synonym_augment(text)}")
        print(f"Back-translate: {augmenter.back_translate(text)}")
        print("-" * 80)
