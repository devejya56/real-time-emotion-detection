"""Quick test script to verify emotion detection model."""
import os
import sys
from src.predict import EmotionPredictor

def main():
    """Run quick test with example texts."""
    print("="*60)
    print("Real-Time Emotion Detection - Quick Test")
    print("="*60)
    
    # Check if model exists
    model_path = "./saved_model"
    if not os.path.exists(model_path):
        print("\n❌ Error: Model not found!")
        print(f"Expected model at: {model_path}")
        print("\nPlease train the model first by running:")
        print("  python src/train.py")
        sys.exit(1)
    
    # Initialize predictor
    print("\n⏳ Loading model...")
    try:
        predictor = EmotionPredictor(model_path)
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)
    
    # Test examples for each emotion
    test_texts = [
        "I am so angry about this situation!",
        "I'm really scared about what might happen.",
        "This is the best day ever! I'm so happy!",
        "I love spending time with you.",
        "I feel so sad and lonely today.",
        "Wow, I can't believe this just happened!"
    ]
    
    print("Testing emotion detection on sample texts:\n")
    print("-"*60)
    
    # Run predictions
    for i, text in enumerate(test_texts, 1):
        result = predictor.predict(text)
        
        print(f"\nTest {i}:")
        print(f"Text: \"{text}\"")
        print(f"Detected Emotion: {result['emotion'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nAll Probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"  {emotion:10s}: {bar:20s} {prob:.2%}")
        print("-"*60)
    
    print("\n✅ Quick test completed!")
    print("\nTo test with your own text, use:")
    print("  from src.predict import EmotionPredictor")
    print("  predictor = EmotionPredictor('./saved_model')")
    print("  result = predictor.predict('Your text here')")
    print("\nOr run the Streamlit app:")
    print("  streamlit run app.py\n")

if __name__ == "__main__":
    main()
