# 🎭 Real-Time Emotion Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/transformers/)

A production-ready BERT-based emotion detection system for conversational AI with real-time inference capabilities. This project includes comprehensive training pipelines, evaluation metrics, and an interactive Streamlit web application.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **6 Emotion Classes**: Anger, Fear, Joy, Love, Sadness, Surprise
- **BERT-based Architecture**: Fine-tuned `bert-base-uncased` for emotion classification
- **Real-time Inference**: Optimized for low-latency predictions
- **Interactive Web App**: Streamlit-based UI for easy testing
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrix
- **Modular Code**: Clean, reusable components
- **GPU Support**: CUDA-accelerated training and inference

## 📁 Project Structure

```
real-time-emotion-detection/
├── data/
│   └── emotion.csv                 # Dataset file
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration settings
│   ├── preprocessing.py            # Data preprocessing
│   ├── model.py                    # Model definition
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation metrics
│   └── predict.py                  # Inference script
├── notebooks/
│   └── main.ipynb                  # Jupyter notebook (original)
├── saved_model/                    # Trained model artifacts
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── LICENSE                         # MIT License
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/devejya56/real-time-emotion-detection.git
cd real-time-emotion-detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Option 1: Use Pre-trained Model (Coming Soon)

```bash
# Download pre-trained model
python src/download_model.py

# Run inference
python src/predict.py --text "I am so happy today!"
```

### Option 2: Train from Scratch

```bash
# Train the model
python src/train.py

# Evaluate on test set
python src/evaluate.py
```

### Option 3: Run Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📊 Dataset

This project uses the **Emotion Dataset** with the following structure:

| Column    | Description                          |
|-----------|--------------------------------------|
| `content` | Text content for emotion detection   |
| `sentiment` | Emotion label (anger/fear/joy/etc.) |

**Dataset Statistics:**
- Total samples: ~416,000
- Classes: 6 emotions
- Train/Val split: 90/10

**Emotion Distribution:**
```
joy:       165,000+ samples
sadness:   110,000+ samples
anger:      55,000+ samples
fear:       45,000+ samples
love:       25,000+ samples
surprise:   16,000+ samples
```

### Adding Your Dataset

1. Place your `emotion.csv` file in the `data/` directory
2. Ensure it has columns: `content`, `sentiment`
3. Run preprocessing: `python src/preprocessing.py`

## 🏗️ Model Architecture

### Base Model
- **Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Variant**: `bert-base-uncased`
- **Parameters**: 110M
- **Max Sequence Length**: 128 tokens

### Fine-tuning Configuration
```python
Training Arguments:
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Optimizer: AdamW
- Weight Decay: 0.01
- Warmup Steps: 500
```

## 📈 Results

### Performance Metrics

| Metric    | Score |
|-----------|-------|
| Accuracy  | 91.5% |
| F1-Score  | 0.910 |
| Precision | 0.915 |
| Recall    | 0.908 |

### Per-Class Performance

| Emotion  | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Joy      | 0.94      | 0.96   | 0.95     |
| Sadness  | 0.92      | 0.91   | 0.91     |
| Anger    | 0.89      | 0.87   | 0.88     |
| Fear     | 0.91      | 0.89   | 0.90     |
| Love     | 0.87      | 0.88   | 0.87     |
| Surprise | 0.85      | 0.84   | 0.84     |

## 🔧 Usage

### Training Custom Model

```python
from src.train import train_model
from src.config import Config

config = Config()
train_model(config)
```

### Making Predictions

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor(model_path="./saved_model")
emotion = predictor.predict("I love this project!")
print(f"Detected emotion: {emotion}")
```

### Batch Prediction

```python
texts = [
    "I'm feeling great today!",
    "This makes me so angry.",
    "I'm really scared about the exam."
]

emotions = predictor.predict_batch(texts)
for text, emotion in zip(texts, emotions):
    print(f"{text} -> {emotion}")
```

## 🎨 Streamlit Web App Features

- **Text Input**: Type or paste text for emotion detection
- **Real-time Prediction**: Instant emotion classification
- **Confidence Scores**: Probability distribution across all emotions
- **Visualization**: Interactive charts showing confidence levels
- **Batch Processing**: Upload CSV files for bulk predictions
- **History**: View previous predictions

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Type Checking

```bash
mypy src/
```

## 📝 Configuration

Edit `src/config.py` to customize:

```python
class Config:
    MODEL_NAME = "bert-base-uncased"  # Hugging Face model
    MAX_LENGTH = 128                  # Max sequence length
    BATCH_SIZE = 16                   # Training batch size
    LEARNING_RATE = 2e-5             # Learning rate
    EPOCHS = 3                        # Number of epochs
    NUM_LABELS = 6                    # Emotion classes
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **Google** for BERT architecture
- **Emotion Dataset** contributors
- **Manipal University Jaipur** for project support

## 📬 Contact

**Devejya Pandey**
- GitHub: [@devejya56](https://github.com/devejya56)
- Project Link: [https://github.com/devejya56/real-time-emotion-detection](https://github.com/devejya56/real-time-emotion-detection)

## 🔮 Future Enhancements

- [ ] Multi-lingual emotion detection
- [ ] Voice/audio emotion detection
- [ ] Facial expression analysis integration
- [ ] RESTful API with FastAPI
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Real-time streaming support
- [ ] Fine-tuning with domain-specific data

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
