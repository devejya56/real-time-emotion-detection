"""Streamlit web application for real-time emotion detection."""
import streamlit as st
import os
import sys
from src.predict import EmotionPredictor
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Real-Time Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.emotion-card {
    padding: 1.5rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    margin: 1rem 0;
}
.emotion-result {
    font-size: 2.5rem;
    font-weight: bold;
    margin: 1rem 0;
}
.confidence-score {
    font-size: 1.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Emotion emojis
EMOTION_EMOJIS = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊',
    'love': '❤️',
    'sadness': '😢',
    'surprise': '😲'
}

# Title
st.markdown('<div class="main-header">🎭 Real-Time Emotion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by DistilBERT & TensorFlow</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    st.subheader("📊 Model Information")
    st.write("**Model:** DistilBERT")
    st.write("**Framework:** TensorFlow 2.14+")
    st.write("**Emotions:** 6 classes")
    
    st.markdown("---")
    st.subheader("🎯 Detected Emotions")
    for emotion, emoji in EMOTION_EMOJIS.items():
        st.write(f"{emoji} {emotion.capitalize()}")
    
    st.markdown("---")
    st.subheader("📝 Instructions")
    st.write("""
    1. Enter text in the input box
    2. Click 'Analyze Emotion'
    3. View results and confidence scores
    4. Try different texts!
    """)

# Initialize session state
if 'predictor' not in st.session_state:
    model_path = "./saved_model"
    if os.path.exists(model_path):
        with st.spinner('⏳ Loading model... Please wait...'):
            try:
                st.session_state.predictor = EmotionPredictor(model_path)
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.stop()
    else:
        st.error(f"""
        ❌ **Model not found!**
        
        Expected model at: `{model_path}`
        
        Please train the model first by running:
        ```bash
        python src/train.py
        ```
        """)
        st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Enter Your Text")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Text Box", "Example Texts"],
        horizontal=True
    )
    
    if input_method == "Text Box":
        user_text = st.text_area(
            "Type or paste your text here:",
            height=150,
            placeholder="e.g., I'm so happy today! Everything is going great!"
        )
    else:
        example_texts = [
            "I am so angry about this situation!",
            "I'm really scared about what might happen.",
            "This is the best day ever! I'm so happy!",
            "I love spending time with you.",
            "I feel so sad and lonely today.",
            "Wow, I can't believe this just happened!"
        ]
        user_text = st.selectbox(
            "Select an example:",
            ["-- Choose an example --"] + example_texts
        )
        if user_text == "-- Choose an example --":
            user_text = ""
    
    analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

with col2:
    st.subheader("💡 Tips")
    st.info("""
    - Be specific with your emotions
    - Use complete sentences
    - Try different writing styles
    - Mix positive and negative sentiments
    """)

# Analysis results
if analyze_button and user_text:
    with st.spinner('🧠 Analyzing emotion...'):
        try:
            result = st.session_state.predictor.predict(user_text)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Analysis Results")
            
            # Main result card
            emotion = result['emotion']
            confidence = result['confidence']
            emoji = EMOTION_EMOJIS.get(emotion, '😐')
            
            st.markdown(f"""
            <div class="emotion-card">
                <div class="emotion-result">{emoji} {emotion.upper()}</div>
                <div class="confidence-score">Confidence: {confidence:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed probabilities
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("📊 Probability Breakdown")
                for emo, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                    emoji_icon = EMOTION_EMOJIS.get(emo, '😐')
                    st.progress(prob, text=f"{emoji_icon} {emo.capitalize()}: {prob:.2%}")
            
            with col_b:
                st.subheader("📊 Visualization")
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=[e.capitalize() for e in result['all_probabilities'].keys()],
                    values=list(result['all_probabilities'].values()),
                    hole=.3,
                    marker_colors=['#ff6b6b', '#ee5a6f', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
                )])
                fig.update_layout(
                    showlegend=True,
                    height=300,
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Input text display
            with st.expander("📝 View Input Text"):
                st.write(user_text)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.exception(e)

elif analyze_button and not user_text:
    st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Built with ❤️ using Streamlit, DistilBERT, and TensorFlow</p>
    <p>For more information, check the <a href="https://github.com/devejya56/real-time-emotion-detection" target="_blank">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
