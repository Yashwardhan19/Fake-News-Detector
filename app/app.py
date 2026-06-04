import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.preprocessor import TextPreprocessor
from src.feature_extractor import FeatureExtractor


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_artifacts():

    pp = TextPreprocessor()

    fe = FeatureExtractor()
    fe.load()

    model = joblib.load(ROOT_DIR / "models" / "ensemble_model.pkl")

    return pp, fe, model


pp, fe, model = load_artifacts()


# -----------------------------
# Prediction Function
# -----------------------------
def predict_news(text):

    # Clean text
    clean_text = pp.clean(text)

    # Meta features
    meta_features = pp.get_meta_features(text)

    # Create DataFrame
    row = {
        "clean_text": clean_text,
        **meta_features
    }

    df = pd.DataFrame([row])

    # Feature Extraction
    X = fe.transform(df)

    # Prediction
    prediction = model.predict(X)[0]

    # Probability
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities) * 100

    return prediction, confidence


# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detector")

st.markdown(
    """
    Paste a news article below and click **Analyze**.
    """
)

news_text = st.text_area(
    "Paste News Article",
    height=300,
    placeholder="Paste news content here..."
)

if st.button("Analyze", use_container_width=True):

    if not news_text.strip():
        st.warning("Please enter some news text.")
        st.stop()

    with st.spinner("Analyzing article..."):

        try:

            prediction, confidence = predict_news(news_text)

            st.divider()

            # 0 = REAL, 1 = FAKE
            if prediction == 0:
                st.success("✅ REAL NEWS")
            else:
                st.error("🚨 FAKE NEWS")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")