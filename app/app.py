import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to Python path so src/ modules are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.preprocessor import TextPreprocessor
from src.feature_extractor import FeatureExtractor
from src.explainer import FakeNewsExplainer

# Configure browser tab title, icon, and wide layout
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


@st.cache_resource  # Cache models in memory — avoids reloading on every interaction
def load_artifacts():
    pp = TextPreprocessor()

    fe = FeatureExtractor()
    fe.load()  # Load saved TF-IDF + SBERT fitted objects from disk

    # Load individually trained models for comparison display
    lr_model       = joblib.load(ROOT_DIR / "models" / "lr_model.pkl")
    rf_model       = joblib.load(ROOT_DIR / "models" / "rf_model.pkl")
    ensemble_model = joblib.load(ROOT_DIR / "models" / "ensemble_model.pkl")

    # Load pre-built SHAP explainer (background fitted on training data)
    exp = FakeNewsExplainer()
    exp.load()

    return pp, fe, lr_model, rf_model, ensemble_model, exp


# Unpack all artifacts into module-level variables
pp, fe, lr_model, rf_model, ensemble_model, exp = load_artifacts()


def predict_all(text: str):
    # Step 1: Clean raw text using our TextPreprocessor
    clean = pp.clean(text)

    # Step 2: Extract meta features (word count, avg word len, etc.)
    meta  = pp.get_meta_features(text)

    # Step 3: Build a single-row DataFrame (FeatureExtractor expects a DataFrame)
    df    = pd.DataFrame([{"clean_text": clean, **meta}])

    # Step 4: Transform into combined TF-IDF + SBERT + meta feature vector
    X     = fe.transform(df)

    results = {}
    for name, mdl in [("Logistic Regression", lr_model),
                      ("Random Forest",        rf_model),
                      ("Ensemble",             ensemble_model)]:
        pred  = mdl.predict(X)[0]          # 0 = Real, 1 = Fake
        proba = mdl.predict_proba(X)[0]    # Probability for each class
        results[name] = {
            "label":      pred,
            "confidence": max(proba) * 100  # Highest class probability as %
        }

    # Step 5: Get SHAP word-level explanation for this article
    explanation = exp.explain(df, top_n=20)

    return results, explanation, clean


# ── UI Layout ──────────────────────────────────────────────
st.title("📰 Fake News Detector")
st.markdown("Paste a news article and click **Analyze**.")

# Two tabs: one for prediction, one for model metrics table
tab_predict, tab_compare = st.tabs(["Analyze Article", "Model Comparison"])

with tab_predict:

    # Side-by-side layout: input on left, result on right
    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        st.subheader("Input")
        news_text = st.text_area(
            "Paste news article",
            height=300,
            placeholder="Paste news content here..."
        )
        analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Result")

        if analyze_btn:
            if not news_text.strip():
                st.warning("Please enter some news text.")
                st.stop()  # Halt further execution if input is empty

            with st.spinner("Running models..."):
                try:
                    all_results, explanation, clean_text = predict_all(news_text)
                    ensemble_res = all_results["Ensemble"]

                    # Show green banner for Real, red banner for Fake
                    if ensemble_res["label"] == 0:
                        st.success("## ✅ REAL NEWS")
                    else:
                        st.error("## 🚨 FAKE NEWS")

                    # Show ensemble model's confidence percentage
                    st.metric(
                        "Ensemble Confidence",
                        f"{ensemble_res['confidence']:.1f}%"
                    )

                    # Show each model's individual vote
                    st.markdown("**Individual model votes:**")
                    for mname, res in all_results.items():
                        icon = "✅" if res["label"] == 0 else "🚨"
                        st.write(f"{icon} **{mname}** — {res['confidence']:.1f}%")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.stop()

    # SHAP section — full width, rendered below both columns
    if analyze_btn and news_text.strip():
        st.divider()
        st.subheader("Why did the model decide this?")
        st.caption(
            "🔴 Red = word pushed toward FAKE  |  "
            "🟢 Green = word pushed toward REAL  |  "
            "Darker = stronger signal"
        )

        # Render HTML with color-highlighted words based on SHAP values
        highlight_html = exp.build_highlight_html(clean_text, explanation)
        st.markdown(highlight_html, unsafe_allow_html=True)

        # Bar chart of top contributing words
        st.divider()
        st.subheader("Top contributing words")

        import plotly.graph_objects as go

        # Combine top fake-pushing + real-pushing words
        top_words = explanation["top_fake_words"][:8] + explanation["top_real_words"][:8]
        words_    = [w for w, _ in top_words]
        scores_   = [v for _, v in top_words]

        # Positive SHAP = fake signal (red), negative = real signal (green)
        colors_   = ["#dc3c3c" if s > 0 else "#1ea05a" for s in scores_]

        fig = go.Figure(go.Bar(
            x=scores_,
            y=words_,
            orientation="h",        # Horizontal bar chart
            marker_color=colors_,
            text=[f"{s:+.3f}" for s in scores_],
            textposition="outside"
        ))
        fig.update_layout(
            xaxis_title="SHAP value",
            yaxis=dict(autorange="reversed"),  # Highest magnitude word on top
            height=420,
            margin=dict(l=10, r=60, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",     # Transparent background
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_compare:
    st.subheader("Model performance comparison")
    st.caption("Evaluated on held-out test set.")

    metrics_path = ROOT_DIR / "models" / "model_metrics.pkl"
    if metrics_path.exists():
        metrics    = joblib.load(metrics_path)
        df_metrics = pd.DataFrame(metrics).T.round(3)

        # Highlight best value in each column with green background
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color="#d4edda"),
            use_container_width=True
        )
    else:
        # Guide user if models haven't been trained yet
        st.info("Train models first — then model_metrics.pkl will appear here.")