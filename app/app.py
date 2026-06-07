"""
Streamlit web application for the Fake News Detector.

Provides a two-tab UI:
  1. **Analyze Article** - paste a news article, run three models (Logistic
     Regression, Random Forest, Ensemble), show SHAP word-level explanations,
     and perform credibility / fact-check lookups.
  2. **Model Comparison** - display held-out test-set metrics loaded from
     the CSV produced by model_trainer.py.

Launch with:  streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to Python path -- must be before src imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.preprocessor import TextPreprocessor
from src.feature_extractor import FeatureExtractor
from src.explainer import FakeNewsExplainer
from src.credibility_checker import CredibilityChecker

# Configure browser tab title, icon, and wide layout
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    """
    Load all ML artifacts once and cache them across Streamlit reruns.

    Returns:
        Tuple of (TextPreprocessor, FeatureExtractor, lr_model, rf_model,
        ensemble_model, FakeNewsExplainer, CredibilityChecker).
    """
    pp = TextPreprocessor()

    fe = FeatureExtractor()
    fe.load()

    # Load all three trained classifiers from disk
    lr_model       = joblib.load(ROOT_DIR / "models" / "lr_model.pkl")
    rf_model       = joblib.load(ROOT_DIR / "models" / "rf_model.pkl")
    ensemble_model = joblib.load(ROOT_DIR / "models" / "ensemble_model.pkl")

    # Explainer loads persisted SHAP background -- no raw data needed
    exp = FakeNewsExplainer()
    exp.load()

    # Credibility checker uses Google Fact Check API + source trust list
    checker = CredibilityChecker()

    return pp, fe, lr_model, rf_model, ensemble_model, exp, checker


pp, fe, lr_model, rf_model, ensemble_model, exp, checker = load_artifacts()


def predict_all(text: str):
    """
    Run all three models on a single article and generate SHAP explanation.

    Args:
        text: Raw news article text (not yet preprocessed).

    Returns:
        Tuple of (results dict, SHAP explanation dict, cleaned text string).
    """
    clean = pp.clean(text)
    meta  = pp.get_meta_features(text)
    # Build a single-row DataFrame matching the training schema
    df    = pd.DataFrame([{"clean_text": clean, **meta}])
    X     = fe.transform(df)

    results = {}
    for name, mdl in [("Logistic Regression", lr_model),
                      ("Random Forest",        rf_model),
                      ("Ensemble",             ensemble_model)]:
        pred  = mdl.predict(X)[0]
        proba = mdl.predict_proba(X)[0]
        results[name] = {
            "label":      pred,
            "confidence": max(proba) * 100
        }

    # SHAP explanation is always based on the LR model (most interpretable)
    explanation = exp.explain(df, top_n=20)
    return results, explanation, clean


# ── UI Layout ──────────────────────────────────────────────
st.title("📰 Fake News Detector")
st.markdown("Paste a news article and click **Analyze**.")

tab_predict, tab_compare = st.tabs(["Analyze Article", "Model Comparison"])

with tab_predict:

    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        st.subheader("Input")
        news_text = st.text_area(
            "Paste news article",
            height=250,
            placeholder="Paste news content here..."
        )
        # Optional source URL for credibility check
        source_url = st.text_input(
            "Source URL (optional)",
            placeholder="https://example.com/article"
        )
        analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Result")

        if analyze_btn:
            if not news_text.strip():
                st.warning("Please enter some news text.")
                st.stop()

            with st.spinner("Running models..."):
                try:
                    all_results, explanation, clean_text = predict_all(news_text)
                    # Ensemble is the primary prediction shown to the user
                    ensemble_res = all_results["Ensemble"]

                    if ensemble_res["label"] == 0:
                        st.success("## ✅ REAL NEWS")
                    else:
                        st.error("## 🚨 FAKE NEWS")

                    st.metric(
                        "Ensemble Confidence",
                        f"{ensemble_res['confidence']:.1f}%"
                    )

                    # Show how each individual model voted for transparency
                    st.markdown("**Individual model votes:**")
                    for mname, res in all_results.items():
                        icon = "✅" if res["label"] == 0 else "🚨"
                        st.write(f"{icon} **{mname}** — {res['confidence']:.1f}%")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.stop()

    # ── SHAP Section ──
    if analyze_btn and news_text.strip():
        st.divider()
        st.subheader("Why did the model decide this?")
        st.caption(
            "🔴 Red = word pushed toward FAKE  |  "
            "🟢 Green = word pushed toward REAL  |  "
            "Darker = stronger signal"
        )

        # Render color-coded article text using SHAP values
        highlight_html = exp.build_highlight_html(clean_text, explanation)
        st.markdown(highlight_html, unsafe_allow_html=True)

        st.divider()
        st.subheader("Top contributing words")

        import plotly.graph_objects as go

        # Combine top fake + real words into one horizontal bar chart
        top_words = explanation["top_fake_words"][:8] + explanation["top_real_words"][:8]
        words_    = [w for w, _ in top_words]
        scores_   = [v for _, v in top_words]
        colors_   = ["#dc3c3c" if s > 0 else "#1ea05a" for s in scores_]

        fig = go.Figure(go.Bar(
            x=scores_,
            y=words_,
            orientation="h",
            marker_color=colors_,
            text=[f"{s:+.3f}" for s in scores_],
            textposition="outside"
        ))
        fig.update_layout(
            xaxis_title="SHAP value",
            yaxis=dict(autorange="reversed"),
            height=420,
            margin=dict(l=10, r=60, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Credibility Section ──
        st.divider()
        st.subheader("🔍 Credibility Check")

        with st.spinner("Checking fact databases..."):
            cred_result = checker.full_check(news_text, source_url)

        # Source trust score (only shown if user provided a URL)
        if source_url:
            src = cred_result["source_check"]
            if src and src["domain"]:
                score_text = f" (Trust score: {src['score']})" if src["score"] else ""
                st.markdown(f"**Source:** `{src['domain']}` — {src['label']}{score_text}")

        # Fact check results from Google Fact Check API
        fact = cred_result["fact_check"]
        if fact["found"]:
            st.warning("⚠️ Related fact-checks found in database:")
            for r in fact["results"]:
                st.markdown(f"""
**Claim:** {r['claim'][:120]}  
**Rating:** `{r['rating']}` — *{r['source']}*  
[Read more]({r['url']})

---
""")
        else:
            st.info("ℹ️ No matching fact-checks found in database.")


with tab_compare:
    st.subheader("Model performance comparison")
    st.caption("Evaluated on held-out test set.")

    # model_trainer.py saves metrics as model_comparison.csv, not a pickle
    metrics_path = ROOT_DIR / "models" / "plots" / "model_comparison.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color="#d4edda"),
            use_container_width=True
        )
    else:
        st.info("Train models first — then model_comparison.csv will appear here.")