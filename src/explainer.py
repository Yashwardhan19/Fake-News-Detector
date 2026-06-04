import shap
import pickle
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from feature_extractor import FeatureExtractor


# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

MODELS_DIR = Path(CONFIG["models"]["dir"])
PLOTS_DIR = Path(CONFIG["models"]["plots_dir"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class FakeNewsExplainer:
    """
    SHAP-based explainer for the Logistic Regression model.
    Explains which words pushed the prediction toward Fake or Real.
    """

    def __init__(self):
        # Load fitted LR model from disk
        with open(MODELS_DIR / "lr_model.pkl", "rb") as f:
            self.lr_model = pickle.load(f)

        # Load fitted FeatureExtractor (tfidf + scaler already fitted)
        self.extractor = FeatureExtractor()
        self.extractor.load()

        # LinearExplainer works directly on LR coefficients — very fast
        # masker = mean of training data (background distribution)
        self.explainer = None
        self._feature_names = None

    def setup(self, X_train_sample: np.ndarray, feature_names: list):
        """
        Initialize SHAP explainer with a background sample.
        X_train_sample: small subset of training data (100-500 rows) as numpy array
        feature_names: list of all feature names (tfidf tokens + meta cols)
        """
        # shap.sample picks a random background — reduces compute significantly
        background = shap.sample(X_train_sample, 100)

        # LinearExplainer uses model coefficients directly - no need to run model repeatedly
        self.explainer = shap.LinearExplainer(self.lr_model, background)
        self._feature_names = feature_names
        print(f"[Explainer] Ready — {len(feature_names)} features")

    def explain(self, df_row: pd.DataFrame, top_n: int = 15) -> dict:
        """
        Explain a single prediction.
        df_row: one row DataFrame with clean_text + meta cols
        Returns dict with prediction, probability, and top contributing words
        """
        # Transform single row to feature matrix
        X = self.extractor.transform(df_row)

        # Get SHAP values — shape: (1, n_features)
        shap_values = self.explainer.shap_values(X)
        print(type(shap_values))

        if isinstance(shap_values, list):
            print("Class 0 shape:", shap_values[0].shape)
            print("Class 1 shape:", shap_values[1].shape)
        else:
            print("Shape:", shap_values.shape)

        # shap_values[0] = SHAP for class 0 (Real), shap_values[1] = class 1 (Fake)
        # We care about Fake class explanation
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # Fake class, first (only) row
        else:
            sv = shap_values[0]     # some versions return single array

        # Map feature name → shap value
        feature_shap = dict(zip(self._feature_names, sv))

        # Sort by absolute contribution — highest impact words first
        sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)

        # Split into fake-pushing (positive) and real-pushing (negative)
        fake_words = [(k, v) for k, v in sorted_features if v > 0][:top_n]
        real_words = [(k, v) for k, v in sorted_features if v < 0][:top_n]

        prob = self.lr_model.predict_proba(X)[0][1]  # probability of Fake
        pred = "FAKE" if prob >= 0.5 else "REAL"

        return {
            "prediction": pred,
            "fake_probability": round(float(prob), 4),
            "top_fake_words": fake_words,   # words pushing toward Fake
            "top_real_words": real_words,   # words pushing toward Real
        }

    def plot_explanation(self, explanation: dict, save_path: str = None):
        """
        Bar chart of top words — red = fake signal, green = real signal.
        This plot goes into Streamlit UI later.
        """
        fake_words = explanation["top_fake_words"][:10]
        real_words = explanation["top_real_words"][:10]

        # Combine and sort for clean visualization
        words = [w for w, _ in fake_words] + [w for w, _ in real_words]
        values = [v for _, v in fake_words] + [v for _, v in real_words]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]  # red=fake, green=real

        plt.figure(figsize=(10, 6))
        bars = plt.barh(words, values, color=colors)
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.xlabel("SHAP Value (contribution to Fake prediction)")
        plt.title(f"Prediction: {explanation['prediction']} "
                  f"({explanation['fake_probability']*100:.1f}% Fake)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"[Saved] Explanation plot → {save_path}")
        else:
            plt.show()
        plt.close()


# Quick test
if __name__ == "__main__":
    # Load dataset to get a background sample for SHAP
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    df["clean_text"] = df["clean_text"].fillna("")

    explainer = FakeNewsExplainer()

    # Get feature names and a small training sample for background
    feature_names = explainer.extractor.get_feature_names()
    X_sample = explainer.extractor.transform(df.sample(500, random_state=42))

    explainer.setup(X_sample, feature_names)

    # Test on one fake and one real article
    print("\n--- Testing on a FAKE article ---")
    fake_row = df[df["label"] == 1].iloc[[0]]
    result = explainer.explain(fake_row)
    print(f"Prediction: {result['prediction']} ({result['fake_probability']*100:.1f}% Fake)")
    print(f"Top Fake words: {result['top_fake_words'][:5]}")
    print(f"Top Real words: {result['top_real_words'][:5]}")
    explainer.plot_explanation(result, save_path=str(PLOTS_DIR / "shap_fake_example.png"))

    print("\n--- Testing on a REAL article ---")
    real_row = df[df["label"] == 0].iloc[[0]]
    result = explainer.explain(real_row)
    print(f"Prediction: {result['prediction']} ({result['fake_probability']*100:.1f}% Fake)")
    print(f"Top Fake words: {result['top_fake_words'][:5]}")
    print(f"Top Real words: {result['top_real_words'][:5]}")
    explainer.plot_explanation(result, save_path=str(PLOTS_DIR / "shap_real_example.png"))


#explainer.py model ke prediction ko explain karta hai.

# Ye batata hai ki article ke kaunse words Fake prediction ko support kar rahe hain aur kaunse words Real prediction ko support kar rahe hain.