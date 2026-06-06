import shap
import pickle
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from src.feature_extractor import FeatureExtractor


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

MODELS_DIR = Path(CONFIG["models"]["dir"])
PLOTS_DIR  = Path(CONFIG["models"]["plots_dir"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class FakeNewsExplainer:
    """
    SHAP-based explainer for the Logistic Regression model.
    LR was trained on 50,010 features (50,000 TF-IDF + 10 meta).
    SHAP runs on full 50,010 features, but word highlights use TF-IDF slice only.
    """

    def __init__(self):
        # Load fitted LR model from disk
        with open(MODELS_DIR / "lr_model.pkl", "rb") as f:
            self.lr_model = pickle.load(f)

        # Load fitted FeatureExtractor (tfidf + scaler already on disk)
        self.extractor = FeatureExtractor()
        self.extractor.load()

        self.explainer      = None
        self._feature_names = None  # TF-IDF vocab only (50,000 words)

    def setup(self, df_sample: pd.DataFrame):
        """
        Build SHAP LinearExplainer using a background sample from training data.
        df_sample: sample DataFrame with clean_text + meta cols (raw df, not pre-transformed).
        LR model was trained on full 50,010 features, so background must match shape.
        """
        # Full feature matrix — must match LR training shape (50,010)
        X_full = self.extractor.transform(df_sample)

        # shap.sample picks a random subset — keeps memory manageable
        background = shap.sample(X_full, min(200, X_full.shape[0]))

        # LinearExplainer uses LR coefficients directly — fast on CPU
        self.explainer      = shap.LinearExplainer(self.lr_model, background)
        self._feature_names = self.extractor.tfidf.get_feature_names_out().tolist()
        print(f"[Explainer] Ready — background shape: {background.shape}")

    def save_background(self):
        """
        Save fitted explainer to disk so Streamlit can load without raw data.
        Must call setup() before this.
        """
        if self.explainer is None:
            raise RuntimeError("Call setup() before save_background()")
        with open(MODELS_DIR / "shap_explainer.pkl", "wb") as f:
            pickle.dump(self.explainer, f)
        print(f"[Explainer] Saved → {MODELS_DIR / 'shap_explainer.pkl'}")

    def load(self):
        """
        Load saved explainer from disk — used in Streamlit app at inference time.
        No raw training data needed.
        """
        with open(MODELS_DIR / "shap_explainer.pkl", "rb") as f:
            self.explainer = pickle.load(f)
        self._feature_names = self.extractor.tfidf.get_feature_names_out().tolist()
        print("[Explainer] Loaded from disk")

    def explain(self, df_row: pd.DataFrame, top_n: int = 15) -> dict:
        """
        Explain a single prediction — returns top contributing words.
        df_row: single-row DataFrame with clean_text + meta cols.
        """
        if self.explainer is None:
            raise RuntimeError("Call setup() or load() before explain()")

        # Full features for SHAP — must match LR training shape (50,010)
        X_full    = self.extractor.transform(df_row)
        shap_vals = self.explainer.shap_values(X_full)

        # Handle old SHAP (returns list per class) vs new SHAP (returns single array)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]   # index 1 = Fake class, index 0 = first row
        else:
            sv = shap_vals[0]      # new versions return single array

        # Slice only TF-IDF portion — first 50,000 are words, last 10 are meta
        n_tfidf  = len(self._feature_names)   # 50,000
        sv_tfidf = sv[:n_tfidf]               # drop meta SHAP values, keep words only

        # Filter only words actually present in this article (non-zero TF-IDF score)
        X_tfidf_dense = np.asarray(self.extractor.get_tfidf_only(df_row).todense())[0]
        word_shap = [
            (self._feature_names[i], float(sv_tfidf[i]))
            for i in range(n_tfidf)
            if X_tfidf_dense[i] != 0.0   # word must appear in this article
        ]

        # Sort by absolute contribution — strongest signals first
        word_shap.sort(key=lambda x: abs(x[1]), reverse=True)

        fake_words = [(w, v) for w, v in word_shap if v > 0][:top_n]
        real_words = [(w, v) for w, v in word_shap if v < 0][:top_n]

        prob = self.lr_model.predict_proba(X_full)[0][1]   # P(Fake)
        pred = "FAKE" if prob >= 0.5 else "REAL"

        return {
            "prediction":       pred,
            "fake_probability": round(float(prob), 4),
            "top_fake_words":   fake_words,    # words pushing toward Fake
            "top_real_words":   real_words,    # words pushing toward Real
            "all_word_shap":    word_shap,     # used by build_highlight_html()
        }

    def build_highlight_html(self, clean_text: str, explanation: dict) -> str:
        """
        Returns HTML with words color-coded by SHAP value.
        Red = pushing toward FAKE, Green = pushing toward REAL.
        Plug directly into st.markdown(..., unsafe_allow_html=True).
        """
        word_map = {w: v for w, v in explanation["all_word_shap"]}

        if not word_map:
            return f'<div style="line-height:2.4;font-size:15px;">{clean_text}</div>'

        max_abs = max(abs(v) for v in word_map.values()) or 1.0
        tokens  = clean_text.split()
        parts   = []

        for token in tokens:
            if token in word_map:
                score = word_map[token]
                alpha = round(0.15 + (abs(score) / max_abs) * 0.6, 2)
                bg    = (
                    f"rgba(220,60,60,{alpha})"    # red   → FAKE signal
                    if score > 0 else
                    f"rgba(30,160,90,{alpha})"    # green → REAL signal
                )
                parts.append(
                    f'<span style="background:{bg};padding:2px 5px;'
                    f'border-radius:4px;margin:1px;display:inline-block;">'
                    f'{token}</span>'
                )
            else:
                parts.append(
                    f'<span style="margin:1px;display:inline-block;">{token}</span>'
                )

        return '<div style="line-height:2.4;font-size:15px;">' + " ".join(parts) + "</div>"

    def plot_explanation(self, explanation: dict, save_path: str = None):
        """
        Horizontal bar chart of top words.
        Red bars = fake signal, green bars = real signal.
        Saves PNG to models/plots/ or shows inline.
        """
        fake_words = explanation["top_fake_words"][:10]
        real_words = explanation["top_real_words"][:10]

        words  = [w for w, _ in fake_words] + [w for w, _ in real_words]
        values = [v for _, v in fake_words] + [v for _, v in real_words]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        plt.figure(figsize=(10, 6))
        plt.barh(words, values, color=colors)
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.xlabel("SHAP value  (positive = FAKE signal, negative = REAL signal)")
        plt.title(
            f"Prediction: {explanation['prediction']}  "
            f"({explanation['fake_probability']*100:.1f}% Fake probability)"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"[Saved] → {save_path}")
        else:
            plt.show()
        plt.close()


# Quick test
if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    df["clean_text"] = df["clean_text"].fillna("")

    exp = FakeNewsExplainer()

    # Pass raw df — transform happens inside setup()
    exp.setup(df.sample(500, random_state=42))
    exp.save_background()

    print("\n--- FAKE article ---")
    fake_row = df[df["label"] == 1].iloc[[0]]
    result   = exp.explain(fake_row)
    print(f"Prediction : {result['prediction']}  ({result['fake_probability']*100:.1f}%)")
    print(f"Top FAKE words : {result['top_fake_words'][:5]}")
    print(f"Top REAL words : {result['top_real_words'][:5]}")
    exp.plot_explanation(result, save_path=str(PLOTS_DIR / "shap_fake_example.png"))

    print("\n--- REAL article ---")
    real_row = df[df["label"] == 0].iloc[[0]]
    result   = exp.explain(real_row)
    print(f"Prediction : {result['prediction']}  ({result['fake_probability']*100:.1f}%)")
    print(f"Top FAKE words : {result['top_fake_words'][:5]}")
    print(f"Top REAL words : {result['top_real_words'][:5]}")
    exp.plot_explanation(result, save_path=str(PLOTS_DIR / "shap_real_example.png"))