
import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer#tfidf vectorizer converts text into a matrix of TF-IDF features, which are numerical representations of the importance of words in the documents.
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "models" / "config.yaml"#in this way, we can load config.yaml from anywhere (train.py, app.py) and it will always resolve to the correct path relative to this file
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)#load config.yaml into a dict, so we can access all our settings from there .

MODELS_DIR = Path(CONFIG["paths"]["models_dir"])#
MODELS_DIR.mkdir(exist_ok=True)

TFIDF_CFG = CONFIG["features"]["tfidf"]
META_COLS  = CONFIG["features"]["meta"]["columns"]


class FeatureExtractor:
    """
    Converts cleaned text → numeric feature matrix.
    Produces 3 feature types:
      1. TF-IDF  (sparse, 50k features)
      2. Meta    (10 hand-crafted numeric features)
      3. Combined (TF-IDF + scaled meta, horizontally stacked)
    """

    def __init__(self):
        # ngram_range [1,2] means unigrams + bigrams — "fake news" as one token beats "fake" alone
        self.tfidf = TfidfVectorizer(
            max_features = TFIDF_CFG["max_features"],
            ngram_range  = tuple(TFIDF_CFG["ngram_range"]),
            sublinear_tf = TFIDF_CFG["sublinear_tf"],
            min_df       = TFIDF_CFG["min_df"],    # ignore tokens appearing in <2 docs (typos, noise)
            max_df       = TFIDF_CFG["max_df"],    # ignore tokens in >95% docs (too common to be useful)
        )
        # StandardScaler normalises meta features to mean=0, std=1
        # Required because TF-IDF values are tiny (0.0–0.9) but char_count can be 5000+
        self.scaler = StandardScaler()
        self._fitted = False  # guard so transform() can't be called before fit()

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit TF-IDF + scaler on training data only.
        Never call this on test/val data — that causes data leakage.
        """
        self.tfidf.fit(df["clean_text"])

        meta = self._get_meta_matrix(df)
        self.scaler.fit(meta)   # learn mean + std from train set only

        self._fitted = True
        print(f"[FeatureExtractor] Fitted — vocab size: {len(self.tfidf.vocabulary_)}")

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Transform df → combined feature matrix (TF-IDF + scaled meta).
        Returns a scipy sparse matrix ready to pass into sklearn models.
        """
        self._check_fitted()

        tfidf_matrix = self.tfidf.transform(df["clean_text"])   # shape: (n, 50000)
        meta_matrix  = self._get_meta_matrix(df)
        meta_scaled  = self.scaler.transform(meta_matrix)          # shape: (n, 10)

        # hstack joins sparse + dense horizontally → (n, 50010)
        # convert meta to sparse first so hstack works uniformly
        combined = sp.hstack([tfidf_matrix, sp.csr_matrix(meta_scaled)])
        return combined

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Convenience method — fit then transform in one call (for training set)."""
        self.fit(df)
        return self.transform(df)

    def get_tfidf_only(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Returns TF-IDF matrix without meta features.
        Used by SHAP LinearExplainer — SHAP needs feature names,
        and meta feature names would confuse the word-level explanation.
        """
        self._check_fitted()
        return self.tfidf.transform(df["clean_text"])

    def get_feature_names(self) -> list[str]:
        """Returns all feature names — TF-IDF tokens + meta column names."""
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        return tfidf_names + META_COLS

    def save(self) -> None:
        """Persist fitted vectorizer + scaler to disk so app can load them."""
        with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(MODELS_DIR / "meta_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[FeatureExtractor] Saved → {MODELS_DIR}")

    def load(self) -> None:
        """Load persisted vectorizer + scaler (used in Streamlit app at inference time)."""
        with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf = pickle.load(f) 
        with open(MODELS_DIR / "meta_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self._fitted = True
        print("[FeatureExtractor] Loaded from disk")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_meta_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the 10 meta columns as a numpy array in consistent order."""
        # META_COLS order must match what get_meta_features() returns in preprocessor.py
        return df[META_COLS].values.astype(np.float32)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before transform().")

        

# Quick test
if __name__ == "__main__":
    import pandas as pd

    # load preprocessed dataset
    df = pd.read_csv("data/processed/cleaned_dataset.csv")

    print(f"Rows loaded: {len(df)}")

    fe = FeatureExtractor()

    # Fit + transform
    X = fe.fit_transform(df)

    # Save fitted vectorizer and scaler
    fe.save()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Total feature names: {len(fe.get_feature_names())}")

    # Expected:
    # [FeatureExtractor] Fitted — vocab size: 50000
    # [FeatureExtractor] Saved → models
    # Feature matrix shape: (44182, 50010)
    