import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

MODELS_DIR = Path(CONFIG["paths"]["models_dir"])
MODELS_DIR.mkdir(exist_ok=True)

TFIDF_CFG   = CONFIG["features"]["tfidf"]
META_COLS   = CONFIG["features"]["meta"]["columns"]
SBERT_MODEL = "all-MiniLM-L6-v2"


class FeatureExtractor:
    """
    Converts cleaned text → numeric feature matrix.
    Feature types:
      - Without SBERT: TF-IDF (50,000) + Meta (10) = 50,010
      - With SBERT:    TF-IDF (50,000) + SBERT (384) + Meta (10) = 50,394
    SBERT is optional — app works without it.
    """

    def __init__(self):
        self.tfidf  = TfidfVectorizer(
            max_features = TFIDF_CFG["max_features"],
            ngram_range  = tuple(TFIDF_CFG["ngram_range"]),
            sublinear_tf = TFIDF_CFG["sublinear_tf"],
            min_df       = TFIDF_CFG["min_df"],
            max_df       = TFIDF_CFG["max_df"],
        )
        self.scaler  = StandardScaler()
        self.sbert   = None   # loaded only if sbert_model_name.pkl exists
        self._fitted = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit TF-IDF + scaler on training data only.
        SBERT is pretrained — no fitting needed.
        Never call on test/val data — causes data leakage.
        """
        self.tfidf.fit(df["clean_text"])

        meta = self._get_meta_matrix(df)
        self.scaler.fit(meta)

        self._fitted = True
        print(f"[FeatureExtractor] Fitted — vocab size: {len(self.tfidf.vocabulary_)}")

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Transform df → combined feature matrix.
        SBERT included only if available — otherwise TF-IDF + meta only.
        """
        self._check_fitted()

        tfidf_matrix = self.tfidf.transform(df["clean_text"])      # (n, 50000)
        meta_matrix  = self._get_meta_matrix(df)
        meta_scaled  = self.scaler.transform(meta_matrix)          # (n, 10)

        if self.sbert is not None:
            # SBERT available — full feature set (50,394)
            sbert_matrix = self.sbert.encode(
                df["clean_text"].tolist(),
                batch_size=64,
                show_progress_bar=False,
            )
            combined = sp.hstack([
                tfidf_matrix,
                sp.csr_matrix(sbert_matrix),
                sp.csr_matrix(meta_scaled)
            ])
        else:
            # SBERT not available — TF-IDF + meta only (50,010)
            combined = sp.hstack([
                tfidf_matrix,
                sp.csr_matrix(meta_scaled)
            ])

        return combined

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Convenience method — fit then transform in one call (training set only)."""
        self.fit(df)
        return self.transform(df)

    def get_tfidf_only(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Returns TF-IDF matrix only — no SBERT, no meta.
        Used by SHAP LinearExplainer for word-level explanation.
        """
        self._check_fitted()
        return self.tfidf.transform(df["clean_text"])

    def get_feature_names(self) -> list[str]:
        """Returns all feature names in same order as transform() output."""
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        if self.sbert is not None:
            sbert_names = [f"sbert_{i}" for i in range(384)]
            return tfidf_names + sbert_names + META_COLS
        return tfidf_names + META_COLS

    def save(self) -> None:
        """Persist fitted TF-IDF + scaler. Save SBERT model name if loaded."""
        with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(MODELS_DIR / "meta_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        if self.sbert is not None:
            with open(MODELS_DIR / "sbert_model_name.pkl", "wb") as f:
                pickle.dump(SBERT_MODEL, f)
        print(f"[FeatureExtractor] Saved -> {MODELS_DIR}")

    def load(self) -> None:
        """
        Load TF-IDF + scaler from disk.
        SBERT loaded only if sbert_model_name.pkl exists — optional.
        """
        with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        with open(MODELS_DIR / "meta_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        sbert_path = MODELS_DIR / "sbert_model_name.pkl"
        if sbert_path.exists():
            # Import only if needed — avoids slow import when SBERT unused
            from sentence_transformers import SentenceTransformer
            with open(sbert_path, "rb") as f:
                model_name = pickle.load(f)
            self.sbert = SentenceTransformer(model_name)
            print("[FeatureExtractor] SBERT loaded")
        else:
            self.sbert = None
            print("[FeatureExtractor] SBERT not found — using TF-IDF + meta only")

        self._fitted = True
        print("[FeatureExtractor] Loaded from disk")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_meta_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract meta columns as numpy array in consistent order."""
        return df[META_COLS].values.astype(np.float32)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before transform().")


# Quick test
if __name__ == "__main__":
    import gc

    df = pd.read_csv("data/processed/cleaned_dataset.csv")

    # Sirf zaruri columns rakho — RAM bachao
    df = df[["clean_text", "label"] + META_COLS].copy()
    gc.collect()

    print(f"Rows loaded: {len(df)}")

    fe = FeatureExtractor()
    X  = fe.fit_transform(df)
    fe.save()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Total feature names: {len(fe.get_feature_names())}")

    # Without SBERT: (44182, 50010)
    # With SBERT:    (44182, 50394)