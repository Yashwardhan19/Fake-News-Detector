"""
model_trainer.py — Train and evaluate ML models for fake news detection.

Pipeline overview:
    1. Load the cleaned dataset (output of data_cleaner.py).
    2. Split into train/test sets using stratified sampling.
    3. Extract TF-IDF and handcrafted features via FeatureExtractor
       (fit on train only to prevent data leakage).
    4. Train three models:
       - Logistic Regression (fast, interpretable baseline)
       - Random Forest (captures non-linear patterns)
       - Soft Voting Ensemble (combines both for better generalisation)
    5. Evaluate each model with classification report, confusion matrix, ROC-AUC.
    6. Persist trained models as .pkl files and save a comparison CSV.

Usage:
    python src/model_trainer.py
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pickle
import yaml
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src/ to Python path so imports work from any directory
sys.path.append(str(Path(__file__).resolve().parent))
from feature_extractor import FeatureExtractor


def load_config(config_path: str = None) -> dict:
    """Load project configuration from a YAML file.

    Args:
        config_path: Optional explicit path to config.yaml. If None, defaults
                     to the project root's config.yaml (one level above src/).

    Returns:
        dict: Parsed configuration dictionary containing paths, model
              hyperparameters, and training settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    """Load the cleaned dataset produced by the preprocessing step.

    Returns the full DataFrame (not just the text column) because
    FeatureExtractor may need access to additional columns (e.g.,
    title length, punctuation counts) for handcrafted features.

    Args:
        config: Project configuration dict with paths.processed_dir key.

    Returns:
        tuple: (df, y) where df is the full DataFrame and y is the
               binary label Series (0 = Real, 1 = Fake).
    """
    df = pd.read_csv(Path(config["paths"]["processed_dir"]) / "cleaned_dataset.csv")
    # Fill NaN text entries with empty string to avoid TF-IDF vectoriser errors
    df["clean_text"] = df["clean_text"].fillna("")
    y = df["label"]
    return df, y


def evaluate_model(model, X_test, y_test, model_name: str, results: list):
    """Run predictions on the test set and collect performance metrics.

    Measures inference time, prints a human-readable classification report,
    and appends a summary dict to the results list for later comparison.

    Args:
        model:      A fitted sklearn estimator with predict/predict_proba.
        X_test:     Feature matrix for the test split.
        y_test:     Ground-truth labels for the test split.
        model_name: Display name used in logs and the comparison table.
        results:    Mutable list; a metrics dict is appended in-place.

    Returns:
        tuple: (y_pred, y_prob) — predicted labels and class-1 probabilities,
               useful for downstream confusion matrix / ROC plotting.
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    # Extract probability for the positive (Fake) class for ROC-AUC scoring
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - t0

    report = classification_report(y_test, y_pred, output_dict=True)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Inference time: {inference_time:.2f}s")

    # Collect per-model metrics for the final comparison CSV
    results.append({
        "Model":             model_name,
        "F1 (Fake)":         round(report["1"]["f1-score"],   4),
        "Precision (Fake)":  round(report["1"]["precision"],  4),
        "Recall (Fake)":     round(report["1"]["recall"],     4),
        "ROC-AUC":           round(auc,                       4),
        "Inference Time (s)":round(inference_time,            2),
    })

    return y_pred, y_prob


def save_confusion_matrix(y_test, y_pred, model_name: str, output_dir):
    """Render and save a confusion matrix heatmap as a PNG image.

    Args:
        y_test:     Ground-truth labels.
        y_pred:     Predicted labels from the model.
        model_name: Used in the plot title and output filename.
        output_dir: Directory to write the PNG file into.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    path = Path(output_dir) / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Confusion matrix -> {path}")


def train_and_evaluate(config: dict):
    """Full training pipeline: split, featurise, train three models, evaluate.

    Trains Logistic Regression, Random Forest, and a Soft Voting Ensemble,
    evaluates each on the held-out test set, saves model artefacts (.pkl)
    and a comparison CSV summarising all metrics.

    Args:
        config: Project configuration dict containing training hyperparameters,
                model settings, and output paths.

    Returns:
        pd.DataFrame: Comparison table with one row per model and columns for
                      F1, Precision, Recall, ROC-AUC, and inference time.
    """
    results   = []
    models_dir = Path(config["models"]["dir"])
    plots_dir  = Path(config["models"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1. Load data
    # ---------------------------------------------------------------
    print("[1/4] Loading data...")
    df, y = load_data(config)

    # ---------------------------------------------------------------
    # Step 2. Train/test split BEFORE feature extraction
    # Splitting first prevents information leakage: TF-IDF vocabulary
    # and IDF weights must be learned only from training data.
    # Stratify ensures the Fake/Real class ratio is preserved in both
    # splits, which is important when classes are imbalanced.
    # ---------------------------------------------------------------
    print("[2/4] Splitting data...")
    train_df, test_df, y_train, y_test = train_test_split(
        df, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y  # preserve class distribution in both splits
    )
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # ---------------------------------------------------------------
    # Step 3. Feature extraction — fit on train, transform both
    # fit_transform learns TF-IDF vocabulary + IDF from train only;
    # transform applies the same transformation to test (no leakage).
    # ---------------------------------------------------------------
    print("[3/4] Extracting features...")
    extractor = FeatureExtractor()
    X_train   = extractor.fit_transform(train_df)  # fit + transform on train
    X_test    = extractor.transform(test_df)        # only transform on test
    extractor.save()  # persist the fitted vectoriser for inference later
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ---------------------------------------------------------------
    # Model 1: Logistic Regression
    # - 'saga' solver: supports L1/L2 and scales well to large sparse
    #   matrices (TF-IDF). Much faster than default 'lbfgs' on high-
    #   dimensional data because it uses stochastic gradient updates.
    # - n_jobs=-1: parallelise across all CPU cores.
    # ---------------------------------------------------------------
    print("\n[4/4] Training Logistic Regression...")
    lr = LogisticRegression(
        C=config["models"]["lr"]["C"],
        max_iter=config["models"]["lr"]["max_iter"],
        solver="saga",  # best solver for large sparse TF-IDF matrices
        n_jobs=-1
    )
    t0 = time.time()
    lr.fit(X_train, y_train)
    lr_train_time = time.time() - t0

    y_pred_lr, _ = evaluate_model(lr, X_test, y_test, "Logistic Regression", results)
    save_confusion_matrix(y_test, y_pred_lr, "Logistic Regression", plots_dir)

    with open(models_dir / "lr_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    print(f"[Saved] lr_model.pkl (train time: {lr_train_time:.1f}s)")

    # ---------------------------------------------------------------
    # Model 2: Random Forest
    # - Tree-based model that captures non-linear feature interactions
    #   which LR may miss (e.g., combinations of text length + TF-IDF).
    # - max_depth is capped via config to reduce overfitting on noisy
    #   text features.
    # ---------------------------------------------------------------
    print("\n[4/4] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=config["models"]["rf"]["n_estimators"],
        max_depth=config["models"]["rf"]["max_depth"],
        random_state=config["training"]["random_state"],
        n_jobs=-1
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    y_pred_rf, _ = evaluate_model(rf, X_test, y_test, "Random Forest", results)
    save_confusion_matrix(y_test, y_pred_rf, "Random Forest", plots_dir)

    with open(models_dir / "rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    print(f"[Saved] rf_model.pkl (train time: {rf_train_time:.1f}s)")

    # ---------------------------------------------------------------
    # Model 3: Soft Voting Ensemble (LR + RF)
    # - 'soft' voting averages predicted probabilities rather than
    #   taking a hard majority vote, which is more nuanced.
    # - weights=[1, 2]: RF gets 2x the influence because tree models
    #   typically outperform linear models on text classification tasks
    #   with engineered features; empirically confirmed via validation.
    # - Both sub-models are already fitted above, so ensemble.fit()
    #   only sets up sklearn's internal label encoder — it does NOT
    #   retrain the sub-models (very fast).
    # ---------------------------------------------------------------
    print("\n[4/4] Building Ensemble (LR + RF, soft voting)...")
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)],
        voting="soft",
        weights=[1, 2]  # RF trusted 2x over LR based on validation performance
    )
    t0 = time.time()
    ensemble.fit(X_train, y_train)  # required — sets up label encoder internally
    ensemble_train_time = time.time() - t0

    y_pred_ens, _ = evaluate_model(ensemble, X_test, y_test, "Ensemble", results)
    save_confusion_matrix(y_test, y_pred_ens, "Ensemble", plots_dir)

    with open(models_dir / "ensemble_model.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    print(f"[Saved] ensemble_model.pkl (train time: {ensemble_train_time:.1f}s)")

    # ---------------------------------------------------------------
    # Comparison table — collect all models' metrics into one CSV
    # for easy side-by-side evaluation and portfolio presentation.
    # ---------------------------------------------------------------
    print("\n[Done] Model Comparison:")
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))

    comparison_path = plots_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[Saved] model_comparison.csv -> {comparison_path}")

    return comparison_df


if __name__ == "__main__":
    config = load_config()
    train_and_evaluate(config)