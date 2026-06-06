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
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    """Load cleaned dataset — returns full df so FeatureExtractor can access all columns."""
    df = pd.read_csv(Path(config["paths"]["processed_dir"]) / "cleaned_dataset.csv")
    df["clean_text"] = df["clean_text"].fillna("")
    y = df["label"]
    return df, y


def evaluate_model(model, X_test, y_test, model_name: str, results: list):
    """Run predictions, measure inference time, print report, append to results."""
    t0 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Fake class probability
    inference_time = time.time() - t0

    report = classification_report(y_test, y_pred, output_dict=True)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Inference time: {inference_time:.2f}s")

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
    """Save confusion matrix PNG for portfolio."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    path = Path(output_dir) / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Confusion matrix → {path}")


def train_and_evaluate(config: dict):
    results   = []
    models_dir = Path(config["models"]["dir"])
    plots_dir  = Path(config["models"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    print("[1/4] Loading data...")
    df, y = load_data(config)

    # --- 2. Split FIRST — then extract features (no leakage) ---
    print("[2/4] Splitting data...")
    train_df, test_df, y_train, y_test = train_test_split(
        df, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # --- 3. Extract features — fit only on train ---
    print("[3/4] Extracting features...")
    extractor = FeatureExtractor()
    X_train   = extractor.fit_transform(train_df)  # fit + transform on train
    X_test    = extractor.transform(test_df)        # only transform on test
    extractor.save()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- Model 1: Logistic Regression ---
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

    # --- Model 2: Random Forest ---
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

    # --- Model 3: Soft Voting Ensemble ---
    # LR + RF already fitted — ensemble.fit() is fast (no retraining)
    print("\n[4/4] Building Ensemble (LR + RF, soft voting)...")
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)],
        voting="soft",
        weights=[1, 2]  # RF trusted 2x over LR
    )
    t0 = time.time()
    ensemble.fit(X_train, y_train)  # required — sets up label encoder internally
    ensemble_train_time = time.time() - t0

    y_pred_ens, _ = evaluate_model(ensemble, X_test, y_test, "Ensemble", results)
    save_confusion_matrix(y_test, y_pred_ens, "Ensemble", plots_dir)

    with open(models_dir / "ensemble_model.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    print(f"[Saved] ensemble_model.pkl (train time: {ensemble_train_time:.1f}s)")

    # --- Model Comparison Table ---
    print("\n[Done] Model Comparison:")
    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))

    comparison_path = plots_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[Saved] model_comparison.csv → {comparison_path}")

    return comparison_df


if __name__ == "__main__":
    config = load_config()
    train_and_evaluate(config)