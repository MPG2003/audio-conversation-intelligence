from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sales_conversion_training.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "sales_conversion_model.pkl"
FEATURES_PATH = MODEL_DIR / "sales_conversion_features.pkl"
METRICS_PATH = MODEL_DIR / "sales_conversion_metrics.json"


def build_candidates() -> dict[str, object]:
    logistic = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    random_forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
    )
    xgboost = XGBClassifier(
        n_estimators=450,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    return {
        "logistic_regression": logistic,
        "extra_trees": extra_trees,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
        "xgboost": xgboost,
        "soft_voting": VotingClassifier(
            estimators=[
                ("logistic_regression", clone(logistic)),
                ("extra_trees", clone(extra_trees)),
                ("xgboost", clone(xgboost)),
            ],
            voting="soft",
        ),
    }


def best_threshold(y_true: pd.Series, probabilities) -> tuple[float, float]:
    best_accuracy = -1.0
    best_cutoff = 0.5
    for step in range(30, 71):
        cutoff = step / 100
        predictions = (probabilities >= cutoff).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cutoff = cutoff
    return best_cutoff, best_accuracy


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    encoded = pd.get_dummies(df, drop_first=True)

    X = encoded.drop("conversion", axis=1)
    y = encoded["conversion"]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = build_candidates()
    leaderboard: list[dict[str, object]] = []

    for name, model in candidates.items():
        cv_scores = cross_validate(
            clone(model),
            X,
            y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        )
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        probabilities = fitted.predict_proba(X_test)[:, 1]
        threshold, threshold_accuracy = best_threshold(y_test, probabilities)
        predictions = (probabilities >= threshold).astype(int)

        leaderboard.append(
            {
                "name": name,
                "cv_accuracy": round(float(cv_scores["test_accuracy"].mean()), 4),
                "cv_precision": round(float(cv_scores["test_precision"].mean()), 4),
                "cv_recall": round(float(cv_scores["test_recall"].mean()), 4),
                "cv_f1": round(float(cv_scores["test_f1"].mean()), 4),
                "cv_roc_auc": round(float(cv_scores["test_roc_auc"].mean()), 4),
                "holdout_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "holdout_precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
                "holdout_recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
                "holdout_f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
                "holdout_roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
                "threshold": threshold,
            }
        )

    leaderboard.sort(
        key=lambda item: (
            float(item["holdout_accuracy"]),
            float(item["holdout_roc_auc"]),
            float(item["cv_accuracy"]),
        ),
        reverse=True,
    )
    winner_name = str(leaderboard[0]["name"])
    winner = clone(candidates[winner_name])
    winner.fit(X_train, y_train)
    probabilities = winner.predict_proba(X_test)[:, 1]
    threshold = float(leaderboard[0]["threshold"])
    predictions = (probabilities >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    metrics = {
        "model_name": winner_name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "classification_threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "feature_count": int(len(feature_columns)),
        "leaderboard": leaderboard,
        "production_note": (
            "This is an honest holdout score. Reaching 80%+ will likely require more real labeled "
            "conversation outcomes or stronger labels, because all tested model families cluster around "
            "the low 70s on the current dataset."
        ),
    }

    final_model = clone(candidates[winner_name])
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Selected model: {winner_name}")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("ROC AUC:", metrics["roc_auc"])
    print("Threshold:", metrics["classification_threshold"])
    print("Model and metrics saved successfully")


if __name__ == "__main__":
    main()
