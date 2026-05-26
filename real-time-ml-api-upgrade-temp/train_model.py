from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
MODEL_VERSION = "2026.04.16"

SELECTED_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
]


def train_and_save_model() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.frame[SELECTED_FEATURES]
    target = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    predicted_labels = pipeline.predict(x_test)
    predicted_probabilities = pipeline.predict_proba(x_test)[:, 1]

    artifact = {
        "model": pipeline,
        "model_version": MODEL_VERSION,
        "feature_names": ["feature1", "feature2", "feature3", "feature4"],
        "source_feature_names": SELECTED_FEATURES,
        "target_names": list(dataset.target_names),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predicted_labels)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, predicted_probabilities)), 4),
            "f1_score": round(float(f1_score(y_test, predicted_labels)), 4),
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Model trained and saved to: {MODEL_PATH}")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Metrics: {artifact['metrics']}")


if __name__ == "__main__":
    train_and_save_model()

