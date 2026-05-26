from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

SELECTED_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
]


def train_and_save_model() -> None:
    dataset = load_breast_cancer(as_frame=True)
    dataframe = dataset.frame[SELECTED_FEATURES]
    target = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe,
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

    test_predictions = pipeline.predict(x_test)
    test_probabilities = pipeline.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, test_predictions)
    roc_auc = roc_auc_score(y_test, test_probabilities)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": pipeline,
        "feature_names": [
            "feature1",
            "feature2",
            "feature3",
            "feature4",
        ],
        "source_feature_names": SELECTED_FEATURES,
        "target_names": list(dataset.target_names),
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "roc_auc": round(float(roc_auc), 4),
        },
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"Model trained and saved to: {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    train_and_save_model()
