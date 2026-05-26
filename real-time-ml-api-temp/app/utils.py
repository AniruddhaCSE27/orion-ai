import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


def load_model_artifact(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Load the serialized model artifact from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    artifact = joblib.load(model_path)
    required_keys = {"model", "feature_names", "target_names"}
    missing_keys = required_keys.difference(artifact.keys())
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required keys: {sorted(missing_keys)}"
        )

    logger.info("Loaded model artifact from %s", model_path)
    return artifact


def prepare_features(payload: Dict[str, float], artifact: Dict[str, Any]) -> pd.DataFrame:
    """Convert request payload into a model-ready dataframe."""
    ordered_values: List[float] = [
        payload["feature1"],
        payload["feature2"],
        payload["feature3"],
        payload["feature4"],
    ]
    return pd.DataFrame([ordered_values], columns=artifact["source_feature_names"])


def build_prediction_response(artifact: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
    """Generate a clean API response with class label and probability."""
    model = artifact["model"]
    prediction = int(model.predict(features)[0])
    prediction_label = artifact["target_names"][prediction]

    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probability": round(confidence, 6),
        "class_probabilities": {
            artifact["target_names"][index]: round(float(score), 6)
            for index, score in enumerate(probabilities)
        },
        "model_features": artifact["feature_names"],
    }
