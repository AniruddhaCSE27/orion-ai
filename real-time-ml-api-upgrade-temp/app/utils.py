from typing import Any, Dict, List

import numpy as np
import pandas as pd


def prepare_features(payload: Dict[str, float], artifact: Dict[str, Any]) -> pd.DataFrame:
    ordered_values: List[float] = [
        payload["feature1"],
        payload["feature2"],
        payload["feature3"],
        payload["feature4"],
    ]
    return pd.DataFrame([ordered_values], columns=artifact["source_feature_names"])


def build_prediction_response(
    artifact: Dict[str, Any],
    payload: Dict[str, float],
    request_id: str,
) -> Dict[str, Any]:
    features = prepare_features(payload, artifact)
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
        "model_version": artifact["model_version"],
        "request_id": request_id,
    }
