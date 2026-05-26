import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


logger = logging.getLogger(__name__)


REQUIRED_ARTIFACT_KEYS = {
    "model",
    "model_version",
    "feature_names",
    "source_feature_names",
    "target_names",
    "metrics",
}


def load_model_artifact(model_path: str) -> Dict[str, Any]:
    artifact_path = Path(model_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

    artifact = joblib.load(artifact_path)
    missing_keys = REQUIRED_ARTIFACT_KEYS.difference(artifact.keys())
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required keys: {sorted(missing_keys)}"
        )

    logger.info(
        "Model artifact loaded successfully from %s with version=%s",
        artifact_path,
        artifact["model_version"],
    )
    return artifact


def is_model_ready(artifact: Optional[Dict[str, Any]]) -> bool:
    return artifact is not None
