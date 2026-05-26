import os
import sys
from importlib import import_module, reload
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_client() -> TestClient:
    os.environ["API_USERNAME"] = "predict-user"
    os.environ["API_PASSWORD"] = "predict-password"
    os.environ["JWT_SECRET_KEY"] = "predict-secret"
    from app.config import get_settings

    get_settings.cache_clear()
    main_module = import_module("app.main")
    reload(main_module)
    return TestClient(main_module.create_app())


def fetch_token(client: TestClient) -> str:
    response = client.post(
        "/auth/token",
        data={"username": "predict-user", "password": "predict-password"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_authenticated_prediction_succeeds() -> None:
    with build_client() as client:
        token = fetch_token(client)
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {token}"},
            json={"feature1": 10, "feature2": 5, "feature3": 2, "feature4": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "prediction" in payload
    assert "request_id" in payload
    assert "model_version" in payload


def test_unauthenticated_prediction_fails() -> None:
    with build_client() as client:
        response = client.post(
            "/predict",
            json={"feature1": 10, "feature2": 5, "feature3": 2, "feature4": 1},
        )

    assert response.status_code == 401


def test_invalid_payload_returns_validation_error() -> None:
    with build_client() as client:
        token = fetch_token(client)
        response = client.post(
            "/predict",
            headers={"Authorization": f"Bearer {token}"},
            json={"feature1": 10, "feature2": 5},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid request payload."
