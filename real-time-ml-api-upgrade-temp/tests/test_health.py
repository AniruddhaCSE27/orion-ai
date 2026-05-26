import os
import sys
from importlib import import_module, reload
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_client() -> TestClient:
    os.environ["API_USERNAME"] = "test-user"
    os.environ["API_PASSWORD"] = "test-password"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key"
    os.environ["ENVIRONMENT"] = "development"
    from app.config import get_settings

    get_settings.cache_clear()
    main_module = import_module("app.main")
    reload(main_module)
    return TestClient(main_module.create_app())


def test_health_live() -> None:
    with build_client() as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service_name"] == "Real-Time ML Prediction API"


def test_health_ready() -> None:
    with build_client() as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
