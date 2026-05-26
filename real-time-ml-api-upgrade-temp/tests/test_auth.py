import os
import sys
from importlib import import_module, reload
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_client() -> TestClient:
    os.environ["API_USERNAME"] = "ml-user"
    os.environ["API_PASSWORD"] = "ml-password"
    os.environ["JWT_SECRET_KEY"] = "auth-test-secret"
    from app.config import get_settings

    get_settings.cache_clear()
    main_module = import_module("app.main")
    reload(main_module)
    return TestClient(main_module.create_app())


def test_valid_login_returns_token() -> None:
    with build_client() as client:
        response = client.post(
            "/auth/token",
            data={"username": "ml-user", "password": "ml-password"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "access_token" in payload
    assert payload["token_type"] == "bearer"


def test_invalid_login_fails() -> None:
    with build_client() as client:
        response = client.post(
            "/auth/token",
            data={"username": "ml-user", "password": "wrong-password"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password."
