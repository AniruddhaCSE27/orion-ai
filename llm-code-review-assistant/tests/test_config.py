from __future__ import annotations

from app.config import get_settings


def test_get_settings_requires_openai_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        get_settings()
    except ValueError as exc:
        assert "OPENAI_API_KEY is required" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected get_settings() to reject missing OPENAI_API_KEY")
    finally:
        get_settings.cache_clear()


def test_settings_report_missing_optional_integrations(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.missing_optional_integrations() == [
        "GITHUB_TOKEN is not set; GitHub API rate limits will be much lower."
    ]
    get_settings.cache_clear()
