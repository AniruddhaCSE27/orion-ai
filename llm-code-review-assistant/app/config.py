from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")


class AnalyzerThresholds(BaseModel):
    """Configurable thresholds for AST-based review checks."""

    model_config = ConfigDict(frozen=True)

    long_function_lines: int = 40
    deep_nesting_level: int = 3
    too_many_parameters: int = 5


class Settings(BaseModel):
    """Runtime settings loaded from environment variables."""

    model_config = ConfigDict(frozen=True)

    github_token: Optional[str]
    openai_api_key: str
    openai_model: str
    github_api_base_url: str
    request_timeout_seconds: float
    max_files_for_review: int

    @property
    def prompt_path(self) -> Path:
        return PROJECT_ROOT / "prompts" / "review_prompt.txt"

    @property
    def thresholds(self) -> AnalyzerThresholds:
        return AnalyzerThresholds()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load cached application settings from local and parent .env files."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")

    return Settings(
        github_token=os.getenv("GITHUB_TOKEN"),
        openai_api_key=openai_api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        github_api_base_url=os.getenv("GITHUB_API_BASE_URL", "https://api.github.com"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
        max_files_for_review=int(os.getenv("MAX_FILES_FOR_REVIEW", "20")),
    )
