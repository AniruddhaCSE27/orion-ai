from functools import lru_cache
from typing import List, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Real-Time ML Prediction API")
    app_version: str = Field(default="2.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    api_username: str = Field(default="admin")
    api_password: str = Field(default="change-me-in-production")

    jwt_secret_key: str = Field(default="change-this-jwt-secret-in-production")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

    auth_rate_limit: str = Field(default="5/minute")
    predict_rate_limit: str = Field(default="20/minute")

    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    trusted_hosts: List[str] = Field(default_factory=lambda: ["*"])

    model_path: str = Field(default="models/model.pkl")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("allowed_origins", "trusted_hosts", mode="before")
    @classmethod
    def parse_comma_separated_values(
        cls, value: Union[str, List[str]]
    ) -> List[str]:
        if isinstance(value, list):
            return value
        if not value:
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
