from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "feature1": 10.0,
                "feature2": 5.0,
                "feature3": 2.0,
                "feature4": 1.0,
            }
        },
    )

    feature1: float = Field(..., description="Feature 1 numeric input.")
    feature2: float = Field(..., description="Feature 2 numeric input.")
    feature3: float = Field(..., description="Feature 3 numeric input.")
    feature4: float = Field(..., description="Feature 4 numeric input.")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    class_probabilities: Dict[str, float]
    model_features: List[str]
    model_version: str
    request_id: str


class ServiceMetadataResponse(BaseModel):
    service_name: str
    version: str
    environment: str
    docs_enabled: bool


class HealthResponse(BaseModel):
    status: str
    service_name: str
    version: str
    environment: str
    model_loaded: Optional[bool] = None
