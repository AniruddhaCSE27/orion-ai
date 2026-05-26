from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Validated input payload for real-time inference."""

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

    feature1: float = Field(..., description="Mean radius-like feature value.")
    feature2: float = Field(..., description="Mean texture-like feature value.")
    feature3: float = Field(..., description="Mean perimeter-like feature value.")
    feature4: float = Field(..., description="Mean area-like feature value.")

