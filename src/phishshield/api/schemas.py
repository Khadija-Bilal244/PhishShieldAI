from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


def clean_url(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("URL must not be empty")
    if "://" not in v:
        v = "http://" + v
    return v


class URLRequest(BaseModel):
    url: str = Field(..., max_length=2048, examples=["http://paypa1-secure-login.xyz/verify"])

    @field_validator("url")
    @classmethod
    def _check(cls, v: str) -> str:
        return clean_url(v)


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=1000)

    @field_validator("urls")
    @classmethod
    def _check(cls, v: list[str]) -> list[str]:
        return [clean_url(u) for u in v]


class Prediction(BaseModel):
    url: str
    phishing_probability: float
    prediction: str
    risk_level: str
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class FeedbackRequest(BaseModel):
    url: str = Field(..., max_length=2048)
    label: int = Field(..., ge=0, le=1, description="1 = phishing, 0 = legitimate")

    @field_validator("url")
    @classmethod
    def _check(cls, v: str) -> str:
        return clean_url(v)
