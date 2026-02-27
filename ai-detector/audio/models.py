"""Pydantic models for audio analysis request/response validation."""

from typing import Dict

from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    """Audio file metadata."""

    duration: float
    sample_rate: int
    channels: int
    format: str


class AudioAnalysisResult(BaseModel):
    """Audio analysis result model."""

    filename: str
    is_deepfake: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    deepfake_probability: float = Field(..., ge=0.0, le=1.0)
    authentic_probability: float = Field(..., ge=0.0, le=1.0)
    features: Dict[str, float]
    metadata: AudioMetadata
    message: str
