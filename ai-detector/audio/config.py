"""Configuration settings for the audio deepfake detector."""

from pathlib import Path

from pydantic_settings import BaseSettings


class AudioSettings(BaseSettings):
    """Audio detection settings."""

    # File Settings
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: set = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    UPLOAD_DIR: Path = Path(__file__).parent / "uploads"
    MODEL_DIR: Path = Path(__file__).parent / "models"

    # Audio Processing Settings
    SAMPLE_RATE: int = 16000
    FRAME_LENGTH: int = 2048
    HOP_LENGTH: int = 512

    # Model Settings
    MODEL_NAME: str = "audio_deepfake_detector.pkl"
    CONFIDENCE_THRESHOLD: float = 0.5

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


audio_settings = AudioSettings()

# Create necessary directories
audio_settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
audio_settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
