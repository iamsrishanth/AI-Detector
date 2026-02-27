"""Tests for the AudioProcessor class."""

import pytest
from ai_detector.audio.processor import AudioProcessor


@pytest.fixture
def audio_processor():
    """Create an AudioProcessor instance."""
    return AudioProcessor()


def test_audio_processor_initialization(audio_processor):
    """Test AudioProcessor initialization."""
    assert audio_processor.sample_rate == 16000
    assert audio_processor.frame_length == 2048
    assert audio_processor.hop_length == 512


def test_extract_features_callable(audio_processor):
    """Test that extract_features is callable."""
    assert hasattr(audio_processor, "extract_features")
    assert callable(audio_processor.extract_features)


def test_get_metadata_callable(audio_processor):
    """Test that get_metadata is callable."""
    assert hasattr(audio_processor, "get_metadata")
    assert callable(audio_processor.get_metadata)
