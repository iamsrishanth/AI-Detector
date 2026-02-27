"""Tests for the DeepfakeDetector class."""

import pytest
import numpy as np
from ai_detector.audio.detector import DeepfakeDetector


@pytest.fixture
def detector():
    """Create a DeepfakeDetector instance with loaded model."""
    det = DeepfakeDetector()
    det.load_model()
    return det


def test_detector_initialization(detector):
    """Test DeepfakeDetector initialization."""
    assert detector.model_loaded is True
    assert detector.threshold == 0.5


def test_predict_with_valid_features(detector):
    """Test prediction with valid features."""
    features = {f"feature_{i}": float(np.random.rand()) for i in range(36)}
    result = detector.predict(features)

    assert "is_deepfake" in result
    assert "confidence" in result
    assert "deepfake_probability" in result
    assert "authentic_probability" in result
    assert "message" in result

    assert isinstance(result["is_deepfake"], bool)
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["deepfake_probability"] <= 1
    assert 0 <= result["authentic_probability"] <= 1


def test_predict_probabilities_sum_to_one(detector):
    """Test that probabilities sum to approximately 1."""
    features = {f"feature_{i}": float(np.random.rand()) for i in range(36)}
    result = detector.predict(features)

    prob_sum = result["deepfake_probability"] + result["authentic_probability"]
    assert abs(prob_sum - 1.0) < 0.01
