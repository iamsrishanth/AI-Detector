"""Tests for the ImageDetector class."""

import pytest
from ai_detector.image.detector import ImageDetector


def test_image_detector_initialization():
    """Test ImageDetector initializes without loading models."""
    detector = ImageDetector()
    assert detector.models_loaded is False
    assert detector.clip_model is None
    assert detector.vit_model is None


def test_image_detector_has_methods():
    """Test ImageDetector exposes expected methods."""
    detector = ImageDetector()
    assert callable(detector.load_models)
    assert callable(detector.classify_image)
    assert callable(detector.process_image)
    assert callable(detector.process_video)


def test_static_helpers_exist():
    """Test static analysis helpers are accessible."""
    assert callable(ImageDetector.estimate_noise)
    assert callable(ImageDetector.analyze_texture)
    assert callable(ImageDetector.detect_repeating_patterns)
    assert callable(ImageDetector.analyze_metadata)
    assert callable(ImageDetector.analyze_color_distribution)
    assert callable(ImageDetector.detect_watermark)
    assert callable(ImageDetector.load_image)
