"""Deepfake detection model and inference logic."""

import logging
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ai_detector.audio.config import audio_settings

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """Handles deepfake detection using ML models."""

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.threshold = audio_settings.CONFIDENCE_THRESHOLD

    def load_model(self):
        """Load the trained model or create a default one."""
        model_path = audio_settings.MODEL_DIR / audio_settings.MODEL_NAME

        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.model_loaded = True
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._create_default_model()
        else:
            logger.warning("No trained model found, creating default model")
            self._create_default_model()

    def _create_default_model(self):
        """Create a default trained model for demonstration."""
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        # Synthetic training data for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_features = 36

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)

        # Deepfakes tend to have different spectral characteristics
        X_train[y_train == 1] += (
            np.random.randn(np.sum(y_train == 1), n_features) * 0.5
        )

        self.model.fit(X_train, y_train)
        self.model_loaded = True

        # Save the default model
        model_path = audio_settings.MODEL_DIR / audio_settings.MODEL_NAME
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"Saved default model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving default model: {e}")

    def predict(self, features: Dict[str, float]) -> Dict:
        """Predict if audio is deepfake."""
        if not self.model_loaded or self.model is None:
            logger.error("Model not loaded")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "deepfake_probability": 0.0,
                "authentic_probability": 0.0,
                "message": "Model not loaded",
            }

        try:
            feature_vector = self._features_to_array(features)
            probabilities = self.model.predict_proba(feature_vector)[0]
            authentic_prob = float(probabilities[0])
            deepfake_prob = float(probabilities[1])

            is_deepfake = deepfake_prob > self.threshold
            confidence = max(authentic_prob, deepfake_prob)

            if is_deepfake:
                message = (
                    f"⚠️ Warning: This audio appears to be AI-generated (Deepfake) "
                    f"with {confidence * 100:.1f}% confidence"
                )
            else:
                message = (
                    f"✓ This audio appears to be authentic "
                    f"with {confidence * 100:.1f}% confidence"
                )

            return {
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "deepfake_probability": deepfake_prob,
                "authentic_probability": authentic_prob,
                "message": message,
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "deepfake_probability": 0.0,
                "authentic_probability": 0.0,
                "message": f"Error during prediction: {str(e)}",
            }

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array."""
        sorted_keys = sorted(features.keys())
        feature_values = [features[key] for key in sorted_keys]

        expected_features = 36
        if len(feature_values) < expected_features:
            feature_values.extend([0.0] * (expected_features - len(feature_values)))
        elif len(feature_values) > expected_features:
            feature_values = feature_values[:expected_features]

        return np.array(feature_values).reshape(1, -1)
