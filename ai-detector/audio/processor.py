"""Audio processing utilities for feature extraction."""

import logging
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import soundfile as sf

from ai_detector.audio.config import audio_settings
from ai_detector.audio.models import AudioMetadata

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file processing and feature extraction."""

    def __init__(self):
        self.sample_rate = audio_settings.SAMPLE_RATE
        self.frame_length = audio_settings.FRAME_LENGTH
        self.hop_length = audio_settings.HOP_LENGTH

    def load_audio(self, file_path: Path) -> tuple:
        """Load audio file and resample to target sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            logger.info(
                f"Loaded audio: {file_path.name}, duration: {len(audio)/sr:.2f}s"
            )
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise

    def extract_features(self, file_path: Path) -> Dict[str, float]:
        """Extract audio features for deepfake detection."""
        audio, sr = self.load_audio(file_path)
        features: Dict[str, float] = {}

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length
        )[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length
        )[0]
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length
        )[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        features["zero_crossing_rate_mean"] = float(np.mean(zcr))
        features["zero_crossing_rate_std"] = float(np.std(zcr))

        # MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        for i in range(13):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=self.frame_length, hop_length=self.hop_length
        )
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))

        # RMS Energy
        rms = librosa.feature.rms(
            y=audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        # Temporal features
        features["duration"] = float(len(audio) / sr)

        logger.info(f"Extracted {len(features)} features from {file_path.name}")
        return features

    def get_metadata(self, file_path: Path) -> AudioMetadata:
        """Get metadata from audio file."""
        try:
            info = sf.info(str(file_path))
            return AudioMetadata(
                duration=info.duration,
                sample_rate=info.samplerate,
                channels=info.channels,
                format=info.format,
            )
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            # Fallback to librosa
            audio, sr = self.load_audio(file_path)
            return AudioMetadata(
                duration=len(audio) / sr,
                sample_rate=sr,
                channels=1,
                format=file_path.suffix[1:],
            )
