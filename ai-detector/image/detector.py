"""Image and video AI-generated content detector using CLIP and ViT models."""

import logging
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import piexif
import piexif.helper
from accelerate import Accelerator
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, ViTImageProcessor, ViTModel

try:
    import pillow_avif  # noqa: F401
    import pillow_heif  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ImageDetector:
    """Detects AI-generated images and videos using CLIP and ViT models."""

    def __init__(self):
        self.accelerator = Accelerator()
        self.clip_model = None
        self.clip_processor = None
        self.vit_model = None
        self.vit_processor = None
        self._models_loaded = False

    def load_models(self):
        """Load CLIP and ViT models for image feature extraction."""
        try:
            logger.info("🔄 Initializing image detection models...")

            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.accelerator.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14", use_fast=False
            )

            self.vit_model = ViTModel.from_pretrained(
                "google/vit-large-patch32-224-in21k"
            ).to(self.accelerator.device)
            self.vit_processor = ViTImageProcessor.from_pretrained(
                "google/vit-large-patch32-224-in21k"
            )

            self._models_loaded = True
            logger.info("✅ Image detection models are ready.")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    # ── Analysis helpers ──────────────────────────────────────────────

    @staticmethod
    def load_image(image_path: str) -> Image.Image | None:
        """Open an image and ensure it's in RGB mode."""
        try:
            logger.info(f"🖼️ Loading image: {image_path}")
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"❌ Error loading image: {e}")
            return None

    @staticmethod
    def estimate_noise(image: Image.Image) -> float:
        """Estimate image noise using pixel standard deviation."""
        gray = np.array(image.convert("L"))
        return float(np.std(gray))

    @staticmethod
    def analyze_texture(image: Image.Image) -> float:
        """Analyze texture complexity using edge detection."""
        gray = np.array(image.convert("L"))
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges) / edges.size)

    @staticmethod
    def detect_repeating_patterns(image: Image.Image) -> float:
        """Detect unnatural repeating patterns via Fourier Transform."""
        img_np = np.array(image.convert("L"))
        fft = np.fft.fft2(img_np)
        magnitude_spectrum = np.log1p(np.abs(np.fft.fftshift(fft)))
        return float(np.mean(magnitude_spectrum))

    @staticmethod
    def analyze_metadata(image_path: str) -> str:
        """Check image metadata for AI-generation clues."""
        try:
            exif_data = piexif.load(image_path)
            if piexif.ExifIFD.UserComment in exif_data["Exif"]:
                exif_dict = piexif.helper.UserComment.load(
                    exif_data["Exif"][piexif.ExifIFD.UserComment]
                )
                if "Stable Diffusion" in exif_dict:
                    return "AI tool detected in metadata"
            return "No AI tool detected in metadata"
        except piexif._exceptions.InvalidImageDataError:
            return "Invalid EXIF data"
        except Exception as e:
            logger.error(f"❌ Error analyzing metadata: {e}")
            return "Metadata analysis failed"

    @staticmethod
    def analyze_color_distribution(image: Image.Image) -> float:
        """Analyze color distribution for unnatural patterns."""
        np_image = np.array(image)
        hist_r, _ = np.histogram(np_image[:, :, 0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(np_image[:, :, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(np_image[:, :, 2], bins=256, range=(0, 256))
        return float(np.std(hist_r) + np.std(hist_g) + np.std(hist_b))

    @staticmethod
    def detect_watermark(image: Image.Image) -> str:
        """Detect the presence of invisible watermarks."""
        try:
            exif_data = piexif.load(image.info.get("exif", b""))
            if piexif.ExifIFD.UserComment in exif_data["Exif"]:
                watermark = piexif.helper.UserComment.load(
                    exif_data["Exif"][piexif.ExifIFD.UserComment]
                )
                return watermark
            return "No watermark detected"
        except piexif._exceptions.InvalidImageDataError:
            return "Invalid EXIF data"
        except Exception as e:
            logger.error(f"❌ Error detecting watermark: {e}")
            return "No watermark detected"

    # ── Classification ────────────────────────────────────────────────

    def classify_image(self, image: Image.Image, image_path: str) -> str:
        """Determine whether an image is AI-generated, likely real, or real."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        try:
            noise_level = self.estimate_noise(image)
            edge_density = self.analyze_texture(image)
            pattern_score = self.detect_repeating_patterns(image)
            metadata_info = self.analyze_metadata(image_path)
            color_distribution = self.analyze_color_distribution(image)
            watermark_info = self.detect_watermark(image)

            # CLIP features
            clip_inputs = self.clip_processor(images=image, return_tensors="pt").to(
                self.accelerator.device
            )
            clip_outputs = (
                self.clip_model.get_image_features(**clip_inputs)
                .detach()
                .cpu()
                .numpy()
            )
            clip_confidence = np.clip(
                np.interp(np.median(clip_outputs), [-0.3, 0.3], [0, 100]), 0, 100
            )

            # ViT features
            vit_inputs = self.vit_processor(images=image, return_tensors="pt").to(
                self.accelerator.device
            )
            vit_outputs = (
                self.vit_model(**vit_inputs)
                .last_hidden_state.mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            vit_confidence = np.clip(
                np.interp(np.median(vit_outputs), [-0.3, 0.3], [0, 100]), 0, 100
            )

            combined_confidence = (clip_confidence + vit_confidence) / 2

            if noise_level < 50:
                classification = "AI-Generated or Heavily Processed Content"
            elif 50 <= noise_level < 60:
                classification = "Likely Real Content (Possibly AI-Generated)"
            else:
                classification = "Real Content (Unlikely AI-Generated)"

            results = [
                f"📊 Noise Level: {noise_level:.2f}, Edge Density: {edge_density:.2f}, "
                f"Pattern Score: {pattern_score:.2f}, Color Distribution: {color_distribution:.2f}",
                f"📊 Metadata Info: {metadata_info}",
                f"📊 Watermark Info: {watermark_info}",
                f"🤖 Prediction Results: {100 - combined_confidence:.2f}% confidence that the image is "
                f"human-made, {combined_confidence:.2f}% confidence that it is AI-generated.",
                f"🔍 Verdict: {classification} (Confidence: {combined_confidence:.2f}%)",
            ]

            for r in results:
                print(r)

            return "\n".join(results)
        except Exception as e:
            logger.error(f"❌ Error classifying image: {e}")
            return "Error in classification"

    def process_image(self, image_path: str) -> str | None:
        """Load and classify an image file."""
        image = self.load_image(image_path)
        if image:
            return self.classify_image(image, image_path)
        return None

    def process_video(self, video_path: str) -> list[str]:
        """Analyze video frames and return classification per keyframe."""
        results = []
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % fps == 0:
                    timestamp = str(timedelta(seconds=frame_count // fps))
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    print(f"🕒 Analyzing frame at {timestamp}")
                    result = self.classify_image(image, video_path)
                    results.append(f"[{timestamp}]\n{result}")
                frame_count += 1

            cap.release()
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
        return results
