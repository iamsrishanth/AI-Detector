"""CLI entry point for AI Detector."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ai-detect",
        description="AI Detector — Detect AI-generated images, videos, and deepfake audio.",
    )
    parser.add_argument("--image", type=str, help="Path to an image file to analyze.")
    parser.add_argument("--video", type=str, help="Path to a video file to analyze.")
    parser.add_argument("--audio", type=str, help="Path to an audio file to analyze.")
    parser.add_argument(
        "--gui", action="store_true", help="Launch the Gradio web interface."
    )
    args = parser.parse_args()

    if args.gui:
        from ai_detector.app import launch_app

        launch_app()

    elif args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)
        from ai_detector.image.detector import ImageDetector

        detector = ImageDetector()
        detector.load_models()
        detector.process_image(str(path))

    elif args.video:
        path = Path(args.video)
        if not path.exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)
        from ai_detector.image.detector import ImageDetector

        detector = ImageDetector()
        detector.load_models()
        detector.process_video(str(path))

    elif args.audio:
        path = Path(args.audio)
        if not path.exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)
        from ai_detector.audio.processor import AudioProcessor
        from ai_detector.audio.detector import DeepfakeDetector

        processor = AudioProcessor()
        detector = DeepfakeDetector()
        detector.load_model()

        features = processor.extract_features(path)
        result = detector.predict(features)
        metadata = processor.get_metadata(path)

        print(f"\n🎵 Audio Analysis: {path.name}")
        print(f"   Duration: {metadata.duration:.2f}s | Sample Rate: {metadata.sample_rate} Hz")
        print(f"   {result['message']}")
        print(f"   Deepfake Probability: {result['deepfake_probability'] * 100:.1f}%")
        print(f"   Authentic Probability: {result['authentic_probability'] * 100:.1f}%")

    else:
        print("❌ No input provided! Use --image, --video, --audio, or --gui.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
