<div align="center">

# 🛡️ AI Detector

**Unified AI-generated content detection for images, videos, and deepfake audio.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web_UI-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)

<p>
  A powerful, modular Python toolkit that combines <b>image/video AI detection</b> (CLIP & ViT) and <b>audio deepfake detection</b> (spectral analysis & RandomForest) into one unified CLI and web interface.
</p>

---

</div>

> [!WARNING]
> Detection results may not always be accurate. AI‑generated content detection is an evolving field — always verify with additional methods when necessary.

## ✨ Features

| Module | Capabilities |
|--------|-------------|
| 🖼️ **Image Detection** | CLIP & ViT model classification, noise analysis, texture analysis, Fourier Transform pattern detection, metadata inspection, invisible watermark detection |
| 🎬 **Video Detection** | Frame‑by‑frame analysis using the same image detection pipeline |
| 🎵 **Audio Deepfake Detection** | Spectral feature extraction (MFCCs, spectral centroid, chroma, ZCR) with a RandomForest classifier |
| 🌐 **Gradio Web UI** | Tabbed interface for image and audio detection — upload & analyze in your browser |
| ⌨️ **CLI** | Scriptable command‑line interface for quick or batch processing |

---

## 📋 Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | `>= 3.11` |
| **PyTorch** | `>= 2.5` (CUDA‑enabled GPU recommended for faster inference) |
| **OS** | Windows / Linux / macOS |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/iamsrishanth/AI-Detector.git
cd AI-Detector
```

### 2. Create a virtual environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

**Using pip:**

```bash
pip install -e .
```

**Using uv (fast Python package manager):**

```bash
uv sync
```

**Using requirements.txt:**

```bash
pip install -r requirements.txt
```

> [!TIP]
> If you have an NVIDIA GPU, ensure you install the CUDA‑enabled version of PyTorch for significantly faster inference. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for platform-specific instructions.

---

## 🎯 Usage

### Command‑Line Interface (CLI)

```bash
# Analyze an image
ai-detect --image path/to/image.jpg

# Analyze a video
ai-detect --video path/to/video.mp4

# Analyze an audio file
ai-detect --audio path/to/audio.wav

# Launch the Gradio web interface
ai-detect --gui
```

### Python API

```python
from ai_detector.image import ImageDetector
from ai_detector.audio import AudioProcessor, DeepfakeDetector

# ── Image / Video Detection ──────────────────────────────
detector = ImageDetector()
detector.load_models()
result = detector.process_image("photo.jpg")       # single image
result = detector.process_video("clip.mp4")         # video

# ── Audio Deepfake Detection ─────────────────────────────
processor = AudioProcessor()
det = DeepfakeDetector()
det.load_model()

features  = processor.extract_features("audio.wav")
result    = det.predict(features)

print(f"Deepfake probability: {result['deepfake_probability'] * 100:.1f}%")
```

### Gradio Web UI

Launch the web interface and open it in your browser:

```bash
ai-detect --gui
```

The UI provides a tabbed interface with separate tabs for **Image Detection** and **Audio Detection** — simply upload a file and click **Analyze**.

---

## 📂 Project Structure

```
AI-Detector/
├── ai_detector/
│   ├── __init__.py               # Package init
│   ├── app.py                    # Gradio web interface
│   ├── cli.py                    # CLI entry point
│   ├── image/
│   │   ├── __init__.py
│   │   └── detector.py           # Image & video detection (CLIP + ViT)
│   └── audio/
│       ├── __init__.py
│       ├── config.py             # Audio detection settings
│       ├── models.py             # Pydantic response models
│       ├── processor.py          # Audio feature extraction
│       └── detector.py           # Deepfake detection (RandomForest)
├── tests/
│   ├── test_image_detector.py
│   ├── test_audio_detector.py
│   └── test_audio_processor.py
├── pyproject.toml                # Project metadata & dependencies
├── requirements.txt              # Pip requirements
├── LICENSE                       # Apache License 2.0
└── README.md
```

---

## 🛠️ Tech Stack

<table>
  <tr>
    <td align="center"><b>Deep Learning</b></td>
    <td>PyTorch · Transformers (HuggingFace) · CLIP · ViT</td>
  </tr>
  <tr>
    <td align="center"><b>Audio Analysis</b></td>
    <td>Librosa · SoundFile · SciPy</td>
  </tr>
  <tr>
    <td align="center"><b>ML</b></td>
    <td>scikit‑learn (RandomForest)</td>
  </tr>
  <tr>
    <td align="center"><b>Computer Vision</b></td>
    <td>OpenCV · Pillow</td>
  </tr>
  <tr>
    <td align="center"><b>Web UI</b></td>
    <td>Gradio</td>
  </tr>
  <tr>
    <td align="center"><b>Data Validation</b></td>
    <td>Pydantic</td>
  </tr>
</table>

---

## 🧪 Running Tests

```bash
# Using pytest
pytest

# Verbose output
pytest -v
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. **Fork** the repository
2. **Create** a feature branch — `git checkout -b feature/amazing-feature`
3. **Commit** your changes — `git commit -m "Add amazing feature"`
4. **Push** to your branch — `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/iamsrishanth">iamsrishanth</a></sub>
</div>
