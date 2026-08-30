"""
Centralized Configuration Module
Defines paths, emotion labels, hyperparameters, model constants, and defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
MUSIC_DIR = BASE_DIR / "music_library"
DATA_DIR = BASE_DIR / "data"

# Ensure runtime directories exist
for directory in [MODELS_DIR, ASSETS_DIR, MUSIC_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Supported Emotion Labels (Strictly 5 standard primary emotions)
EMOTIONS: List[str] = ["happy", "sad", "angry", "surprise", "neutral"]
EMOTION_TO_IDX: Dict[str, int] = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION: Dict[int, str] = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

# Emotion UI Metadata (Color badges, emojis, and mood descriptions)
EMOTION_METADATA: Dict[str, Dict[str, str]] = {
    "happy": {
        "emoji": "😄",
        "color": "#10B981",  # Emerald green
        "description": "Joyful, energetic, upbeat vibes",
        "genre": "Pop / Dance / Upbeat EDM"
    },
    "sad": {
        "emoji": "😢",
        "color": "#3B82F6",  # Cool Blue
        "description": "Melancholic, reflective, gentle acoustic tones",
        "genre": "Acoustic / Lo-Fi / Soft Piano"
    },
    "angry": {
        "emoji": "🔥",
        "color": "#EF4444",  # Intense Red
        "description": "High tempo, driving rhythms, cathartic energy",
        "genre": "Rock / Heavy Beats / Fast Tempo"
    },
    "surprise": {
        "emoji": "⚡",
        "color": "#F59E0B",  # Vibrant Amber
        "description": "Dynamic, unpredictable, uplifting grooves",
        "genre": "Funk / Synthwave / Electronic"
    },
    "neutral": {
        "emoji": "🌿",
        "color": "#6366F1",  # Soft Indigo
        "description": "Balanced, focused, chill background atmosphere",
        "genre": "Ambient / Chillhop / Lo-Fi Beats"
    }
}


@dataclass
class ModelConfig:
    """Hyperparameters and configuration for Emotion Recognition Model."""
    image_height: int = 224
    image_width: int = 224
    num_channels: int = 3
    num_classes: int = len(EMOTIONS)
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 1e-4
    dropout_rate: float = 0.35
    l2_reg: float = 1e-4
    fine_tune_at: int = 100  # Layer from which to unfreeze MobileNetV2
    model_save_path: Path = MODELS_DIR / "emotion_mobilenetv2.keras"
    weights_save_path: Path = MODELS_DIR / "emotion_weights.weights.h5"
    labels_save_path: Path = MODELS_DIR / "emotion_labels.json"


@dataclass
class VisionConfig:
    """Vision and Face Detection Configuration."""
    face_min_size: Tuple[int, int] = (60, 60)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    face_margin_ratio: float = 0.10  # Balanced margin to focus on facial features
    confidence_threshold: float = 0.35  # Minimum prediction confidence to accept
    neutral_suppression_factor: float = 0.70  # Calibration factor to counter dataset neutral dominance
    emotion_sensitivity: float = 1.25  # Sensitivity boost for active facial expressions
    haar_cascade_path: Path = MODELS_DIR / "haarcascade_frontalface_default.xml"


@dataclass
class SmoothingConfig:
    """Temporal Smoothing and Rolling Buffer Configuration."""
    window_size: int = 8  # Number of historical frames
    decay_factor: float = 0.60  # Responsive exponential moving average decay
    switch_confidence_gain: float = 1.15  # Hysteresis multiplier to avoid rapid track-switching


@dataclass
class MusicConfig:
    """Audio and Music Player Configuration."""
    music_library_path: Path = MUSIC_DIR
    sample_rate: int = 44100
    default_volume: float = 0.85
    supported_audio_formats: Tuple[str, ...] = (".mp3", ".wav", ".ogg", ".flac")
    default_emotion: str = "neutral"


# Global Default Instances
model_config = ModelConfig()
vision_config = VisionConfig()
smoothing_config = SmoothingConfig()
music_config = MusicConfig()
