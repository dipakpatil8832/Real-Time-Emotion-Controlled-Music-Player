"""
Utility Helpers for Computer Vision, Image Conversions, and Diagnostics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from src.config import EMOTION_METADATA, EMOTIONS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmotionMusicPlayer")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image array to RGB."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB array to OpenCV BGR image."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    rgb_arr = np.array(pil_image)
    if len(rgb_arr.shape) == 2:
        return cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
    if rgb_arr.shape[2] == 4:
        rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def draw_face_annotations(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    emotion: str,
    confidence: float,
    all_probabilities: Optional[Dict[str, float]] = None,
    is_uncertain: bool = False,
    other_faces: Optional[List[Tuple[int, int, int, int]]] = None
) -> np.ndarray:
    """
    Annotates an image frame with styled bounding box, emotion badge, and confidence percentage.
    Supports secondary face indicators and uncertainty badges.
    """
    annotated = frame.copy()
    
    # 1. Draw secondary faces with subtle muted boxes if present
    if other_faces:
        for idx, (ox, oy, ow, oh) in enumerate(other_faces, start=2):
            cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), (120, 140, 160), 1, lineType=cv2.LINE_AA)
            sub_label = f"Face #{idx}"
            cv2.putText(
                annotated,
                sub_label,
                (ox + 4, max(15, oy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 200, 220),
                1,
                lineType=cv2.LINE_AA
            )

    # 2. Draw primary face
    x, y, w, h = bbox
    meta = EMOTION_METADATA.get(emotion.lower(), {"color": "#6366F1", "emoji": "🌿"})
    
    if is_uncertain:
        # Amber warning color for uncertain/fallback states
        bgr_color = (30, 160, 245)  # Amber in BGR
        label_text = f"Uncertain: {emotion.capitalize()} ({confidence * 100:.1f}%)"
    else:
        # Convert hex color to BGR for OpenCV
        hex_color = meta.get("color", "#6366F1").lstrip("#")
        rgb_tuple = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])
        label_text = f"{emotion.capitalize()} ({confidence * 100:.1f}%)"

    # Draw primary face bounding box (thickness 3)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr_color, 3, lineType=cv2.LINE_AA)

    # Header label banner calculation
    font_scale = 0.65
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    banner_height = text_h + 12
    if y >= banner_height:
        banner_top = y - banner_height
        banner_bottom = y
        text_y = y - 6
    else:
        banner_top = y + h
        banner_bottom = y + h + banner_height
        text_y = banner_bottom - 6

    banner_right = min(annotated.shape[1], x + text_w + 16)
    
    # Fill label banner background
    cv2.rectangle(annotated, (x, banner_top), (banner_right, banner_bottom), bgr_color, -1)
    
    # Draw label text (white)
    cv2.putText(
        annotated,
        label_text,
        (x + 8, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA
    )

    return annotated


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "FER-2013 Emotion Classification Confusion Matrix"
) -> plt.Figure:
    """
    Generates and optionally saves a normalized confusion matrix heatmap.
    """
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar=True,
        ax=ax
    )
    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Emotion", fontsize=11, labelpad=8)
    ax.set_ylabel("True Emotion", fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved confusion matrix plot to: %s", save_path)
    
    return fig


def save_json(data: Union[dict, list], path: Path) -> None:
    """Safely serialize JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Union[dict, list]:
    """Safely deserialize JSON data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
