"""
Face Detection and Face Extraction Module
Uses OpenCV Haar Cascade and DNN with margin expansion, aspect ratio handling, and pre-processing.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import vision_config
from src.utils import logger


class FaceDetector:
    """
    Robust Face Detector capable of finding the primary face in an image frame,
    expanding the bounding box to capture full facial context (forehead, chin, cheeks),
    and cropping/aligning for neural network input.
    """

    def __init__(
        self,
        cascade_path: Optional[Path] = None,
        min_size: Tuple[int, int] = vision_config.face_min_size,
        scale_factor: float = vision_config.scale_factor,
        min_neighbors: int = vision_config.min_neighbors,
        margin_ratio: float = vision_config.face_margin_ratio
    ):
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.margin_ratio = margin_ratio
        
        # Load OpenCV default Haar cascade
        self.cascade = self._load_cascade(cascade_path)

    def _load_cascade(self, cascade_path: Optional[Path]) -> cv2.CascadeClassifier:
        """Loads Haar Cascade Classifier from cv2.data or fallback."""
        candidates = []
        if cascade_path and Path(cascade_path).exists():
            candidates.append(str(cascade_path))
        
        # OpenCV built-in data directory
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidates.append(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            candidates.append(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

        for cand in candidates:
            if Path(cand).exists():
                cascade = cv2.CascadeClassifier(cand)
                if not cascade.empty():
                    logger.info("Loaded Haar Cascade from: %s", cand)
                    return cascade

        # If not found, attempt to initialize default
        cascade = cv2.CascadeClassifier()
        logger.warning("Haar cascade file not found directly, using default empty fallback.")
        return cascade

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects all faces in the given frame.
        
        Args:
            frame: BGR or Grayscale image (numpy array).
            
        Returns:
            List of bounding boxes: [(x, y, w, h), ...]
        """
        if frame is None or frame.size == 0:
            return []

        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Histogram equalization for lighting robustness
        gray_eq = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray_eq,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return []

        # Sort faces by area descending (largest face first)
        faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return [tuple(map(int, f)) for f in faces_sorted]

    def get_primary_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Returns the bounding box of the largest face in the frame."""
        faces = self.detect_faces(frame)
        return faces[0] if faces else None

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        apply_margin: bool = True
    ) -> Optional[np.ndarray]:
        """
        Crops face region with optional margin expansion to capture full facial expression context.
        
        Args:
            frame: Original image (BGR or RGB).
            bbox: (x, y, w, h) bounding box.
            apply_margin: Whether to add a percentage margin.
            
        Returns:
            Cropped face image or None.
        """
        if frame is None or bbox is None:
            return None

        h_img, w_img = frame.shape[:2]
        x, y, w, h = bbox

        if apply_margin:
            mx = int(w * self.margin_ratio)
            my = int(h * self.margin_ratio)
            x_start = max(0, x - mx)
            y_start = max(0, y - my)
            x_end = min(w_img, x + w + mx)
            y_end = min(h_img, y + h + my)
        else:
            x_start, y_start = max(0, x), max(0, y)
            x_end, y_end = min(w_img, x + w), min(h_img, y + h)

        if x_end <= x_start or y_end <= y_start:
            return None

        face_roi = frame[y_start:y_end, x_start:x_end]
        return face_roi

    def preprocess_face_for_model(
        self,
        face_roi: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        to_rgb: bool = True
    ) -> np.ndarray:
        """
        Resizes and prepares a cropped face for deep learning model inference.
        
        Args:
            face_roi: Cropped face numpy array.
            target_size: (width, height) desired by model.
            to_rgb: Convert BGR to RGB.
            
        Returns:
            Preprocessed normalized image array of shape (1, height, width, 3).
        """
        if to_rgb and len(face_roi.shape) == 3:
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        elif len(face_roi.shape) == 2:
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        else:
            face_img = face_roi

        resized = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
        # MobileNetV2 expects normalized input in [-1, 1] or [0, 1] via tf.keras.applications.mobilenet_v2.preprocess_input
        # We store as float32 array in [0, 255] or batch format
        expanded = np.expand_dims(resized.astype(np.float32), axis=0)
        return expanded
