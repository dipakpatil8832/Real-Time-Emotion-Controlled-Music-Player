"""
Deep Learning Model Architecture and Production Inference Module.
Features:
1. State-of-the-Art Vision Transformer (ViT) Emotion Classifier loaded from pre-trained open-source models (e.g. trpakov/vit-face-expression).
2. High-Performance OpenCV DNN Emotion Classifier loaded from FERPlus ONNX model.
3. Clean probability mapping & confidence calibration across 5 primary emotions:
   ['happy', 'sad', 'angry', 'surprise', 'neutral']
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

def _ensure_torch_environment():
    """Lazy configuration of Windows DLL paths for PyTorch when ViT is used."""
    if sys.platform == "win32":
        import site
        search_dirs = [Path(p) for p in sys.path if p]
        if hasattr(site, "getusersitepackages"):
            try:
                search_dirs.append(Path(site.getusersitepackages()))
            except Exception:
                pass
        if hasattr(site, "getsitepackages"):
            try:
                search_dirs.extend([Path(p) for p in site.getsitepackages()])
            except Exception:
                pass
        appdata = os.environ.get("APPDATA")
        if appdata:
            search_dirs.append(Path(appdata) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages")

        for p in search_dirs:
            torch_lib = p / "torch" / "lib"
            if torch_lib.exists():
                try:
                    os.add_dll_directory(str(torch_lib))
                except Exception:
                    pass
                os.environ["PATH"] = str(torch_lib) + ";" + os.environ.get("PATH", "")

import cv2
import numpy as np
from PIL import Image

from src.config import EMOTIONS, ModelConfig, model_config, MODELS_DIR, vision_config
from src.utils import logger, save_json, load_json

ONNX_MODEL_PATH = MODELS_DIR / "emotion_ferplus.onnx"
FERPLUS_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
VIT_DEFAULT_MODEL = "trpakov/vit-face-expression"


def build_emotion_model(
    config: ModelConfig = model_config,
    trainable_base: bool = False
) -> Any:
    """
    Constructs a lightweight emotion model representation for testing and compilation.
    """
    class LightweightEmotionModel:
        def __init__(self, num_classes=5):
            self.num_classes = num_classes
            self.input_shape = (None, 224, 224, 3)
            self.output_shape = (None, num_classes)
        def __call__(self, x):
            return np.ones((1, self.num_classes), dtype=np.float32) / self.num_classes

    return LightweightEmotionModel(config.num_classes)


class EmotionClassifier:
    """
    Production Deep Learning inference engine supporting both:
    1. Vision Transformer (ViT - SOTA Deep Learning for high accuracy on angry/sad/happy/neutral)
    2. OpenCV DNN with FERPlus ONNX (Lightweight offline alternative)
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model_path: Optional[Path] = None,
        vit_model_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.30,
        neutral_logit_bias: float = 0.0,
        sadness_boost: float = 0.0,
        emotion_sensitivity: float = 1.0,
        temperature: float = 1.0,
        neutral_bias_weight: Optional[float] = None
    ):
        self.labels = labels or EMOTIONS
        self.confidence_threshold = confidence_threshold
        self.neutral_logit_bias = neutral_logit_bias
        self.sadness_boost = sadness_boost
        self.emotion_sensitivity = emotion_sensitivity
        self.temperature = temperature
        self.neutral_bias_weight = neutral_bias_weight if neutral_bias_weight is not None else 1.0
        
        # Selected backend: 'vit' or 'onnx'
        self.backend = (backend or getattr(vision_config, "default_backend", "vit")).lower()
        self.onnx_path = Path(model_path or ONNX_MODEL_PATH)
        self.vit_model_name = vit_model_name or getattr(vision_config, "vit_model_name", VIT_DEFAULT_MODEL)
        
        self.vit_pipeline = None
        self.net = None
        
        self._init_backend()

    def _init_backend(self):
        """Initializes the active classification backend."""
        if self.backend in ["vit", "transformer", "vit-face-expression"]:
            try:
                _ensure_torch_environment()
                import torch
                from transformers import pipeline
                logger.info("Initializing Vision Transformer Emotion Model: %s", self.vit_model_name)
                self.vit_pipeline = pipeline(
                    "image-classification",
                    model=self.vit_model_name,
                    device="cpu"
                )
                self.backend = "vit"
                logger.info("Vision Transformer Emotion Model initialized successfully.")
            except Exception as e:
                logger.warning("Failed to initialize Vision Transformer (%s). Falling back to ONNX DNN: %s", self.vit_model_name, e)
                self.backend = "onnx"
                self.net = self._load_dnn_model()
        else:
            self.backend = "onnx"
            self.net = self._load_dnn_model()

    def set_backend(self, backend_name: str):
        """Switches the active inference backend between 'vit' and 'onnx'."""
        clean_name = backend_name.lower().strip()
        if "vit" in clean_name or "transformer" in clean_name:
            if self.vit_pipeline is None:
                self.backend = "vit"
                self._init_backend()
            else:
                self.backend = "vit"
        else:
            if self.net is None:
                self.backend = "onnx"
                self._init_backend()
            else:
                self.backend = "onnx"

    def _load_dnn_model(self) -> Optional[cv2.dnn.Net]:
        """Loads the pre-trained Deep Neural Network via OpenCV DNN."""
        if self.onnx_path.exists() and self.onnx_path.stat().st_size > 100000:
            try:
                net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
                logger.info("Loaded pre-trained FERPlus ONNX deep neural network from: %s", self.onnx_path)
                return net
            except Exception as e:
                logger.error("Failed to load ONNX model via OpenCV DNN: %s", e)

        logger.warning("ONNX model not found at %s. Attempting to download...", self.onnx_path)
        try:
            import urllib.request
            url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
            self.onnx_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, self.onnx_path)
            net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
            logger.info("Successfully downloaded and initialized ONNX Emotion model.")
            return net
        except Exception as e:
            logger.error("Failed to auto-download ONNX model: %s", e)
            return None

    def preprocess_face_roi_vit(self, face_roi: np.ndarray) -> Optional[Image.Image]:
        """
        Preprocesses cropped face ROI for Vision Transformer (RGB PIL Image, 224x224).
        """
        if face_roi is None or not isinstance(face_roi, np.ndarray) or face_roi.size == 0:
            return None

        # Handle batch dimension if present
        face_crop = face_roi[0] if len(face_roi.shape) == 4 else face_roi

        # Convert to uint8 if float
        if face_crop.dtype in [np.float32, np.float64]:
            if face_crop.max() <= 1.0:
                face_crop = (face_crop * 255.0).astype(np.uint8)
            else:
                face_crop = face_crop.astype(np.uint8)

        # Convert to RGB image
        if len(face_crop.shape) == 3:
            if face_crop.shape[2] == 4:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2RGB)
            elif face_crop.shape[2] == 3:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(face_crop[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif len(face_crop.shape) == 2:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
        else:
            return None

        if rgb.shape[0] < 5 or rgb.shape[1] < 5:
            return None

        # Resize smoothly to 224x224 with high quality interpolation
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(resized)

    def preprocess_face_roi(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocesses cropped face ROI for FERPlus ONNX model (64x64 Grayscale).
        """
        if face_roi is None or not isinstance(face_roi, np.ndarray) or face_roi.size == 0:
            return None

        face_crop = face_roi[0] if len(face_roi.shape) == 4 else face_roi

        if face_crop.dtype in [np.float32, np.float64]:
            if face_crop.max() <= 1.0:
                face_crop = (face_crop * 255.0).astype(np.uint8)
            else:
                face_crop = face_crop.astype(np.uint8)

        if len(face_crop.shape) == 3:
            if face_crop.shape[2] == 4:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2GRAY)
            elif face_crop.shape[2] == 3:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_crop[:, :, 0]
        elif len(face_crop.shape) == 2:
            gray = face_crop
        else:
            return None

        if gray.shape[0] < 5 or gray.shape[1] < 5:
            return None

        resized_64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        return resized_64

    def predict_emotion(
        self,
        face_roi: np.ndarray,
        neutral_weight: Optional[float] = None,
        sensitivity: Optional[float] = None,
        neutral_bias: Optional[float] = None,
        sad_boost: Optional[float] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs deep learning inference on face ROI image with principled probability mapping.
        """
        result = self.predict_emotion_detailed(
            face_roi,
            neutral_weight=neutral_weight,
            sensitivity=sensitivity,
            neutral_bias=neutral_bias,
            sad_boost=sad_boost
        )
        return result["display_emotion"], result["confidence"], result["probabilities"]

    def _predict_vit(
        self,
        face_roi: np.ndarray,
        neutral_bias: float = 0.0,
        sad_boost: float = 0.0,
        sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Inference using Vision Transformer (ViT).
        Accurately recognizes all 7 facial expressions without bias.
        """
        pil_img = self.preprocess_face_roi_vit(face_roi)
        if pil_img is None or self.vit_pipeline is None:
            raise ValueError("ViT image preprocessing failed or pipeline unavailable")

        raw_preds = self.vit_pipeline(pil_img)
        # raw_preds is list of dicts: [{'label': 'happy', 'score': 0.85}, ...]
        vit_scores = {r["label"].lower(): float(r["score"]) for r in raw_preds}

        # SOTA ViT models typically output 7 emotion classes:
        # angry, disgust, fear, happy, neutral, sad, surprise
        raw_happy = vit_scores.get("happy", 0.0)
        raw_sad = vit_scores.get("sad", 0.0)
        raw_angry = vit_scores.get("angry", 0.0)
        raw_surprise = vit_scores.get("surprise", 0.0)
        raw_neutral = vit_scores.get("neutral", 0.0)
        raw_disgust = vit_scores.get("disgust", 0.0)
        raw_fear = vit_scores.get("fear", 0.0)

        # Map 7 classes cleanly into 5 player moods:
        # angry encompasses facial expressions of anger and disgust
        # sad encompasses sadness and melancholic/fearful distress
        # surprise encompasses sudden surprise / high arousal
        total_distress = raw_sad + raw_fear + raw_angry + raw_disgust
        
        # Grimace / Distress filter: When a person grimaces in sadness, pain, or distress (e.g. baring teeth),
        # FER models can produce a false positive happy score (e.g. 0.44) due to visible teeth.
        # If distress/fear/sad is elevated, suppress the teeth-exposure artifact so genuine distress is recognized.
        if total_distress >= 0.25 and (raw_fear + raw_sad + raw_disgust) >= 0.20:
            smile_to_distress_ratio = raw_happy / (total_distress + 1e-6)
            if smile_to_distress_ratio < 1.4:
                p_happy = raw_happy * 0.30
            else:
                p_happy = raw_happy
        else:
            p_happy = raw_happy

        p_sad = raw_sad + 0.85 * raw_fear
        p_angry = raw_angry + 0.85 * raw_disgust
        p_surprise = raw_surprise + 0.20 * raw_fear
        p_neutral = raw_neutral

        # Convert to logits for optional user calibration/tuning
        scores_5 = np.array([p_happy, p_sad, p_angry, p_surprise, p_neutral], dtype=np.float32)
        scores_5 = np.clip(scores_5, 1e-6, 1.0)
        logits = np.log(scores_5)

        if neutral_bias != 0.0:
            logits[4] -= neutral_bias
        if sad_boost != 0.0:
            logits[1] += sad_boost
        if sensitivity != 1.0 and sensitivity > 0:
            logits[0:4] += float(np.log(max(0.1, sensitivity)))

        t = max(0.2, self.temperature)
        scaled = logits / t
        exp_l = np.exp(scaled - np.max(scaled))
        norm_scores = exp_l / (np.sum(exp_l) + 1e-9)

        prob_dict = {
            "happy": float(norm_scores[0]),
            "sad": float(norm_scores[1]),
            "angry": float(norm_scores[2]),
            "surprise": float(norm_scores[3]),
            "neutral": float(norm_scores[4]),
        }

        dominant_emotion = max(prob_dict, key=prob_dict.get)
        confidence = float(prob_dict[dominant_emotion])

        sorted_by_prob = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        top1_emotion, top1_prob = sorted_by_prob[0]
        top2_emotion, top2_prob = sorted_by_prob[1]

        is_uncertain = False
        display_emotion = dominant_emotion
        status_label = f"{dominant_emotion.capitalize()} ({confidence * 100:.1f}%)"

        if confidence < self.confidence_threshold:
            is_uncertain = True
            display_emotion = dominant_emotion  # Keep predicted dominant emotion for transparency
            status_label = f"{dominant_emotion.capitalize()} (Subtle / Low Confidence {confidence * 100:.1f}%)"

        return {
            "dominant_emotion": dominant_emotion,
            "display_emotion": display_emotion,
            "confidence": confidence,
            "probabilities": prob_dict,
            "is_uncertain": is_uncertain,
            "status_label": status_label,
            "top2_emotion": top2_emotion,
            "top2_confidence": top2_prob,
            "raw_model_scores": vit_scores,
            "backend": "Vision Transformer (ViT SOTA)"
        }

    def _predict_onnx(
        self,
        face_roi: np.ndarray,
        neutral_bias: float = 0.0,
        sad_boost: float = 0.0,
        sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Inference using OpenCV DNN FERPlus ONNX model.
        """
        resized_64 = self.preprocess_face_roi(face_roi)
        if resized_64 is None or self.net is None:
            raise ValueError("ONNX image preprocessing failed or net is None")

        blob = cv2.dnn.blobFromImage(
            resized_64.astype(np.float32),
            scalefactor=1.0,
            size=(64, 64),
            mean=(0,),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        raw_logits = self.net.forward()[0].copy()  # 8 FERPlus classes

        calibrated_logits = raw_logits.astype(np.float32)

        # Baseline class balancing for FERPlus (correcting for FER2013 neutral dataset imbalance)
        # Neutral naturally overpowers subtle sadness, pouts, and frowns by ~0.50 logit margin.
        calibrated_logits[0] -= (0.45 + neutral_bias)
        calibrated_logits[3] += (0.35 + sad_boost)
        calibrated_logits[4] += 0.20
        calibrated_logits[6] += 0.25
        if sensitivity != 1.0 and sensitivity > 0:
            calibrated_logits[1:] += float(np.log(max(0.1, sensitivity)))

        t = max(0.2, self.temperature)
        scaled = calibrated_logits / t
        exp_logits = np.exp(scaled - np.max(scaled))
        probs_8 = exp_logits / (np.sum(exp_logits) + 1e-9)

        # 0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt
        raw_h = float(probs_8[1])
        raw_s = float(probs_8[3])
        raw_a = float(probs_8[4] + probs_8[5] + probs_8[7])
        raw_f = float(probs_8[6])
        raw_surp = float(probs_8[2])
        raw_n = float(probs_8[0])

        total_distress_onnx = raw_s + raw_f + raw_a
        if total_distress_onnx >= 0.25 and (raw_s + raw_f) >= 0.20:
            ratio = raw_h / (total_distress_onnx + 1e-6)
            if ratio < 1.4:
                p_happy = raw_h * 0.30
            else:
                p_happy = raw_h
        else:
            p_happy = raw_h

        # When sad or pouting facial expressions are present and dominate positive valence
        if raw_s >= 0.12 and raw_s > raw_h:
            p_sad = raw_s * 1.35 + 0.85 * raw_f
        else:
            p_sad = raw_s + 0.85 * raw_f

        p_angry = raw_a
        p_surprise = raw_surp + 0.20 * raw_f
        p_neutral = raw_n

        raw_scores = np.array([p_happy, p_sad, p_angry, p_surprise, p_neutral], dtype=np.float32)
        norm_scores = raw_scores / (np.sum(raw_scores) + 1e-9)

        prob_dict = {
            "happy": float(norm_scores[0]),
            "sad": float(norm_scores[1]),
            "angry": float(norm_scores[2]),
            "surprise": float(norm_scores[3]),
            "neutral": float(norm_scores[4]),
        }

        dominant_emotion = max(prob_dict, key=prob_dict.get)
        confidence = float(prob_dict[dominant_emotion])

        sorted_by_prob = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        top1_emotion, top1_prob = sorted_by_prob[0]
        top2_emotion, top2_prob = sorted_by_prob[1]

        is_uncertain = False
        display_emotion = dominant_emotion
        status_label = f"{dominant_emotion.capitalize()} ({confidence * 100:.1f}%)"

        if confidence < self.confidence_threshold:
            is_uncertain = True
            display_emotion = dominant_emotion
            status_label = f"{dominant_emotion.capitalize()} (Low Confidence {confidence * 100:.1f}%)"

        return {
            "dominant_emotion": dominant_emotion,
            "display_emotion": display_emotion,
            "confidence": confidence,
            "probabilities": prob_dict,
            "is_uncertain": is_uncertain,
            "status_label": status_label,
            "top2_emotion": top2_emotion,
            "top2_confidence": top2_prob,
            "raw_model_scores": {
                FERPLUS_CLASSES[i]: float(probs_8[i]) for i in range(len(FERPLUS_CLASSES))
            },
            "backend": "FERPlus ONNX (OpenCV DNN)"
        }

    def predict_emotion_detailed(
        self,
        face_roi: np.ndarray,
        neutral_weight: Optional[float] = None,
        sensitivity: Optional[float] = None,
        neutral_bias: Optional[float] = None,
        sad_boost: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive inference returning detailed probability metrics, uncertainty flags,
        and fallback states across active backend.
        """
        uniform = {e: 1.0 / len(self.labels) for e in self.labels}
        fallback_result = {
            "dominant_emotion": "neutral",
            "display_emotion": "neutral",
            "confidence": 0.20,
            "probabilities": uniform,
            "is_uncertain": True,
            "raw_model_scores": {},
            "status_label": "No Face / Low Signal",
            "backend": self.backend
        }

        if face_roi is None or not isinstance(face_roi, np.ndarray) or face_roi.size == 0:
            return fallback_result

        bias_neutral = self.neutral_logit_bias if neutral_bias is None else neutral_bias
        boost_sad = self.sadness_boost if sad_boost is None else sad_boost
        w_sens = self.emotion_sensitivity if sensitivity is None else sensitivity

        try:
            if self.backend == "vit" and self.vit_pipeline is not None:
                return self._predict_vit(
                    face_roi,
                    neutral_bias=bias_neutral,
                    sad_boost=boost_sad,
                    sensitivity=w_sens
                )
            else:
                return self._predict_onnx(
                    face_roi,
                    neutral_bias=bias_neutral,
                    sad_boost=boost_sad,
                    sensitivity=w_sens
                )
        except Exception as e:
            logger.error("Primary model prediction error (%s): %s", self.backend, e)
            # Try fallback to ONNX if ViT failed
            if self.backend == "vit" and self.net is not None:
                try:
                    logger.info("Retrying with ONNX DNN fallback...")
                    return self._predict_onnx(
                        face_roi,
                        neutral_bias=bias_neutral,
                        sad_boost=boost_sad,
                        sensitivity=w_sens
                    )
                except Exception as ex:
                    logger.error("Fallback ONNX prediction also failed: %s", ex)
            return fallback_result
