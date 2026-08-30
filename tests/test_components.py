"""
Comprehensive Unit Tests for Emotion-Controlled Music Player Components.
Tests face detection, neural network architecture, temporal smoothing,
audio synthesis, local music provider, and recommendation engine.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import EMOTIONS, ModelConfig
from src.face_detector import FaceDetector
from src.model import EmotionClassifier, build_emotion_model
from src.music_engine import (
    LocalMusicProvider,
    MusicRecommender,
    ProceduralAudioSynthesizer,
    Track,
)
from src.smoothing import EmotionSmoother


@pytest.fixture
def sample_face_image():
    """Generates a synthetic 300x300 BGR test image."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw simple facial shape
    cv2.circle(img, (150, 150), 80, (200, 200, 200), -1)
    cv2.circle(img, (125, 130), 10, (30, 30, 30), -1)  # Left eye
    cv2.circle(img, (175, 130), 10, (30, 30, 30), -1)  # Right eye
    cv2.ellipse(img, (150, 180), (30, 15), 0, 0, 180, (30, 30, 30), 3)  # Smile
    return img


@pytest.fixture
def temp_music_dir():
    """Creates a temporary music directory for isolated testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestFaceDetector:
    def test_detector_initialization(self):
        detector = FaceDetector()
        assert detector is not None
        assert detector.min_size == (60, 60)

    def test_crop_face_with_margin(self, sample_face_image):
        detector = FaceDetector()
        bbox = (100, 100, 100, 100)
        cropped = detector.crop_face(sample_face_image, bbox, apply_margin=True)
        assert cropped is not None
        assert len(cropped.shape) == 3

    def test_preprocess_face_for_model(self, sample_face_image):
        detector = FaceDetector()
        face_roi = sample_face_image[50:200, 50:200]
        preprocessed = detector.preprocess_face_for_model(face_roi, target_size=(224, 224))
        assert preprocessed.shape == (1, 224, 224, 3)
        assert preprocessed.dtype == np.float32


class TestModelArchitecture:
    def test_model_builder(self):
        config = ModelConfig(image_height=224, image_width=224, num_classes=5)
        model = build_emotion_model(config)
        assert model is not None

    def test_classifier_inference(self):
        classifier = EmotionClassifier()
        dummy_input = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
        emotion, conf, probs = classifier.predict_emotion(dummy_input)
        
        assert emotion in EMOTIONS
        assert 0.0 <= conf <= 1.0
        assert len(probs) == len(EMOTIONS)
        assert pytest.approx(sum(probs.values()), 0.01) == 1.0


class TestEmotionSmoother:
    def test_smoother_initialization(self):
        smoother = EmotionSmoother(window_size=5, decay_factor=0.8)
        assert smoother.current_dominant_emotion == "neutral"
        assert len(smoother.history) == 0

    def test_smoothing_update_cycle(self):
        smoother = EmotionSmoother(window_size=5, decay_factor=0.7)
        raw_probs = {"happy": 0.85, "sad": 0.05, "angry": 0.02, "surprise": 0.05, "neutral": 0.03}
        
        for _ in range(4):
            emotion, conf, probs, stability = smoother.update("happy", 0.85, raw_probs)

        assert emotion == "happy"
        assert conf > 0.5
        assert stability >= 0.75

    def test_smoother_reset(self):
        smoother = EmotionSmoother()
        smoother.update("happy", 0.9, {"happy": 0.9, "sad": 0.025, "angry": 0.025, "surprise": 0.025, "neutral": 0.025})
        smoother.reset()
        assert len(smoother.history) == 0
        assert smoother.current_dominant_emotion == "neutral"


class TestAudioSynthesizer:
    def test_synthesizer_tracks(self, temp_music_dir):
        synthesizer = ProceduralAudioSynthesizer(sample_rate=22050)
        happy_audio = synthesizer.synthesize_happy_track(duration=1.0)
        sad_audio = synthesizer.synthesize_sad_track(duration=1.0)
        
        assert len(happy_audio) == 22050
        assert len(sad_audio) == 22050
        assert np.max(np.abs(happy_audio)) > 0

    def test_generate_full_sample_library(self, temp_music_dir):
        synthesizer = ProceduralAudioSynthesizer(sample_rate=22050)
        generated = synthesizer.generate_full_sample_library(temp_music_dir)
        
        for emotion in EMOTIONS:
            assert emotion in generated
            assert len(generated[emotion]) > 0
            for file_path in generated[emotion]:
                assert file_path.exists()
                assert file_path.stat().st_size > 0


class TestMusicProviderAndRecommender:
    def test_provider_initialization(self, temp_music_dir):
        provider = LocalMusicProvider(temp_music_dir)
        provider.initialize()
        tracks = provider.get_all_tracks()
        assert len(tracks) >= 5

        for emotion in EMOTIONS:
            emotion_tracks = provider.get_tracks_by_emotion(emotion)
            assert len(emotion_tracks) >= 1

    def test_recommender_lifecycle(self, temp_music_dir):
        provider = LocalMusicProvider(temp_music_dir)
        provider.initialize()
        recommender = MusicRecommender(provider)
        recommender.initialize()

        assert recommender.current_track is not None
        
        # Test emotion switch
        new_track = recommender.on_emotion_update("happy", 0.92, 0.85)
        if new_track:
            assert new_track.emotion_tag == "happy"
            assert recommender.current_emotion == "happy"

        # Test smart next
        next_track = recommender.next_track()
        assert next_track is not None
