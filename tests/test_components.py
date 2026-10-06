"""
Comprehensive Unit Tests for Emotion-Controlled Music Player Components.
Tests face detection, neural network architecture, temporal smoothing,
audio synthesis, local music provider, and recommendation engine.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    YouTubeMusicProvider,
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
        assert detector.min_size == (50, 50) or detector.min_size == (60, 60)

    def test_crop_face_with_margin(self, sample_face_image):
        detector = FaceDetector()
        bbox = (100, 100, 100, 100)
        cropped = detector.crop_face(sample_face_image, bbox, apply_margin=True)
        assert cropped is not None
        assert len(cropped.shape) == 3

    def test_preprocess_face_for_model(self, sample_face_image):
        detector = FaceDetector()
        face_roi = sample_face_image[50:200, 50:200]
        preprocessed_rgb = detector.preprocess_face_for_model(face_roi, target_size=(224, 224), to_rgb=True)
        assert preprocessed_rgb.shape == (1, 224, 224, 3)
        assert preprocessed_rgb.dtype == np.float32

        preprocessed_onnx = detector.preprocess_face_for_model(face_roi, target_size=(64, 64), to_rgb=False)
        assert preprocessed_onnx.shape == (1, 1, 64, 64)
        assert preprocessed_onnx.dtype == np.float32


class TestModelArchitecture:
    def test_model_builder(self):
        config = ModelConfig(image_height=224, image_width=224, num_classes=5)
        model = build_emotion_model(config)
        assert model is not None

    def test_classifier_inference(self):
        classifier = EmotionClassifier()
        dummy_input = np.random.uniform(0, 255, (1, 64, 64, 3)).astype(np.float32)
        emotion, conf, probs = classifier.predict_emotion(dummy_input)
        
        assert emotion in EMOTIONS
        assert 0.0 <= conf <= 1.0
        assert len(probs) == len(EMOTIONS)
        assert pytest.approx(sum(probs.values()), 0.001) == 1.0

        # Test uncorrupted pure baseline inference
        neutral_sample = np.full((64, 64, 3), 128, dtype=np.uint8)
        cv2.circle(neutral_sample, (20, 25), 4, (30, 30, 30), -1)
        cv2.circle(neutral_sample, (44, 25), 4, (30, 30, 30), -1)
        cv2.line(neutral_sample, (22, 45), (42, 45), (30, 30, 30), 2)
        
        detailed = classifier.predict_emotion_detailed(neutral_sample)
        assert "dominant_emotion" in detailed
        assert "display_emotion" in detailed
        assert "probabilities" in detailed
        assert pytest.approx(sum(detailed["probabilities"].values()), 0.001) == 1.0
        assert 0.0 <= detailed["confidence"] <= 1.0

    def test_confidence_threshold_fallback(self):
        classifier = EmotionClassifier(confidence_threshold=0.99)  # Ultra high threshold
        dummy_input = np.full((64, 64, 3), 128, dtype=np.uint8)
        detailed = classifier.predict_emotion_detailed(dummy_input)
        assert detailed["is_uncertain"] is True
        assert detailed["display_emotion"] in EMOTIONS


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


class TestYouTubeMusicProvider:
    def test_youtube_provider_initialization(self):
        yt_provider = YouTubeMusicProvider()
        success = yt_provider.initialize()
        assert success is True
        assert len(yt_provider.get_all_tracks()) >= 15

    def test_youtube_tracks_for_all_emotions(self):
        yt_provider = YouTubeMusicProvider()
        yt_provider.initialize()

        for emotion in EMOTIONS:
            tracks = yt_provider.get_tracks_by_emotion(emotion)
            assert len(tracks) >= 3
            for t in tracks:
                assert t.emotion_tag == emotion
                assert t.stream_url is not None
                assert "youtube.com" in t.stream_url
                assert "embed_url" in t.extra_metadata

    def test_youtube_recommender_auto_switch(self):
        yt_provider = YouTubeMusicProvider()
        yt_provider.initialize()
        recommender = MusicRecommender(yt_provider)
        recommender.initialize()

        assert recommender.current_track is not None
        
        # Test switching to sad YouTube track
        sad_track = recommender.on_emotion_update("sad", 0.88, 0.95)
        if sad_track:
            assert sad_track.emotion_tag == "sad"
            assert recommender.current_emotion == "sad"
            assert "youtube.com" in sad_track.stream_url
