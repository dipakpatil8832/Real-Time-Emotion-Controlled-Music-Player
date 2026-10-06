import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add Windows PyTorch DLL directory if available
if sys.platform == "win32":
    import site
    for p in sys.path + (site.getsitepackages() if hasattr(site, "getsitepackages") else []) + ([site.getusersitepackages()] if hasattr(site, "getusersitepackages") else []):
        t_lib = Path(p) / "torch" / "lib"
        if t_lib.exists():
            try:
                os.add_dll_directory(str(t_lib))
            except Exception:
                pass
            os.environ["PATH"] = str(t_lib) + ";" + os.environ.get("PATH", "")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
import cv2
import tempfile


class TestEmotionMusicPlayer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.music_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_01_face_detector(self):
        print("\n[TEST 1] Testing FaceDetector...")
        detector = FaceDetector()
        self.assertIsNotNone(detector)
        
        # Test synthetic frame
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 60, (255, 255, 255), -1)
        bbox = (100, 100, 100, 100)
        cropped = detector.crop_face(img, bbox, apply_margin=True)
        self.assertIsNotNone(cropped)

        preprocessed_rgb = detector.preprocess_face_for_model(cropped, target_size=(224, 224), to_rgb=True)
        self.assertEqual(preprocessed_rgb.shape, (1, 224, 224, 3))
        
        preprocessed_onnx = detector.preprocess_face_for_model(cropped, target_size=(64, 64), to_rgb=False)
        self.assertEqual(preprocessed_onnx.shape, (1, 1, 64, 64))
        print(" -> FaceDetector: OK")

    def test_02_model_and_classifier(self):
        print("\n[TEST 2] Testing Model Architecture and Emotion Classifier...")
        config = ModelConfig(image_height=224, image_width=224, num_classes=5)
        model = build_emotion_model(config)
        self.assertIsNotNone(model)

        classifier = EmotionClassifier()
        dummy_face = np.random.uniform(0, 255, (1, 64, 64, 3)).astype(np.float32)
        dom_emotion, conf, probs = classifier.predict_emotion(dummy_face)
        
        self.assertIn(dom_emotion, EMOTIONS)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)

        # Test pure baseline prediction on neutral synthetic face
        neutral_img = np.full((64, 64, 3), 128, dtype=np.uint8)
        cv2.circle(neutral_img, (20, 25), 4, (30, 30, 30), -1)
        cv2.circle(neutral_img, (44, 25), 4, (30, 30, 30), -1)
        cv2.line(neutral_img, (22, 45), (42, 45), (30, 30, 30), 2)
        detailed = classifier.predict_emotion_detailed(neutral_img)
        self.assertIn("dominant_emotion", detailed)
        self.assertIn("display_emotion", detailed)
        self.assertAlmostEqual(sum(detailed["probabilities"].values()), 1.0, places=3)

        # Test threshold fallback
        strict_classifier = EmotionClassifier(confidence_threshold=0.99)
        strict_result = strict_classifier.predict_emotion_detailed(neutral_img)
        self.assertTrue(strict_result["is_uncertain"])
        self.assertEqual(strict_result["display_emotion"], "neutral")
        print(f" -> Classifier inference successful! Dominant: {dom_emotion} ({conf:.2f}), Neutral test: {detailed['dominant_emotion']} ({detailed['confidence']:.2f}), Threshold fallback: OK")

    def test_03_temporal_smoothing(self):
        print("\n[TEST 3] Testing Temporal Emotion Smoother...")
        smoother = EmotionSmoother(window_size=6, decay_factor=0.75)
        self.assertEqual(smoother.current_dominant_emotion, "neutral")

        # Simulate 5 frames of "happy"
        happy_probs = {"happy": 0.88, "sad": 0.03, "angry": 0.03, "surprise": 0.03, "neutral": 0.03}
        for _ in range(5):
            em, conf, probs, stab = smoother.update("happy", 0.88, happy_probs)

        self.assertEqual(em, "happy")
        self.assertTrue(conf > 0.6)
        self.assertTrue(stab >= 0.8)

        # Test reset
        smoother.reset()
        self.assertEqual(len(smoother.history), 0)
        self.assertEqual(smoother.current_dominant_emotion, "neutral")
        print(" -> Temporal Smoother (EMA + Sliding Window): OK")

    def test_04_audio_synthesizer(self):
        print("\n[TEST 4] Testing Procedural Audio Synthesizer...")
        synthesizer = ProceduralAudioSynthesizer(sample_rate=22050)
        tracks = synthesizer.generate_full_sample_library(self.music_dir)
        
        for em in EMOTIONS:
            self.assertIn(em, tracks)
            self.assertTrue(len(tracks[em]) >= 2)
            for f in tracks[em]:
                self.assertTrue(f.exists())
                self.assertTrue(f.stat().st_size > 1000)
        print(f" -> Audio Synthesizer generated {sum(len(t) for t in tracks.values())} tracks across all 5 emotions: OK")

    def test_05_local_provider_and_recommender(self):
        print("\n[TEST 5] Testing Music Provider & Recommendation Engine...")
        provider = LocalMusicProvider(self.music_dir)
        provider.initialize()
        
        all_tracks = provider.get_all_tracks()
        self.assertTrue(len(all_tracks) >= 5)

        recommender = MusicRecommender(provider)
        recommender.initialize()
        self.assertIsNotNone(recommender.current_track)

        # Test auto switch on emotion transition
        switched = recommender.on_emotion_update("happy", 0.90, 0.85)
        if switched:
            self.assertEqual(recommender.current_emotion, "happy")
            self.assertEqual(switched.emotion_tag, "happy")

        # Test next track
        next_trk = recommender.next_track()
        self.assertIsNotNone(next_trk)
        print(f" -> Music Provider & Recommender (Current: '{next_trk.title}'): OK")

    def test_06_youtube_music_provider(self):
        print("\n[TEST 6] Testing YouTube Music Provider & Emotion Streaming...")
        from src.music_engine.youtube_provider import YouTubeMusicProvider
        yt_provider = YouTubeMusicProvider()
        self.assertTrue(yt_provider.initialize())
        self.assertTrue(len(yt_provider.get_all_tracks()) >= 15)

        for emotion in EMOTIONS:
            tracks = yt_provider.get_tracks_by_emotion(emotion)
            self.assertTrue(len(tracks) >= 2)
            self.assertIn("youtube.com", tracks[0].stream_url)

        recommender = MusicRecommender(yt_provider)
        recommender.initialize()
        
        # Test auto switch to Sad YouTube track
        sad_trk = recommender.on_emotion_update("sad", 0.92, 0.85)
        if sad_trk:
            self.assertEqual(sad_trk.emotion_tag, "sad")
            self.assertIn("youtube.com", sad_trk.stream_url)
            print(f" -> YouTube Auto-Switch OK! Emotion: {sad_trk.emotion_tag} -> '{sad_trk.title}' ({sad_trk.stream_url})")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmotionMusicPlayer)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
