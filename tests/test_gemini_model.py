import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.gemini_emotion_classifier import GeminiEmotionClassifier


class TestGeminiEmotionClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = GeminiEmotionClassifier()

    def test_classifier_initialization(self):
        self.assertTrue(self.classifier.is_configured())
        self.assertEqual(self.classifier.model_name, "gemini-3.6-flash")
        self.assertEqual(self.classifier.temperature, 0.0)

    def test_empty_text_fallback(self):
        result = self.classifier.predict_text_emotion("")
        self.assertEqual(result["dominant_emotion"], "neutral")
        self.assertIn("probabilities", result)

    def test_gemini_happy_prediction(self):
        result = self.classifier.predict_text_emotion("I passed all my engineering exams with top marks! So happy!")
        self.assertEqual(result["dominant_emotion"], "happy")
        self.assertGreaterEqual(result["confidence"], 0.70)
        self.assertIn("happy", result["probabilities"])

    def test_gemini_sad_prediction(self):
        result = self.classifier.predict_text_emotion("I feel completely broken and crying alone tonight...")
        self.assertEqual(result["dominant_emotion"], "sad")
        self.assertGreaterEqual(result["confidence"], 0.70)


if __name__ == "__main__":
    unittest.main()
