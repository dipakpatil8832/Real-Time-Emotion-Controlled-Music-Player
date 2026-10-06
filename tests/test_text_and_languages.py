import io
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.text_emotion_classifier import TextEmotionClassifier
from src.music_engine.youtube_provider import YouTubeMusicProvider, MUSIC_LANGUAGES
from src.config import EMOTIONS

def test_text_and_multilingual():
    print("==================================================")
    print("🚀 TESTING TEXT EMOTION CLASSIFIER & MULTILINGUAL")
    print("==================================================")
    
    clf = TextEmotionClassifier()
    test_cases = [
        ("I am feeling so joyful and happy today, celebrating a great achievement!", "happy"),
        ("I feel lonely and sad, missing old times and crying.", "sad"),
        ("I am so angry and furious about what happened, feeling intense rage!", "angry"),
        ("Wow what an unbelievable, amazing and shocking surprise!", "surprise"),
        ("Today is a regular calm and peaceful quiet day.", "neutral")
    ]
    
    for text, expected in test_cases:
        res = clf.predict_emotion(text)
        print(f" -> Text: '{text[:35]}...' => {res['dominant_emotion']} ({res['confidence']*100:.1f}%) [Expected: {expected}]")
        assert res["dominant_emotion"] == expected, f"Expected {expected}, got {res['dominant_emotion']}"
    
    print("\n -> All text emotion tests passed! ✅")
    
    print("\n[Step 2] Testing Multilingual Track Provider across Hindi, English, Punjabi, LoFi...")
    yt = YouTubeMusicProvider()
    yt.initialize()
    
    for lang in ["Hindi", "English", "Punjabi", "Lo-Fi / Instrumental", "Spanish"]:
        for emo in EMOTIONS:
            tracks = yt.get_tracks_by_emotion(emo, language=lang)
            assert len(tracks) > 0, f"No tracks found for {lang} - {emo}"
        sample = yt.get_tracks_by_emotion("happy", language=lang)[0]
        print(f" -> {lang:<20s} Happy Track: '{sample.title}' ({sample.genre}) ✅")
        
    print("\n==================================================")
    print("🎉 ALL TEXT & MULTILINGUAL VERIFICATIONS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_text_and_multilingual()
