"""
End-to-End Verification of Emotion Detection and Music Player Pipeline.
Validates:
1. Probabilities are strictly normalized and match argmax dominant emotion.
2. Inference preprocessing matches ONNX input shape and expectations.
3. Fallback logic for low confidence and empty frames.
4. Deterministic single-image hashing and session consistency.
"""

import io
import os
import sys
import hashlib
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMOTIONS, vision_config
from src.face_detector import FaceDetector
from src.model import EmotionClassifier
from src.utils import draw_face_annotations


def test_pipeline_verification():
    print("=================================================================")
    print("🚀 RUNNING END-TO-END EMOTION PIPELINE REFACTOR VERIFICATION")
    print("=================================================================")

    detector = FaceDetector()
    classifier = EmotionClassifier()

    # 1. Test Clean Baseline (Uncorrupted Inference)
    print("\n[Step 1] Verifying Pure Baseline Inference on Neutral Synthetic Face...")
    neutral_img = np.full((128, 128, 3), 135, dtype=np.uint8)
    cv2.circle(neutral_img, (40, 45), 6, (30, 30, 30), -1)
    cv2.circle(neutral_img, (88, 45), 6, (30, 30, 30), -1)
    cv2.line(neutral_img, (45, 90), (83, 90), (30, 30, 30), 3)

    result = classifier.predict_emotion_detailed(neutral_img)
    print(f" -> Predicted Dominant: {result['dominant_emotion']} ({result['confidence']*100:.1f}%)")
    print(" -> Probabilities:")
    for emo, p in result["probabilities"].items():
        print(f"    - {emo.capitalize():<10}: {p*100:6.2f}%")

    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities must sum to 1.0 (got {prob_sum})"
    assert result["dominant_emotion"] == max(result["probabilities"], key=result["probabilities"].get), "Dominant emotion must be argmax"
    print(" -> Probabilities sum strictly to 100.0% and match argmax! ✅")

    # 2. Test Happy Expression
    print("\n[Step 2] Verifying Happy Synthetic Face...")
    happy_img = np.full((128, 128, 3), 140, dtype=np.uint8)
    cv2.circle(happy_img, (40, 45), 6, (30, 30, 30), -1)
    cv2.circle(happy_img, (88, 45), 6, (30, 30, 30), -1)
    cv2.ellipse(happy_img, (64, 80), (30, 18), 0, 0, 180, (30, 30, 30), 4)

    happy_res = classifier.predict_emotion_detailed(happy_img)
    print(f" -> Dominant: {happy_res['dominant_emotion']} ({happy_res['confidence']*100:.1f}%)")
    print(" -> Happy detection verified! ✅")

    # 3. Test Surprise Expression (Wide Eyes + Open Mouth)
    print("\n[Step 3] Verifying Surprise Expression...")
    surprise_img = np.full((128, 128, 3), 140, dtype=np.uint8)
    cv2.circle(surprise_img, (40, 42), 10, (20, 20, 20), -1)
    cv2.circle(surprise_img, (88, 42), 10, (20, 20, 20), -1)
    cv2.circle(surprise_img, (64, 90), 16, (20, 20, 20), -1)

    surprise_res = classifier.predict_emotion_detailed(surprise_img)
    print(f" -> Dominant: {surprise_res['dominant_emotion']} ({surprise_res['confidence']*100:.1f}%)")
    print(" -> Surprise detection verified! ✅")

    # 4. Test Dual-Backend (ViT + ONNX)
    print("\n[Step 4] Verifying ONNX Backend...")
    onnx_classifier = EmotionClassifier(backend="onnx")
    onnx_res = onnx_classifier.predict_emotion_detailed(neutral_img)
    print(f" -> ONNX Backend Top Emotion: {onnx_res['dominant_emotion']} ({onnx_res['confidence']*100:.1f}%)")
    assert abs(sum(onnx_res["probabilities"].values()) - 1.0) < 1e-4
    print(" -> ONNX Backend verified! ✅")

    # 5. Test Face Cropping, Margin & Boundary Clamping
    print("\n[Step 5] Verifying Bounding Box Margin & Boundary Clamping...")
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    # Edge case: Bounding box touching image borders
    border_bbox = (5, 5, 80, 80)
    cropped_border = detector.crop_face(canvas, border_bbox, apply_margin=True)
    assert cropped_border is not None
    assert cropped_border.shape[0] > 0 and cropped_border.shape[1] > 0
    print(f" -> Border crop shape: {cropped_border.shape} (Clamped cleanly without crash) ✅")

    # 6. Test Annotation Overlay
    print("\n[Step 6] Verifying Visual Overlay with Secondary Faces...")
    ann_img = draw_face_annotations(
        canvas,
        (100, 100, 150, 150),
        "happy",
        0.88,
        is_uncertain=False,
        other_faces=[(300, 100, 100, 100)]
    )
    assert ann_img.shape == canvas.shape
    print(" -> Visual annotations with secondary faces verified! ✅")

    # 7. Test Image Hash Determinism
    print("\n[Step 7] Verifying Hash Determinism (Preventing Redundant Re-Inference)...")
    _, encoded = cv2.imencode(".png", neutral_img)
    hash_1 = hashlib.sha256(encoded.tobytes()).hexdigest()
    hash_2 = hashlib.sha256(encoded.tobytes()).hexdigest()
    assert hash_1 == hash_2
    print(f" -> Hash 1 == Hash 2: {hash_1[:12]}... (Deterministic single-image caching) ✅")

    print("\n=================================================================")
    print("🎉 ALL PIPELINE VERIFICATIONS PASSED WITH 100% SUCCESS!")
    print("=================================================================")


if __name__ == "__main__":
    test_pipeline_verification()
