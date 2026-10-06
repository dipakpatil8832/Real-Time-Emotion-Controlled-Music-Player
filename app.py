"""
Real-Time Emotion-Controlled Music Player - Streamlit Dashboard.
Combines:
1. OpenCV & Deep Learning Vision Pipeline (FERPlus ONNX & Vision Transformer).
2. NLP Text Emotion Classifier for thought/journal/mood sentence analysis.
3. Multilingual Music Engine supporting Hindi, English, Punjabi, Lo-Fi, and Spanish songs.
"""

import hashlib
import importlib
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

import src.config
import src.face_detector
import src.model
import src.smoothing
import src.text_emotion_classifier
import src.utils

# Reload project modules on edit
importlib.reload(src.config)
importlib.reload(src.face_detector)
importlib.reload(src.model)
importlib.reload(src.smoothing)
importlib.reload(src.text_emotion_classifier)
importlib.reload(src.utils)

from src.config import (
    DEFAULT_GOOGLE_API_KEY,
    EMOTION_METADATA,
    EMOTIONS,
    GEMINI_DEFAULT_MODEL,
    MUSIC_DIR,
    model_config,
    music_config,
    vision_config
)
from src.face_detector import FaceDetector
from src.gemini_emotion_classifier import GeminiEmotionClassifier
from src.model import EmotionClassifier
from src.music_engine.base import BaseMusicProvider, Track
from src.music_engine.local_provider import LocalMusicProvider
from src.music_engine.recommender import MusicRecommender
from src.music_engine.youtube_provider import MUSIC_LANGUAGES, YouTubeMusicProvider
from src.text_emotion_classifier import TextEmotionClassifier
from src.utils import draw_face_annotations, logger, pil_to_cv2

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="EmotiBeat Controls | AI-Powered Emotion Music Experience",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    /* Dark glassmorphism custom styles */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #151a2e 50%, #0d111d 100%);
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .glass-card {
        background: rgba(26, 34, 52, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.45);
    }
    
    .metric-chip {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 6px;
    }

    .song-title {
        font-size: 24px;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .song-artist {
        font-size: 15px;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    .emotion-badge {
        font-size: 18px;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }

    /* Streamlit button polish */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# Cache Resource Singletons
@st.cache_resource(show_spinner="Initializing Vision & Face Detection Pipeline...")
def load_face_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource(show_spinner="Loading Vision Emotion Model Engine...")
def load_emotion_classifier(backend_type: str = "onnx") -> EmotionClassifier:
    return EmotionClassifier(backend=backend_type)


@st.cache_resource(show_spinner="Loading Text Emotion NLP Engine...")
def load_text_emotion_classifier() -> TextEmotionClassifier:
    return TextEmotionClassifier()


@st.cache_resource(show_spinner="Connecting to Google Gemini (ChatGoogleGenerativeAI)...")
def load_gemini_classifier(api_key: str = "", model_name: str = "gemini-3.6-flash") -> GeminiEmotionClassifier:
    return GeminiEmotionClassifier(api_key=api_key if api_key else None, model_name=model_name, temperature=0.0)


@st.cache_resource(show_spinner="Connecting Music Provider...")
def load_music_engine(engine_type: str = "YouTube", api_key: str = "") -> Tuple[BaseMusicProvider, MusicRecommender]:
    if engine_type == "YouTube":
        provider = YouTubeMusicProvider(api_key=api_key if api_key else None)
    else:
        provider = LocalMusicProvider(MUSIC_DIR)
    provider.initialize()
    recommender = MusicRecommender(provider)
    recommender.initialize()
    return provider, recommender


# Initialize Session State
if "model_engine" not in st.session_state:
    st.session_state.model_engine = "⚡ FERPlus ONNX (Instant 10ms - Fast & Accurate)"

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GOOGLE_API_KEY", DEFAULT_GOOGLE_API_KEY)

if "gemini_model_name" not in st.session_state:
    st.session_state.gemini_model_name = "gemini-3.6-flash"

if "text_model_engine" not in st.session_state:
    st.session_state.text_model_engine = "✨ Google Gemini 3.6 Flash (ChatGoogleGenerativeAI)"

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "All"

if "input_modality" not in st.session_state:
    st.session_state.input_modality = "📸 Facial Image Vision"

if "processed_image_hash" not in st.session_state:
    st.session_state.processed_image_hash = ""

if "processed_text_hash" not in st.session_state:
    st.session_state.processed_text_hash = ""

if "has_face" not in st.session_state:
    st.session_state.has_face = False

if "face_count" not in st.session_state:
    st.session_state.face_count = 0

if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"

if "dominant_raw_emotion" not in st.session_state:
    st.session_state.dominant_raw_emotion = "neutral"

if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.50

if "current_probs" not in st.session_state:
    st.session_state.current_probs = {e: 0.20 for e in EMOTIONS}

if "is_uncertain" not in st.session_state:
    st.session_state.is_uncertain = False

if "status_label" not in st.session_state:
    st.session_state.status_label = "Ready"

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if "text_explanation" not in st.session_state:
    st.session_state.text_explanation = ""

if "active_music_emotion" not in st.session_state:
    st.session_state.active_music_emotion = "neutral"

if "current_track_id" not in st.session_state:
    st.session_state.current_track_id = None

if "auto_switch" not in st.session_state:
    st.session_state.auto_switch = True

if "history_log" not in st.session_state:
    st.session_state.history_log = []

if "youtube_api_key" not in st.session_state:
    st.session_state.youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")

if "music_engine_type" not in st.session_state:
    st.session_state.music_engine_type = "YouTube"


# Load Engine Singletons
backend_key = "vit" if "ViT" in st.session_state.model_engine else "onnx"
detector = load_face_detector()
classifier = load_emotion_classifier(backend_key)
text_classifier = load_text_emotion_classifier()
gemini_classifier = load_gemini_classifier(st.session_state.gemini_api_key, st.session_state.gemini_model_name)
music_provider, recommender = load_music_engine(st.session_state.music_engine_type, st.session_state.youtube_api_key)

# Apply Language Filter to Provider
if hasattr(music_provider, "set_language_filter"):
    music_provider.set_language_filter(st.session_state.selected_language)


# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/music-robot.png", width=64)
    st.title("EmotiBeat Controls")
    st.caption("AI-Powered Multi-Modal Emotion Music Experience")

    st.markdown("---")
    st.subheader("🌐 Song Language & Type")
    
    lang_options = [
        "🌟 All Languages",
        "🇮🇳 Hindi (Bollywood & Indie)",
        "🇬🇧 English (Global Hits)",
        "🎵 Punjabi (Party & Soul)",
        "🎹 Lo-Fi / Instrumental Beats",
        "🇪🇸 Spanish / Latin Beats"
    ]
    lang_map = {
        "🌟 All Languages": "All",
        "🇮🇳 Hindi (Bollywood & Indie)": "Hindi",
        "🇬🇧 English (Global Hits)": "English",
        "🎵 Punjabi (Party & Soul)": "Punjabi",
        "🎹 Lo-Fi / Instrumental Beats": "Lo-Fi / Instrumental",
        "🇪🇸 Spanish / Latin Beats": "Spanish"
    }
    
    current_lang_idx = 0
    for idx, opt in enumerate(lang_options):
        if lang_map[opt] == st.session_state.selected_language:
            current_lang_idx = idx
            break

    selected_lang_label = st.selectbox(
        "Preferred Song Type",
        lang_options,
        index=current_lang_idx,
        help="Filter soundtrack recommendations by your preferred language and song genre."
    )
    new_lang = lang_map[selected_lang_label]
    if new_lang != st.session_state.selected_language:
        st.session_state.selected_language = new_lang
        if hasattr(music_provider, "set_language_filter"):
            music_provider.set_language_filter(new_lang)
        # Update current recommendation in new language
        new_track = recommender.get_recommendation(st.session_state.current_emotion, force_new=True, language=new_lang)
        if new_track:
            recommender.set_current_track(new_track)
            st.session_state.current_track_id = new_track.id
        st.rerun()

    st.markdown("---")
    st.subheader("🧠 Emotion Recognition Model")
    
    vision_models = [
        "⚡ FERPlus ONNX (Instant 10ms - Fast & Accurate)",
        "🤖 Vision Transformer (ViT SOTA - High Accuracy)",
        "✨ Google Gemini 3.6 Vision (ChatGoogleGenerativeAI)"
    ]
    current_m_idx = 0
    for idx, vm in enumerate(vision_models):
        if (vm == st.session_state.model_engine or 
            ("ONNX" in vm and "ONNX" in st.session_state.model_engine) or
            ("ViT" in vm and "ViT" in st.session_state.model_engine) or
            ("Gemini" in vm and "Gemini" in st.session_state.model_engine)):
            current_m_idx = idx
            break

    model_choice = st.selectbox(
        "AI Vision Architecture",
        vision_models,
        index=current_m_idx,
        help="Select model architecture: FERPlus ONNX (10ms offline), Vision Transformer (ViT deep attention), or Google Gemini 3.6 Vision (multimodal LLM)."
    )
    if model_choice != st.session_state.model_engine:
        st.session_state.model_engine = model_choice
        st.session_state.processed_image_hash = ""
        st.cache_resource.clear()
        st.rerun()

    with st.expander("✨ Google Gemini Settings (ChatGoogleGenerativeAI)"):
        st.markdown("**Powered by Google DeepMind Gemini**")
        gemini_key_input = st.text_input(
            "Google API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIzaSy... or your Gemini API Key",
            help="Pre-configured Google Gemini API key for multimodal vision & deep sentiment reasoning."
        )
        if gemini_key_input != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = gemini_key_input
            gemini_classifier.set_api_key(gemini_key_input)
            st.cache_resource.clear()
            st.rerun()

        model_name_choice = st.selectbox(
            "Gemini Model",
            ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-3.5-flash", "gemini-1.5-flash", "gemini-2.5-pro"],
            index=0,
            help="Select the Gemini model version."
        )
        if model_name_choice != st.session_state.gemini_model_name:
            st.session_state.gemini_model_name = model_name_choice
            gemini_classifier.set_model_name(model_name_choice)
            st.cache_resource.clear()
            st.rerun()

        if gemini_classifier.is_configured():
            st.success("🟢 Gemini API is Connected & Ready")
        else:
            st.warning("⚠️ Enter Google API Key to enable Gemini.")

    st.markdown("---")
    st.subheader("🎧 Music Source & Streaming")
    
    selected_source = st.radio(
        "Audio Engine",
        ["🎥 YouTube Music (Auto Stream)", "📁 Local Audio Library"],
        index=0 if st.session_state.music_engine_type == "YouTube" else 1,
        help="Select between curated YouTube music playback and offline local library audio."
    )
    engine_choice = "YouTube" if "YouTube" in selected_source else "Local"
    if engine_choice != st.session_state.music_engine_type:
        st.session_state.music_engine_type = engine_choice
        st.cache_resource.clear()
        st.rerun()

    if engine_choice == "YouTube":
        with st.expander("🔑 YouTube Data API Key (Optional)"):
            custom_key = st.text_input(
                "API Key",
                value=st.session_state.youtube_api_key,
                type="password",
                placeholder="AIzaSy...",
                help="Optional: Paste Google YouTube Data API v3 key for live customized queries."
            )
            if custom_key != st.session_state.youtube_api_key:
                st.session_state.youtube_api_key = custom_key
                st.cache_resource.clear()
                st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Vision & Model Calibration")
    
    preset_mode = st.selectbox(
        "🎯 Expression Calibration Preset",
        options=[
            "Balanced Emotion Recognition (Zero Bias - Recommended)",
            "High Sensitivity (Subtle Pouts & Micro-Expressions)",
            "Raw Softmax (Strict Output)",
            "Custom Calibration"
        ],
        index=0,
        help="Fine-tune sensitivity. 'Balanced Emotion Recognition' ensures anger, sadness, happiness, surprise, and neutrality are detected without bias."
    )

    if preset_mode == "Balanced Emotion Recognition (Zero Bias - Recommended)":
        default_neutral_bias = 0.00
        default_sad_boost = 0.00
        default_sens = 1.00
    elif preset_mode == "High Sensitivity (Subtle Pouts & Micro-Expressions)":
        default_neutral_bias = 0.50
        default_sad_boost = 0.40
        default_sens = 1.15
    elif preset_mode == "Raw Softmax (Strict Output)":
        default_neutral_bias = 0.00
        default_sad_boost = 0.00
        default_sens = 1.00
    else:
        default_neutral_bias = float(getattr(vision_config, "neutral_logit_bias", 0.00))
        default_sad_boost = float(getattr(vision_config, "sadness_boost", 0.00))
        default_sens = float(getattr(vision_config, "emotion_sensitivity", 1.00))

    confidence_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.15,
        max_value=0.85,
        value=float(getattr(vision_config, "confidence_threshold", 0.30)),
        step=0.05,
        help="Minimum probability required for confident emotion display."
    )
    classifier.confidence_threshold = confidence_thresh

    if preset_mode == "Custom Calibration":
        neutral_logit_bias = st.slider(
            "Neutral Bias Suppression",
            min_value=0.0,
            max_value=2.0,
            value=default_neutral_bias,
            step=0.05,
            help="Subtly penalizes baseline neutral over-representation in logit space."
        )
        sadness_boost_val = st.slider(
            "Sadness Micro-Expression Boost",
            min_value=0.0,
            max_value=1.0,
            value=default_sad_boost,
            step=0.05,
            help="Sensitivity boost for gentle sad expressions."
        )
        emotion_sens = st.slider(
            "Active Emotion Sensitivity",
            min_value=0.8,
            max_value=2.0,
            value=default_sens,
            step=0.05,
            help="Multiplier for active emotions."
        )
    else:
        neutral_logit_bias = default_neutral_bias
        sadness_boost_val = default_sad_boost
        emotion_sens = default_sens

    classifier.neutral_logit_bias = neutral_logit_bias
    classifier.sadness_boost = sadness_boost_val
    classifier.emotion_sensitivity = emotion_sens

    st.markdown("---")
    st.subheader("🎛️ Playback Mode")
    
    auto_pilot = st.toggle(
        "Auto-Pilot (Emotion Track Matching)",
        value=st.session_state.auto_switch,
        help="Automatically changes playlist and soundtrack when newly detected emotion changes."
    )
    st.session_state.auto_switch = auto_pilot

    manual_override = st.selectbox(
        "Manual Emotion Override",
        options=["None (Use AI Detection)"] + [e.capitalize() for e in EMOTIONS],
        index=0,
        help="Force a specific mood override manually without changing image/text analysis."
    )

    if manual_override != "None (Use AI Detection)":
        override_key = manual_override.lower()
        if override_key != st.session_state.current_emotion:
            st.session_state.current_emotion = override_key
            st.session_state.active_music_emotion = override_key
            new_track = recommender.get_recommendation(override_key, force_new=True, language=st.session_state.selected_language)
            if new_track:
                recommender.set_current_track(new_track)
                st.session_state.current_track_id = new_track.id
                st.session_state.history_log.append({
                    "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "emotion": override_key,
                    "trigger": "Manual Override",
                    "track": new_track.title
                })
    
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Rescan Music", use_container_width=True):
            music_provider.initialize()
            st.success("Library refreshed!")
    with col_btn2:
        if st.button("⚡ Reset Session", use_container_width=True):
            st.session_state.processed_image_hash = ""
            st.session_state.processed_text_hash = ""
            st.session_state.annotated_image = None
            st.session_state.has_face = False
            st.session_state.current_emotion = "neutral"
            st.cache_resource.clear()
            st.rerun()


# Main Dashboard Header
st.title("🎭 EmotiBeat – AI Emotion-Controlled Music Player")
st.markdown("Real-Time Multi-Modal Emotion Recognition via **Facial Vision** and **NLP Text Sentiment**, dynamically curating **Hindi, English, Punjabi, Lo-Fi, and Spanish** soundtracks.")

tab_player, tab_analytics, tab_architecture = st.tabs(["🎧 Emotion Player", "📊 Session Analytics", "🧠 ML Pipeline & Architecture"])

with tab_player:
    col_input, col_player = st.columns([1.15, 0.85], gap="large")

    # LEFT COLUMN: Input Modality (Vision vs Text)
    with col_input:
        st.markdown("### 🎛️ Emotion Detection Mode")
        
        input_choice = st.radio(
            "Select Emotion Input Modality:",
            ["📸 Facial Image Vision", "✍️ Text Mood & Thought Journal"],
            horizontal=True
        )
        st.session_state.input_modality = input_choice

        # ==================== MODE 1: FACIAL IMAGE VISION ====================
        if input_choice == "📸 Facial Image Vision":
            img_source = st.radio(
                "Image Source",
                ["📁 Upload Image File", "🖼️ Preset Test Faces (Ready to Test)", "📸 Capture Photo with Camera"],
                horizontal=True
            )

            raw_image_bytes = None
            pil_image = None

            if img_source == "📸 Capture Photo with Camera":
                camera_img = st.camera_input("Capture facial expression:")
                if camera_img is not None:
                    raw_image_bytes = camera_img.getvalue()
                    pil_image = Image.open(io.BytesIO(raw_image_bytes))
            elif img_source == "🖼️ Preset Test Faces (Ready to Test)":
                preset_choices = {
                    "😄 Happy Face": "sample_test_images/happy_test.jpg",
                    "😢 Sad / Melancholy Face": "sample_test_images/sad_test.jpg",
                    "🔥 Angry / Frustrated Face": "sample_test_images/angry_test.jpg",
                    "⚡ Surprised Face": "sample_test_images/surprise_test.jpg",
                    "🌿 Neutral Face": "sample_test_images/neutral_test.jpg"
                }
                selected_preset_label = st.selectbox(
                    "Select a sample test expression:",
                    list(preset_choices.keys())
                )
                chosen_path = Path(preset_choices[selected_preset_label])
                if chosen_path.exists():
                    raw_image_bytes = chosen_path.read_bytes()
                    pil_image = Image.open(io.BytesIO(raw_image_bytes))
            else:
                uploaded_file = st.file_uploader("Upload a face image (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp"])
                if uploaded_file is not None:
                    raw_image_bytes = uploaded_file.getvalue()
                    pil_image = Image.open(io.BytesIO(raw_image_bytes))

            if raw_image_bytes is not None and pil_image is not None:
                image_hash = hashlib.sha256(raw_image_bytes).hexdigest()

                if image_hash != st.session_state.processed_image_hash:
                    frame_bgr = pil_to_cv2(pil_image)
                    detected_faces = detector.detect_faces(frame_bgr)

                    if detected_faces:
                        primary_bbox = detected_faces[0]
                        other_bboxes = detected_faces[1:]
                        face_roi = detector.crop_face(frame_bgr, primary_bbox, apply_margin=True)

                        if "Gemini" in st.session_state.model_engine:
                            pred = gemini_classifier.predict_vision_emotion(face_roi)
                            if pred.get("explanation"):
                                st.session_state.text_explanation = pred["explanation"]
                        else:
                            pred = classifier.predict_emotion_detailed(
                                face_roi,
                                neutral_bias=neutral_logit_bias,
                                sad_boost=sadness_boost_val,
                                sensitivity=emotion_sens
                            )

                        dominant_emo = pred["dominant_emotion"]
                        display_emo = pred.get("display_emotion", dominant_emo)
                        conf = pred["confidence"]
                        probs = pred["probabilities"]
                        is_uncert = pred.get("is_uncertain", False)

                        exp_primary_bbox = detector.get_expanded_bbox(primary_bbox, (frame_bgr.shape[0], frame_bgr.shape[1]))
                        exp_other_bboxes = [detector.get_expanded_bbox(ob, (frame_bgr.shape[0], frame_bgr.shape[1])) for ob in other_bboxes]

                        annotated_bgr = draw_face_annotations(
                            frame_bgr,
                            exp_primary_bbox,
                            display_emo if is_uncert else dominant_emo,
                            conf,
                            all_probabilities=probs,
                            is_uncertain=is_uncert,
                            other_faces=exp_other_bboxes
                        )
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                        st.session_state.processed_image_hash = image_hash
                        st.session_state.has_face = True
                        st.session_state.face_count = len(detected_faces)
                        st.session_state.annotated_image = annotated_rgb
                        st.session_state.dominant_raw_emotion = dominant_emo
                        st.session_state.current_emotion = display_emo
                        st.session_state.current_confidence = conf
                        st.session_state.current_probs = probs
                        st.session_state.is_uncertain = is_uncert
                        st.session_state.status_label = pred.get("status_label", f"{display_emo.capitalize()} ({conf * 100:.1f}%)")

                        if manual_override == "None (Use AI Detection)" and st.session_state.auto_switch:
                            if display_emo != st.session_state.active_music_emotion or recommender.current_track is None:
                                st.session_state.active_music_emotion = display_emo
                                new_track = recommender.get_recommendation(display_emo, force_new=True, language=st.session_state.selected_language)
                                if new_track:
                                    recommender.set_current_track(new_track)
                                    st.session_state.current_track_id = new_track.id
                                    st.session_state.history_log.append({
                                        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                                        "emotion": display_emo,
                                        "trigger": f"{'Gemini Vision' if 'Gemini' in st.session_state.model_engine else 'AI Vision'} ({conf * 100:.1f}%)",
                                        "track": new_track.title
                                    })
                    else:
                        st.session_state.processed_image_hash = image_hash
                        st.session_state.has_face = False
                        st.session_state.face_count = 0
                        st.session_state.annotated_image = np.array(pil_image.convert("RGB"))
                        st.session_state.is_uncertain = True
                        st.session_state.status_label = "⚠️ No Face Detected"

            if st.session_state.annotated_image is not None:
                st.image(st.session_state.annotated_image, use_container_width=True)
                if st.session_state.has_face:
                    face_badge_txt = "👤 1 Face Analyzed" if st.session_state.face_count == 1 else f"👥 {st.session_state.face_count} Faces Detected"
                    if st.session_state.is_uncertain:
                        st.warning(f"{face_badge_txt} • **Low Confidence Detection**")
                    else:
                        st.success(f"{face_badge_txt} • Expression: **{st.session_state.current_emotion.capitalize()}** ({st.session_state.current_confidence * 100:.1f}%)")
                else:
                    st.warning("⚠️ No face detected in the image.")
            else:
                st.info("📸 Capture a photo or upload an image to begin facial emotion recognition.")

        # ==================== MODE 2: TEXT EMOTION & JOURNAL ====================
        else:
            st.markdown("#### ✍️ Express Your Current Thoughts or Mood")
            st.caption("Type a sentence, diary thought, or how your day was (English, Hindi/Hinglish supported).")

            # LLM Model Selector for Text Mode
            text_engine_choice = st.radio(
                "NLP Emotion Model:",
                [
                    "✨ Google Gemini 3.6 Flash (ChatGoogleGenerativeAI - SOTA)",
                    "⚡ Fast Local Lexicon NLP"
                ],
                horizontal=True,
                index=0 if "Gemini" in st.session_state.text_model_engine else 1
            )
            st.session_state.text_model_engine = text_engine_choice

            # Quick Prompt Presets
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                if st.button("😄 'Feeling so happy today!'", use_container_width=True):
                    st.session_state.user_text_input = "I am feeling so joyful and happy today, celebrating a great achievement!"
            with col_p2:
                if st.button("😢 'Feeling lonely & sad'", use_container_width=True):
                    st.session_state.user_text_input = "I feel lonely and sad, missing old times and crying."
            with col_p3:
                if st.button("🔥 'So angry & frustrated'", use_container_width=True):
                    st.session_state.user_text_input = "I am so angry and furious about what happened, feeling intense rage!"

            default_text = st.session_state.get("user_text_input", "")
            user_text = st.text_area(
                "How are you feeling right now?",
                value=default_text,
                placeholder="e.g., Today was an amazing and wonderful day, I loved it! OR I am feeling exhausted and sad...",
                height=110
            )

            if st.button("🔍 Analyze Text Emotion & Play Music", type="primary", use_container_width=True):
                if user_text.strip():
                    text_hash = hashlib.sha256((user_text + text_engine_choice).encode("utf-8")).hexdigest()
                    
                    if "Gemini" in text_engine_choice and gemini_classifier.is_configured():
                        with st.spinner("🤖 Gemini 3.6 Flash analyzing emotional sentiment..."):
                            pred = gemini_classifier.predict_text_emotion(user_text)
                    else:
                        pred = text_classifier.predict_emotion(user_text)

                    dom_emo = pred["dominant_emotion"]
                    conf = pred["confidence"]
                    probs = pred["probabilities"]

                    st.session_state.processed_text_hash = text_hash
                    st.session_state.dominant_raw_emotion = dom_emo
                    st.session_state.current_emotion = dom_emo
                    st.session_state.current_confidence = conf
                    st.session_state.current_probs = probs
                    st.session_state.is_uncertain = False
                    st.session_state.text_explanation = pred["explanation"]
                    st.session_state.status_label = f"Text Sentiment: {dom_emo.capitalize()} ({conf * 100:.1f}%)"

                    if manual_override == "None (Use AI Detection)" and st.session_state.auto_switch:
                        st.session_state.active_music_emotion = dom_emo
                        new_track = recommender.get_recommendation(dom_emo, force_new=True, language=st.session_state.selected_language)
                        if new_track:
                            recommender.set_current_track(new_track)
                            st.session_state.current_track_id = new_track.id
                            st.session_state.history_log.append({
                                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                                "emotion": dom_emo,
                                "trigger": f"Text NLP ({conf * 100:.1f}%)",
                                "track": new_track.title
                            })
                    st.rerun()

            if st.session_state.text_explanation:
                st.success(f"💡 **Analysis Insight**: {st.session_state.text_explanation}")

        # Real-Time Emotion Probabilities Section
        st.markdown("#### 📊 Real-Time Emotion Probabilities")
        curr_probs = st.session_state.current_probs
        curr_dominant = st.session_state.dominant_raw_emotion

        for emotion in EMOTIONS:
            meta = EMOTION_METADATA[emotion]
            prob = curr_probs.get(emotion, 0.0)
            is_top = (emotion == curr_dominant)
            
            col_lbl, col_bar = st.columns([1.1, 3.0])
            with col_lbl:
                label_display = f"{meta['emoji']} **{emotion.capitalize()}**"
                if is_top:
                    label_display += " ⭐"
                st.write(label_display)
            with col_bar:
                st.progress(float(min(1.0, max(0.0, prob))), text=f"{prob * 100:.1f}%")

    # RIGHT COLUMN: Music Player & Synchronized Recommendation
    with col_player:
        st.markdown(f"### 🎵 Now Playing • {st.session_state.selected_language}")

        effective_emotion = st.session_state.current_emotion
        meta = EMOTION_METADATA.get(effective_emotion, EMOTION_METADATA["neutral"])

        # Ensure active track matches session state & language
        active_track = recommender.current_track
        if active_track is None or (st.session_state.current_track_id and active_track.id != st.session_state.current_track_id):
            if st.session_state.current_track_id:
                matching = [t for t in music_provider.get_all_tracks() if t.id == st.session_state.current_track_id]
                if matching:
                    active_track = matching[0]
                    recommender.set_current_track(active_track)
            if active_track is None:
                active_track = recommender.get_recommendation(effective_emotion, force_new=False, language=st.session_state.selected_language)
                if active_track:
                    recommender.set_current_track(active_track)
                    st.session_state.current_track_id = active_track.id

        # Glassmorphism Player Card
        is_uncert_card = st.session_state.is_uncertain
        card_badge_color = meta["color"] if not is_uncert_card else "#F59E0B"
        mood_title = effective_emotion.upper() if not is_uncert_card else f"{effective_emotion.upper()} (UNCERTAIN)"

        st.markdown(f"""
        <div class="glass-card">
            <div class="emotion-badge" style="background-color: {card_badge_color}22; color: {card_badge_color}; border: 1px solid {card_badge_color}66;">
                <span>{meta['emoji']}</span>
                <span>Current Mood: {mood_title}</span>
            </div>
            <p class="song-title">{(active_track.title if active_track else 'Select a Soundtrack')}</p>
            <p class="song-artist">👤 {(active_track.artist if active_track else 'EmotiBeat Audio Engine')} • 🏷️ {(active_track.genre if active_track else meta['genre'])}</p>
            <div>
                <span class="metric-chip" style="background-color: #1e293b; color: #38bdf8; border: 1px solid #334155;">⚡ Confidence: {st.session_state.current_confidence * 100:.1f}%</span>
                <span class="metric-chip" style="background-color: #1e293b; color: #34d399; border: 1px solid #334155;">🌐 Type: {st.session_state.selected_language}</span>
                <span class="metric-chip" style="background-color: #1e293b; color: #a78bfa; border: 1px solid #334155;">🎯 Source: {st.session_state.input_modality.split()[1]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Audio / Video Playback Frame
        if active_track:
            if active_track.stream_url and "youtube.com" in active_track.stream_url:
                embed_url = active_track.extra_metadata.get(
                    "embed_url",
                    f"https://www.youtube.com/embed/{active_track.id}?autoplay=1&enablejsapi=1"
                )
                components.html(
                    f"""
                    <div style="position: relative; width: 100%; border-radius: 14px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.6);">
                        <iframe width="100%" height="280" 
                                src="{embed_url}" 
                                title="{active_track.title}" 
                                frameborder="0" 
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                                allowfullscreen 
                                style="border-radius: 14px;">
                        </iframe>
                    </div>
                    """,
                    height=300
                )
                st.caption(f"🎥 Streaming from YouTube • [Watch on YouTube]({active_track.stream_url})")
            else:
                audio_bytes = music_provider.get_track_audio_bytes(active_track)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav", start_time=0)
                else:
                    st.info("🎵 Audio stream ready.")

        # Interactive Controls
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("⏭️ Next Song", use_container_width=True):
                next_trk = recommender.get_recommendation(effective_emotion, force_new=True, language=st.session_state.selected_language)
                if next_trk:
                    recommender.set_current_track(next_trk)
                    st.session_state.current_track_id = next_trk.id
                    st.session_state.history_log.append({
                        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "emotion": effective_emotion,
                        "trigger": "Next Song Button",
                        "track": next_trk.title
                    })
                    st.rerun()

        with btn_col2:
            if st.button("🔀 Smart Selection", use_container_width=True):
                new_trk = recommender.get_recommendation(effective_emotion, force_new=True, language=st.session_state.selected_language)
                if new_trk:
                    recommender.set_current_track(new_trk)
                    st.session_state.current_track_id = new_trk.id
                    st.session_state.history_log.append({
                        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "emotion": effective_emotion,
                        "trigger": "Smart Selection",
                        "track": new_trk.title
                    })
                    st.rerun()

        with btn_col3:
            if st.button("🧹 Clear Input", use_container_width=True):
                st.session_state.processed_image_hash = ""
                st.session_state.processed_text_hash = ""
                st.session_state.annotated_image = None
                st.session_state.text_explanation = ""
                st.session_state.has_face = False
                st.rerun()

        # Playlist for Current Emotion & Selected Language
        st.markdown(f"#### 📑 {st.session_state.selected_language} Playlist for {effective_emotion.capitalize()} Mood")
        matching_tracks = music_provider.get_tracks_by_emotion(effective_emotion, language=st.session_state.selected_language)
        
        if matching_tracks:
            for idx, trk in enumerate(matching_tracks):
                is_active = active_track and trk.id == active_track.id
                prefix = "▶️ **[PLAYING]** " if is_active else f"{idx+1}. "
                
                c_thumb, c_info, c_act = st.columns([1, 4, 1.2])
                with c_thumb:
                    if trk.cover_art_url:
                        st.image(trk.cover_art_url, width=54)
                    else:
                        st.write("🎵")
                with c_info:
                    st.markdown(f"{prefix}**{trk.title}**  \n<span style='font-size:12px; color:#94a3b8;'>{trk.artist} • {trk.genre}</span>", unsafe_allow_html=True)
                with c_act:
                    if not is_active:
                        if st.button("Play", key=f"play_btn_{trk.id}_{idx}"):
                            recommender.set_current_track(trk)
                            st.session_state.current_track_id = trk.id
                            st.session_state.history_log.append({
                                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                                "emotion": effective_emotion,
                                "trigger": "Playlist Item Play",
                                "track": trk.title
                            })
                            st.rerun()
                    else:
                        st.markdown("<span style='font-size:13px; color:#10b981; font-weight:700;'>Active</span>", unsafe_allow_html=True)
        else:
            st.info("No matching tracks in library.")


# TAB 2: Analytics & History
with tab_analytics:
    st.markdown("### 📊 Session Analytics & Emotion Log")
    if st.session_state.history_log:
        history_df = pd.DataFrame(st.session_state.history_log)
        st.dataframe(history_df, use_container_width=True)

        col_pie, col_stats = st.columns(2)
        with col_pie:
            st.markdown("#### Mood Frequency Distribution")
            emotion_counts = history_df["emotion"].value_counts()
            st.bar_chart(emotion_counts)
        with col_stats:
            st.markdown("#### Session Summary")
            st.metric("Total Tracks Triggered", len(history_df))
            st.metric("Active Emotion", st.session_state.current_emotion.capitalize())
            st.metric("Selected Language", st.session_state.selected_language)
            st.metric("Current Model Confidence", f"{st.session_state.current_confidence * 100:.1f}%")
    else:
        st.info("No playback history recorded yet in this session.")


# TAB 3: Model & Architecture
with tab_architecture:
    st.markdown("### 🧠 Multi-Modal Deep Learning & NLP Architecture")
    st.markdown("""
    #### 1. Dual Emotion Recognition Pipelines
    - **Vision Pipeline (Facial Expression)**:
      - OpenCV multi-scale face detector with 15% expanded bounding box.
      - **FERPlus ONNX Deep CNN** running via OpenCV DNN (`~10ms` latency).
      - **Vision Transformer (ViT SOTA)** fine-tuned on AffectNet/FER2013 for subtle micro-expression detection.
    - **NLP Pipeline (Text Sentiment & Mood Journal)**:
      - Multi-lingual semantic sentiment engine analyzing thoughts, diary sentences, and feelings in English and Hindi/Hinglish.
      - Negation-aware keyword intensity scoring and Softmax normalization.

    #### 2. Multilingual Music Recommendation Engine
    - Categorized emotion playlist mapping across **Hindi (Bollywood & Indie)**, **English (Global Hits)**, **Punjabi (Party & Soul)**, **Lo-Fi / Instrumental**, and **Spanish**.
    - Real-time YouTube Data API v3 integration with curated fallback streaming.
    """)
