"""
Real-Time Emotion-Controlled Music Player - Streamlit Dashboard.
Combines OpenCV face detection, MobileNetV2 emotion classification,
temporal emotion smoothing, and an extensible music playback engine.
"""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.config import (
    EMOTION_METADATA,
    EMOTIONS,
    MUSIC_DIR,
    model_config,
    music_config,
    smoothing_config,
    vision_config
)
from src.face_detector import FaceDetector
from src.model import EmotionClassifier
from src.music_engine import LocalMusicProvider, MusicRecommender, Track
from src.smoothing import EmotionSmoother
from src.utils import draw_face_annotations, logger, pil_to_cv2

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="EmotiBeat | Emotion-Controlled Music Player",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    /* Dark glassmorphism custom styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
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
        font-size: 20px;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


import importlib
import src.config
import src.model
import src.face_detector
import src.smoothing

# Ensure modules are freshly loaded from disk
importlib.reload(src.config)
importlib.reload(src.model)
importlib.reload(src.face_detector)
importlib.reload(src.smoothing)

# Cache Resource Singletons
@st.cache_resource(show_spinner="Initializing AI Vision & ML Models...")
def load_face_detector() -> FaceDetector:
    return src.face_detector.FaceDetector()


@st.cache_resource(show_spinner="Loading Emotion Recognition Classifier...")
def load_emotion_classifier() -> EmotionClassifier:
    return src.model.EmotionClassifier()


@st.cache_resource(show_spinner="Loading Local Music Library...")
def load_music_engine() -> Tuple[LocalMusicProvider, MusicRecommender]:
    provider = LocalMusicProvider(MUSIC_DIR)
    provider.initialize()
    recommender = MusicRecommender(provider)
    recommender.initialize()
    return provider, recommender


# Initialize session state variables
if "smoother" not in st.session_state:
    st.session_state.smoother = EmotionSmoother()

if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"

if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.85

if "stability_index" not in st.session_state:
    st.session_state.stability_index = 1.0

if "current_probs" not in st.session_state:
    st.session_state.current_probs = {e: 0.2 for e in EMOTIONS}

if "auto_switch" not in st.session_state:
    st.session_state.auto_switch = True

if "manual_override_emotion" not in st.session_state:
    st.session_state.manual_override_emotion = None

if "history_log" not in st.session_state:
    st.session_state.history_log = []


# Load Singletons
detector = load_face_detector()
classifier = load_emotion_classifier()
music_provider, recommender = load_music_engine()
recommender.auto_switch_enabled = st.session_state.auto_switch


# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/music-robot.png", width=64)
    st.title("EmotiBeat Controls")
    st.caption("AI-Powered Real-Time Music Experience")

    st.markdown("---")
    st.subheader("⚙️ Vision & ML Settings")
    
    confidence_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.20,
        max_value=0.90,
        value=vision_config.confidence_threshold,
        step=0.05,
        help="Minimum neural network confidence required to register an emotion."
    )
    classifier.confidence_threshold = confidence_thresh

    emotion_sens = st.slider(
        "Emotion Sensitivity",
        min_value=1.0,
        max_value=2.5,
        value=float(getattr(vision_config, "emotion_sensitivity", 1.25)),
        step=0.05,
        help="Multiplies sensitivity for active emotions (Happy, Sad, Angry, Surprise)."
    )
    classifier.emotion_sensitivity = emotion_sens

    neutral_suppression = st.slider(
        "Neutral Bias Attenuation",
        min_value=0.30,
        max_value=1.0,
        value=float(getattr(vision_config, "neutral_suppression_factor", 0.70)),
        step=0.05,
        help="Lowers neutral dominance to ensure natural facial expressions are recognized."
    )
    classifier.neutral_bias_weight = neutral_suppression

    smoothing_window = st.slider(
        "Smoothing Buffer Window",
        min_value=3,
        max_value=25,
        value=smoothing_config.window_size,
        step=1,
        help="Number of historical video frames to average over for stable playback."
    )
    st.session_state.smoother.window_size = smoothing_window

    st.markdown("---")
    st.subheader("🎛️ Playback Mode")
    
    auto_pilot = st.toggle(
        "Auto-Pilot (Emotion Switching)",
        value=st.session_state.auto_switch,
        help="Automatically changes music when a new stable emotion is detected."
    )
    st.session_state.auto_switch = auto_pilot
    recommender.auto_switch_enabled = auto_pilot

    manual_override = st.selectbox(
        "Manual Emotion Override",
        options=["None (Use AI Vision)"] + [e.capitalize() for e in EMOTIONS],
        index=0
    )

    if manual_override != "None (Use AI Vision)":
        override_key = manual_override.lower()
        if override_key != st.session_state.current_emotion:
            st.session_state.current_emotion = override_key
            new_track = recommender.get_recommendation(override_key, force_new=True)
            if new_track:
                recommender.set_current_track(new_track)
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
        if st.button("⚡ Reload Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()


# Main Dashboard Header
st.title("🎭 Real-Time Emotion-Controlled Music Player")
st.markdown("Webcam-driven facial emotion classification with MobileNetV2, temporal smoothing, and automatic soundtrack synthesis.")

tab_player, tab_analytics, tab_architecture = st.tabs(["🎧 Live Player & Vision", "📊 Analytics & History", "🧠 Model & Architecture"])

with tab_player:
    col_vision, col_player = st.columns([1.1, 0.9], gap="large")

    # LEFT COLUMN: Vision & Webcam Interface
    with col_vision:
        st.markdown("### 📷 Camera & Facial Expression")
        
        camera_input_mode = st.radio(
            "Camera Input Mode",
            ["Live Camera Snapshot", "Image Upload Test"],
            horizontal=True
        )

        detected_face_roi = None
        annotated_frame = None

        if camera_input_mode == "Live Camera Snapshot":
            camera_img = st.camera_input("Capture live expression to evaluate:")
            if camera_img is not None:
                pil_img = Image.open(camera_img)
                frame = pil_to_cv2(pil_img)
                primary_face = detector.get_primary_face(frame)

                if primary_face is not None:
                    detected_face_roi = detector.crop_face(frame, primary_face)
                    # Predict emotion via Deep Neural Network with calibrated class balance
                    raw_emotion, raw_conf, raw_probs = classifier.predict_emotion(
                        detected_face_roi,
                        neutral_weight=neutral_suppression,
                        sensitivity=emotion_sens
                    )

                    # Apply Temporal Smoothing
                    s_emotion, s_conf, s_probs, stability = st.session_state.smoother.update(
                        raw_emotion, raw_conf, raw_probs
                    )

                    st.session_state.current_emotion = s_emotion
                    st.session_state.current_confidence = s_conf
                    st.session_state.current_probs = s_probs
                    st.session_state.stability_index = stability

                    # Check for Recommender Track Switch
                    if st.session_state.auto_switch and manual_override == "None (Use AI Vision)":
                        switched_track = recommender.on_emotion_update(s_emotion, s_conf, stability)
                        if switched_track:
                            st.session_state.history_log.append({
                                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                                "emotion": s_emotion,
                                "trigger": f"AI Vision ({s_conf*100:.1f}%)",
                                "track": switched_track.title
                            })

                    # Annotate frame
                    annotated_frame = draw_face_annotations(
                        frame, primary_face, s_emotion, s_conf, s_probs
                    )
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.warning("⚠️ No face detected in frame. Please face the camera with good lighting.")
                    st.image(pil_img, use_container_width=True)

        else:
            uploaded_file = st.file_uploader("Upload a face image (JPG / PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                frame = pil_to_cv2(pil_img)
                primary_face = detector.get_primary_face(frame)

                if primary_face is not None:
                    detected_face_roi = detector.crop_face(frame, primary_face)
                    raw_emotion, raw_conf, raw_probs = classifier.predict_emotion(
                        detected_face_roi,
                        neutral_weight=neutral_suppression,
                        sensitivity=emotion_sens
                    )
                    
                    # For static image upload, reset smoother to this photo state
                    st.session_state.smoother.reset()
                    s_emotion, s_conf, s_probs, stability = st.session_state.smoother.update(
                        raw_emotion, raw_conf, raw_probs
                    )

                    st.session_state.current_emotion = raw_emotion
                    st.session_state.current_confidence = raw_conf
                    st.session_state.current_probs = raw_probs
                    st.session_state.stability_index = 1.0

                    if st.session_state.auto_switch and manual_override == "None (Use AI Vision)":
                        new_track = recommender.get_recommendation(raw_emotion, force_new=True)
                        if new_track:
                            recommender.set_current_track(new_track)
                            st.session_state.history_log.append({
                                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                                "emotion": raw_emotion,
                                "trigger": f"Photo Upload ({raw_conf*100:.1f}%)",
                                "track": new_track.title
                            })

                    annotated_frame = draw_face_annotations(frame, primary_face, raw_emotion, raw_conf, raw_probs)
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.warning("⚠️ No face detected in uploaded image.")
                    st.image(pil_img, use_container_width=True)

        # Emotion Probability Distribution Bars
        st.markdown("#### 📊 Real-Time Emotion Probabilities")
        curr_probs = st.session_state.current_probs
        
        for emotion in EMOTIONS:
            meta = EMOTION_METADATA[emotion]
            prob = curr_probs.get(emotion, 0.0)
            col_lbl, col_bar = st.columns([1, 3])
            with col_lbl:
                st.write(f"{meta['emoji']} **{emotion.capitalize()}**")
            with col_bar:
                st.progress(float(min(1.0, max(0.0, prob))), text=f"{prob * 100:.1f}%")

    # RIGHT COLUMN: Music Player & Recommendation Interface
    with col_player:
        st.markdown("### 🎵 Now Playing & Recommendation")

        current_emotion = st.session_state.current_emotion
        meta = EMOTION_METADATA.get(current_emotion, EMOTION_METADATA["neutral"])
        current_track = recommender.current_track

        # Glassmorphism Player Card
        st.markdown(f"""
        <div class="glass-card">
            <div class="emotion-badge" style="background-color: {meta['color']}22; color: {meta['color']}; border: 1px solid {meta['color']}66;">
                <span>{meta['emoji']}</span>
                <span>Current Mood: {current_emotion.upper()}</span>
            </div>
            <p class="song-title">{(current_track.title if current_track else 'Select a Track')}</p>
            <p class="song-artist">👤 {(current_track.artist if current_track else 'Procedural Sound Lab')} • 🏷️ {meta['genre']}</p>
            <div>
                <span class="metric-chip" style="background-color: #334155; color: #38bdf8;">⚡ Confidence: {st.session_state.current_confidence * 100:.1f}%</span>
                <span class="metric-chip" style="background-color: #334155; color: #a78bfa;">🎯 Stability: {st.session_state.stability_index * 100:.0f}%</span>
                <span class="metric-chip" style="background-color: #334155; color: #34d399;">⏱️ Duration: {(current_track.duration_seconds if current_track else 15.0):.0f}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Audio Playback Widget
        if current_track:
            audio_bytes = music_provider.get_track_audio_bytes(current_track)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav", start_time=0)
            else:
                st.info("Audio stream buffer ready.")

        # Interactive Controls
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("⏭️ Next Song", use_container_width=True):
                next_trk = recommender.next_track()
                if next_trk:
                    st.session_state.history_log.append({
                        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "emotion": current_emotion,
                        "trigger": "Manual Next",
                        "track": next_trk.title
                    })
                    st.rerun()

        with btn_col2:
            if st.button("🔀 Smart Shuffle", use_container_width=True):
                new_trk = recommender.get_recommendation(current_emotion, force_new=True)
                if new_trk:
                    recommender.set_current_track(new_trk)
                    st.session_state.history_log.append({
                        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "emotion": current_emotion,
                        "trigger": "Manual Shuffle",
                        "track": new_trk.title
                    })
                    st.rerun()

        with btn_col3:
            if st.button("🧹 Reset Smoother", use_container_width=True):
                st.session_state.smoother.reset()
                st.success("Smoother reset.")

        # Playlist for Current Emotion
        st.markdown(f"#### 📑 Playlist for {current_emotion.capitalize()} Mood")
        matching_tracks = music_provider.get_tracks_by_emotion(current_emotion)
        
        if matching_tracks:
            for idx, trk in enumerate(matching_tracks):
                is_active = current_track and trk.id == current_track.id
                prefix = "▶️ **[PLAYING]** " if is_active else f"{idx+1}. "
                c_info, c_act = st.columns([3, 1])
                with c_info:
                    st.write(f"{prefix}{trk.title} ({trk.genre})")
                with c_act:
                    if not is_active and st.button("Play", key=f"play_{trk.id}"):
                        recommender.set_current_track(trk)
                        st.session_state.history_log.append({
                            "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                            "emotion": current_emotion,
                            "trigger": "Playlist Select",
                            "track": trk.title
                        })
                        st.rerun()
        else:
            st.info("No tracks found for this category.")


# TAB 2: Analytics & Track History
with tab_analytics:
    st.markdown("### 📊 Session Analytics & Emotion Transition Log")
    if st.session_state.history_log:
        history_df = pd.DataFrame(st.session_state.history_log)
        st.dataframe(history_df, use_container_width=True)

        col_pie, col_stats = st.columns(2)
        with col_pie:
            st.markdown("#### Mood Frequency Distribution")
            emotion_counts = history_df["emotion"].value_counts()
            st.bar_chart(emotion_counts)
    else:
        st.info("No playback history recorded yet. Interact with the camera or change songs to view analytics.")


# TAB 3: Model & Architecture
with tab_architecture:
    st.markdown("### 🧠 Neural Network & Pipeline Architecture")
    
    st.markdown("""
    #### 1. Vision & Detection Pipeline
    - **Face Detection**: Multi-scale OpenCV Haar Cascade with histogram equalization for contrast invariance.
    - **Context Padding**: 15% margin expansion around bounding boxes to include forehead and jawline cues.
    - **Resolution**: Resized and normalized to $(224 \\times 224 \\times 3)$ RGB tensor.

    #### 2. MobileNetV2 Emotion Classification Architecture
    - **Backbone**: MobileNetV2 feature extractor pre-trained on ImageNet.
    - **Classification Head**:
      - `GlobalAveragePooling2D`
      - `Dense(256, activation='relu', l2_reg=1e-4)`
      - `BatchNormalization`
      - `Dropout(0.35)`
      - `Dense(128, activation='relu', l2_reg=1e-4)`
      - `Dropout(0.25)`
      - `Dense(5, activation='softmax')` for $[\\text{Happy}, \\text{Sad}, \\text{Angry}, \\text{Surprise}, \\text{Neutral}]$

    #### 3. Temporal Emotion Smoothing Algorithm
    - **Exponential Moving Average (EMA)**:
      $$\\bar{P}_t(e) = \\alpha \\bar{P}_{t-1}(e) + (1-\\alpha) P_t(e)$$
    - **Sliding Window Consensus**: Tracks historical voting buffer (default $N=10$) with stability scoring.
    - **Hysteresis Gate**: Eliminates transient micro-expressions and flickers before triggering audio playback switches.

    #### 4. Extensible Music Provider Architecture
    - Modular `BaseMusicProvider` abstraction ready for Spotify API (`spotipy`) and YouTube Audio Streamer integration.
    """)
