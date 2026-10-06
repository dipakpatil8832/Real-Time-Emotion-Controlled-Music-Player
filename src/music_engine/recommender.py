"""
Music Recommender Engine.
Maps smoothed emotions to playlists, handles intelligent queue progression,
auto-pilot track switches on emotional transitions, and prevents repetition.
"""

import random
from typing import Dict, List, Optional

from src.config import EMOTION_METADATA, EMOTIONS, music_config
from src.music_engine.base import BaseMusicProvider, Track
from src.utils import logger


class MusicRecommender:
    """
    Intelligent recommendation controller connecting the emotion vision layer
    with the music audio player.
    """

    def __init__(self, provider: BaseMusicProvider):
        self.provider = provider
        self.current_emotion: str = music_config.default_emotion
        self.current_track: Optional[Track] = None
        self.auto_switch_enabled: bool = True
        self.recent_track_ids: List[str] = []
        self.max_recent_memory: int = 5
        self.play_history: List[Dict[str, str]] = []

    def initialize(self) -> bool:
        """Initializes the underlying provider and selects a default track."""
        success = self.provider.initialize()
        if success:
            self.current_track = self.get_recommendation(self.current_emotion, force_new=True)
        return success

    def get_recommendation(
        self,
        emotion: str,
        force_new: bool = False,
        language: Optional[str] = None
    ) -> Optional[Track]:
        """
        Recommends a suitable track for the specified emotion and language.
        Avoids recently played tracks if alternatives exist.
        """
        emotion_key = emotion.lower()
        if emotion_key not in EMOTIONS:
            emotion_key = "neutral"

        if hasattr(self.provider, "get_tracks_by_emotion"):
            try:
                available_tracks = self.provider.get_tracks_by_emotion(emotion_key, language=language)
            except TypeError:
                available_tracks = self.provider.get_tracks_by_emotion(emotion_key)
        else:
            available_tracks = self.provider.get_all_tracks()

        if not available_tracks:
            available_tracks = self.provider.get_all_tracks()
            if not available_tracks:
                return None

        # Filter out recently played tracks if possible
        unplayed = [t for t in available_tracks if t.id not in self.recent_track_ids]
        candidate_pool = unplayed if unplayed else available_tracks

        # If we have multiple candidates and don't want the same track as current
        if force_new and len(candidate_pool) > 1 and self.current_track:
            candidate_pool = [t for t in candidate_pool if t.id != self.current_track.id]
            if not candidate_pool:
                candidate_pool = available_tracks

        selected = random.choice(candidate_pool)
        return selected

    def on_emotion_update(
        self,
        smoothed_emotion: str,
        confidence: float,
        stability: float
    ) -> Optional[Track]:
        """
        Called when a stabilized emotion reading is received from the smoother.
        If auto-switch is enabled and a significant emotion shift occurred, switches track.
        """
        if not self.auto_switch_enabled:
            return None

        # Only trigger track change if emotion genuinely changed and is stable
        if smoothed_emotion != self.current_emotion and stability >= 0.60:
            logger.info(
                "Triggering emotion-based track switch: %s -> %s (Confidence: %.2f)",
                self.current_emotion, smoothed_emotion, confidence
            )
            self.current_emotion = smoothed_emotion
            new_track = self.get_recommendation(smoothed_emotion, force_new=True)
            if new_track:
                self.set_current_track(new_track)
                return new_track

        return None

    def set_current_track(self, track: Track) -> None:
        """Sets the active track and updates history logs."""
        self.current_track = track
        self.recent_track_ids.append(track.id)
        if len(self.recent_track_ids) > self.max_recent_memory:
            self.recent_track_ids.pop(0)

        meta = EMOTION_METADATA.get(track.emotion_tag, {})
        self.play_history.append({
            "title": track.title,
            "artist": track.artist,
            "emotion": track.emotion_tag,
            "genre": track.genre or meta.get("genre", "Soundtrack"),
            "emoji": meta.get("emoji", "🎵")
        })
        if len(self.play_history) > 30:
            self.play_history.pop(0)

    def next_track(self) -> Optional[Track]:
        """Manually or automatically skips to the next track for the current emotion."""
        new_track = self.get_recommendation(self.current_emotion, force_new=True)
        if new_track:
            self.set_current_track(new_track)
        return new_track
