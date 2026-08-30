"""
Music Engine Package for Emotion-Based Music Retrieval and Playback.
"""

from src.music_engine.base import BaseMusicProvider, Track
from src.music_engine.local_provider import LocalMusicProvider
from src.music_engine.recommender import MusicRecommender
from src.music_engine.synthesizer import ProceduralAudioSynthesizer

__all__ = [
    "BaseMusicProvider",
    "Track",
    "LocalMusicProvider",
    "MusicRecommender",
    "ProceduralAudioSynthesizer",
]
