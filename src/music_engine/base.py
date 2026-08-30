"""
Abstract Music Provider Base Interface.
Enables pluggable music backends (Local Files, Spotify Web API, YouTube Audio Streamer, SoundCloud).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Track:
    """Standardized Music Track Representation."""
    id: str
    title: str
    artist: str
    emotion_tag: str
    duration_seconds: float
    file_path: Optional[Path] = None
    stream_url: Optional[str] = None
    cover_art_url: Optional[str] = None
    genre: Optional[str] = None
    tempo_bpm: Optional[int] = None
    energy_score: Optional[float] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "emotion_tag": self.emotion_tag,
            "duration_seconds": self.duration_seconds,
            "file_path": str(self.file_path) if self.file_path else None,
            "stream_url": self.stream_url,
            "genre": self.genre,
            "tempo_bpm": self.tempo_bpm,
            "energy_score": self.energy_score
        }


class BaseMusicProvider(ABC):
    """
    Abstract interface for music providers.
    Subclasses can implement Local Storage, Spotify API, YouTube API, SoundCloud, etc.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initializes client connections or reads filesystem directories."""
        pass

    @abstractmethod
    def get_tracks_by_emotion(self, emotion: str) -> List[Track]:
        """Retrieves a list of available tracks mapped to a specific emotion."""
        pass

    @abstractmethod
    def get_track_audio_bytes(self, track: Track) -> Optional[bytes]:
        """Retrieves the raw audio binary bytes for streaming or local playback."""
        pass

    @abstractmethod
    def get_all_tracks(self) -> List[Track]:
        """Returns all indexed tracks across all emotion categories."""
        pass
