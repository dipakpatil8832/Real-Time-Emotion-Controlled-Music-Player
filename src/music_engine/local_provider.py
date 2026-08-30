"""
Local Filesystem Music Provider.
Discovers, indexes, and streams local audio tracks (.mp3, .wav, .ogg, .flac)
organized by emotion subdirectories.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os

from src.config import EMOTION_METADATA, EMOTIONS, MUSIC_DIR, music_config
from src.music_engine.base import BaseMusicProvider, Track
from src.music_engine.synthesizer import ProceduralAudioSynthesizer
from src.utils import logger


class LocalMusicProvider(BaseMusicProvider):
    """
    Local filesystem implementation of BaseMusicProvider.
    Directory structure:
    music_library/
       ├── happy/
       ├── sad/
       ├── angry/
       ├── surprise/
       └── neutral/
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir or music_config.music_library_path)
        self.indexed_tracks: Dict[str, List[Track]] = {e: [] for e in EMOTIONS}
        self.synthesizer = ProceduralAudioSynthesizer()

    def initialize(self) -> bool:
        """Indexes all emotion folders and ensures sample tracks are available."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        total_found = self._scan_library()

        # If empty or missing files for any emotion, auto-synthesize default library
        if total_found == 0 or any(len(self.indexed_tracks[e]) == 0 for e in EMOTIONS):
            logger.info("Local music library is sparse/empty. Synthesizing starter music library...")
            self.synthesizer.generate_full_sample_library(self.root_dir)
            self._scan_library()

        logger.info(
            "LocalMusicProvider initialized with %d tracks across %d categories.",
            len(self.get_all_tracks()), len(EMOTIONS)
        )
        return True

    def _scan_library(self) -> int:
        """Scans filesystem directories and populates indexed_tracks dictionary."""
        self.indexed_tracks = {e: [] for e in EMOTIONS}
        total_count = 0

        for emotion in EMOTIONS:
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                emotion_dir.mkdir(parents=True, exist_ok=True)
                continue

            for file in emotion_dir.iterdir():
                if file.is_file() and file.suffix.lower() in music_config.supported_audio_formats:
                    track_id = f"local_{emotion}_{file.stem}"
                    clean_title = file.stem.replace("_", " ").replace("-", " ").title()
                    meta = EMOTION_METADATA.get(emotion, {})
                    genre = meta.get("genre", "Soundtrack")

                    track = Track(
                        id=track_id,
                        title=clean_title,
                        artist="Local Artist / Procedural Sound Lab",
                        emotion_tag=emotion,
                        duration_seconds=15.0,
                        file_path=file,
                        genre=genre,
                        tempo_bpm=120 if emotion in ["happy", "surprise", "angry"] else 80,
                        energy_score=0.85 if emotion in ["happy", "angry"] else 0.4
                    )
                    self.indexed_tracks[emotion].append(track)
                    total_count += 1

        return total_count

    def get_tracks_by_emotion(self, emotion: str) -> List[Track]:
        """Returns all indexed tracks for a specific emotion."""
        emotion_key = emotion.lower()
        return self.indexed_tracks.get(emotion_key, [])

    def get_track_audio_bytes(self, track: Track) -> Optional[bytes]:
        """Reads audio file from disk into memory bytes."""
        if track.file_path and track.file_path.exists():
            try:
                with open(track.file_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error("Failed to read audio file %s: %s", track.file_path, e)
                return None
        return None

    def get_all_tracks(self) -> List[Track]:
        """Returns flat list of all tracks."""
        all_tracks = []
        for track_list in self.indexed_tracks.values():
            all_tracks.extend(track_list)
        return all_tracks
