"""
YouTube Music Provider.
Fetches real-time YouTube music tracks based on emotion classification,
supports Google YouTube Data API v3, and provides instant fallback with curated high-fidelity streams.
"""

import json
import os
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from src.config import EMOTION_METADATA, EMOTIONS
from src.music_engine.base import BaseMusicProvider, Track
from src.utils import logger

# Emotion-specific search keywords optimized for music selection
EMOTION_SEARCH_QUERIES: Dict[str, str] = {
    "happy": "happy upbeat pop songs feel good music",
    "sad": "sad emotional songs acoustic piano heartbreak melancholy",
    "angry": "intense workout heavy bass energetic rock phonk music",
    "surprise": "upbeat synthwave electronic dynamic dance grooves",
    "neutral": "chillhop lofi beats study relax focus ambient music"
}

# Curated High-Quality YouTube Tracks (Zero-Config fallback)
CURATED_YOUTUBE_TRACKS: Dict[str, List[Dict[str, str]]] = {
    "happy": [
        {
            "id": "ZbZSe6N_BXs",
            "title": "Pharrell Williams - Happy",
            "artist": "Pharrell Williams",
            "genre": "Pop / Feel Good",
            "thumbnail": "https://img.youtube.com/vi/ZbZSe6N_BXs/hqdefault.jpg",
            "duration": 233.0
        },
        {
            "id": "ru0K8uYEZWw",
            "title": "Justin Timberlake - CAN'T STOP THE FEELING!",
            "artist": "Justin Timberlake",
            "genre": "Dance-Pop",
            "thumbnail": "https://img.youtube.com/vi/ru0K8uYEZWw/hqdefault.jpg",
            "duration": 285.0
        },
        {
            "id": "k85mRPqvMbE",
            "title": "American Authors - Best Day Of My Life",
            "artist": "American Authors",
            "genre": "Indie Pop / Upbeat",
            "thumbnail": "https://img.youtube.com/vi/k85mRPqvMbE/hqdefault.jpg",
            "duration": 217.0
        },
        {
            "id": "OPf0YbXqDm0",
            "title": "Mark Ronson - Uptown Funk ft. Bruno Mars",
            "artist": "Mark Ronson ft. Bruno Mars",
            "genre": "Funk / Pop",
            "thumbnail": "https://img.youtube.com/vi/OPf0YbXqDm0/hqdefault.jpg",
            "duration": 270.0
        }
    ],
    "sad": [
        {
            "id": "hLQl3WQQoQ0",
            "title": "Adele - Someone Like You",
            "artist": "Adele",
            "genre": "Soul / Piano Ballad",
            "thumbnail": "https://img.youtube.com/vi/hLQl3WQQoQ0/hqdefault.jpg",
            "duration": 285.0
        },
        {
            "id": "450p7goxZqg",
            "title": "John Legend - All of Me",
            "artist": "John Legend",
            "genre": "Acoustic Piano / Melancholic",
            "thumbnail": "https://img.youtube.com/vi/450p7goxZqg/hqdefault.jpg",
            "duration": 307.0
        },
        {
            "id": "rtOvBOTyX00",
            "title": "Christina Perri - A Thousand Years",
            "artist": "Christina Perri",
            "genre": "Acoustic / Soft Melancholy",
            "thumbnail": "https://img.youtube.com/vi/rtOvBOTyX00/hqdefault.jpg",
            "duration": 295.0
        },
        {
            "id": "rgXOX89g0pM",
            "title": "Dean Lewis - Be Alright",
            "artist": "Dean Lewis",
            "genre": "Indie Acoustic",
            "thumbnail": "https://img.youtube.com/vi/rgXOX89g0pM/hqdefault.jpg",
            "duration": 204.0
        }
    ],
    "angry": [
        {
            "id": "kXYiU_JCYtU",
            "title": "Linkin Park - Numb",
            "artist": "Linkin Park",
            "genre": "Alternative Rock / Cathartic",
            "thumbnail": "https://img.youtube.com/vi/kXYiU_JCYtU/hqdefault.jpg",
            "duration": 187.0
        },
        {
            "id": "1V_xRb0x9aw",
            "title": "Imagine Dragons - Believer",
            "artist": "Imagine Dragons",
            "genre": "Heavy Beat / Rock Energy",
            "thumbnail": "https://img.youtube.com/vi/1V_xRb0x9aw/hqdefault.jpg",
            "duration": 216.0
        },
        {
            "id": "fJ9rUzIMcZQ",
            "title": "Queen - Bohemian Rhapsody",
            "artist": "Queen",
            "genre": "Progressive Rock",
            "thumbnail": "https://img.youtube.com/vi/fJ9rUzIMcZQ/hqdefault.jpg",
            "duration": 359.0
        }
    ],
    "surprise": [
        {
            "id": "4NRXx6U8ABQ",
            "title": "The Weeknd - Blinding Lights",
            "artist": "The Weeknd",
            "genre": "Synthwave / Electric",
            "thumbnail": "https://img.youtube.com/vi/4NRXx6U8ABQ/hqdefault.jpg",
            "duration": 260.0
        },
        {
            "id": "09R8_2nJtjg",
            "title": "Maroon 5 - Sugar",
            "artist": "Maroon 5",
            "genre": "Pop Funk / Upbeat",
            "thumbnail": "https://img.youtube.com/vi/09R8_2nJtjg/hqdefault.jpg",
            "duration": 300.0
        },
        {
            "id": "fRh_vgS2dFE",
            "title": "Justin Bieber - Sorry",
            "artist": "Justin Bieber",
            "genre": "Dancehall Pop",
            "thumbnail": "https://img.youtube.com/vi/fRh_vgS2dFE/hqdefault.jpg",
            "duration": 205.0
        }
    ],
    "neutral": [
        {
            "id": "jfKfPfyJRdk",
            "title": "Lofi Hip Hop Radio - Beats to Relax/Study to",
            "artist": "Lofi Girl",
            "genre": "Lofi / Chillhop / Study Beats",
            "thumbnail": "https://img.youtube.com/vi/jfKfPfyJRdk/hqdefault.jpg",
            "duration": 360.0
        },
        {
            "id": "5qap5aO4i9A",
            "title": "Lofi Hip Hop - Chill Study Beats",
            "artist": "Lofi Chill Records",
            "genre": "Ambient / Chillhop",
            "thumbnail": "https://img.youtube.com/vi/5qap5aO4i9A/hqdefault.jpg",
            "duration": 300.0
        },
        {
            "id": "DWcJFNfaw9c",
            "title": "Relaxing Piano Music & Water Sounds",
            "artist": "Soothing Relaxation",
            "genre": "Ambient Piano / Zen",
            "thumbnail": "https://img.youtube.com/vi/DWcJFNfaw9c/hqdefault.jpg",
            "duration": 360.0
        }
    ]
}


class YouTubeMusicProvider(BaseMusicProvider):
    """
    Dynamic YouTube Music Provider with auto-search and API support.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self.track_cache: Dict[str, List[Track]] = {e: [] for e in EMOTIONS}
        self.is_initialized: bool = False

    def initialize(self) -> bool:
        """Populates baseline curated tracks for instant responsiveness."""
        self.track_cache = {e: [] for e in EMOTIONS}
        for emotion, track_list in CURATED_YOUTUBE_TRACKS.items():
            for item in track_list:
                vid = item["id"]
                track = Track(
                    id=vid,
                    title=item["title"],
                    artist=item["artist"],
                    emotion_tag=emotion,
                    duration_seconds=float(item.get("duration", 200.0)),
                    stream_url=f"https://www.youtube.com/watch?v={vid}",
                    cover_art_url=item.get("thumbnail", f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"),
                    genre=item.get("genre", EMOTION_METADATA.get(emotion, {}).get("genre", "Soundtrack")),
                    extra_metadata={"embed_url": f"https://www.youtube.com/embed/{vid}?autoplay=1&enablejsapi=1"}
                )
                self.track_cache[emotion].append(track)

        self.is_initialized = True
        logger.info("YouTubeMusicProvider initialized with %d curated emotion tracks.", sum(len(v) for v in self.track_cache.values()))
        return True

    def search_youtube_api(self, query: str, emotion: str, max_results: int = 5) -> List[Track]:
        """
        Queries Google YouTube Data API v3 for live search results.
        """
        if not self.api_key:
            return []

        try:
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "videoCategoryId": "10",  # Music category
                "maxResults": max_results,
                "key": self.api_key
            }
            url = f"https://www.googleapis.com/youtube/v3/search?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "EmotiBeat/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    new_tracks: List[Track] = []
                    for item in data.get("items", []):
                        vid = item.get("id", {}).get("videoId")
                        snippet = item.get("snippet", {})
                        if vid:
                            title = snippet.get("title", f"YouTube {emotion.capitalize()} Music")
                            artist = snippet.get("channelTitle", "YouTube Music")
                            thumb = snippet.get("thumbnails", {}).get("high", {}).get("url", f"https://img.youtube.com/vi/{vid}/hqdefault.jpg")
                            
                            t = Track(
                                id=vid,
                                title=title,
                                artist=artist,
                                emotion_tag=emotion,
                                duration_seconds=240.0,
                                stream_url=f"https://www.youtube.com/watch?v={vid}",
                                cover_art_url=thumb,
                                genre=f"YouTube {emotion.capitalize()}",
                                extra_metadata={"embed_url": f"https://www.youtube.com/embed/{vid}?autoplay=1&enablejsapi=1"}
                            )
                            new_tracks.append(t)
                    if new_tracks:
                        logger.info("Fetched %d live YouTube tracks for query '%s'", len(new_tracks), query)
                        return new_tracks
        except Exception as e:
            logger.warning("YouTube API live fetch failed (%s), using curated library.", e)

        return []

    def get_tracks_by_emotion(self, emotion: str) -> List[Track]:
        """Returns YouTube tracks matching the specified emotion."""
        if not self.is_initialized:
            self.initialize()

        emotion_key = emotion.lower()
        if emotion_key not in EMOTIONS:
            emotion_key = "neutral"

        # If user has an API key and hasn't queried live API yet for this emotion
        if self.api_key and len(self.track_cache.get(emotion_key, [])) <= len(CURATED_YOUTUBE_TRACKS.get(emotion_key, [])):
            query = EMOTION_SEARCH_QUERIES.get(emotion_key, f"{emotion_key} music")
            live_tracks = self.search_youtube_api(query, emotion_key)
            if live_tracks:
                existing_ids = {t.id for t in self.track_cache[emotion_key]}
                for lt in live_tracks:
                    if lt.id not in existing_ids:
                        self.track_cache[emotion_key].insert(0, lt)

        return self.track_cache.get(emotion_key, [])

    def get_track_audio_bytes(self, track: Track) -> Optional[bytes]:
        """YouTube playback uses embedded video iframe in Streamlit."""
        return None

    def get_all_tracks(self) -> List[Track]:
        """Returns all cached YouTube tracks across emotions."""
        if not self.is_initialized:
            self.initialize()
        all_t = []
        for t_list in self.track_cache.values():
            all_t.extend(t_list)
        return all_t
