"""
Multilingual YouTube Music Provider.
Fetches real-time YouTube music tracks based on emotion classification and language preferences.
Supports Hindi, English, Punjabi, Lo-Fi / Instrumental, and Spanish music catalogs.
"""

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from src.config import EMOTION_METADATA, EMOTIONS
from src.music_engine.base import BaseMusicProvider, Track
from src.utils import logger

# Supported Music Languages & Types
MUSIC_LANGUAGES = ["All", "Hindi", "English", "Punjabi", "Lo-Fi / Instrumental", "Spanish"]

# Emotion-specific and Language-specific search queries
EMOTION_SEARCH_QUERIES: Dict[str, Dict[str, str]] = {
    "happy": {
        "All": "happy upbeat pop songs feel good music",
        "Hindi": "happy upbeat bollywood hindi dance party songs",
        "English": "happy upbeat pop songs feel good music",
        "Punjabi": "happy punjabi bhangra party songs upbeat",
        "Lo-Fi / Instrumental": "happy upbeat chill lofi beats instrumental",
        "Spanish": "musica feliz alegre reggaeton pop latino"
    },
    "sad": {
        "All": "sad emotional songs acoustic piano heartbreak melancholy",
        "Hindi": "sad emotional hindi songs arijit singh breakup acoustic",
        "English": "sad emotional songs acoustic piano heartbreak melancholy",
        "Punjabi": "sad punjabi romantic heartbreak songs b praak",
        "Lo-Fi / Instrumental": "sad lofi hip hop beats crying melancholy sleep",
        "Spanish": "canciones tristes baladas en espanol desamor"
    },
    "angry": {
        "All": "intense workout heavy bass energetic rock phonk music",
        "Hindi": "angry high energy hindi rap gully boy rock motivation",
        "English": "intense workout heavy bass energetic rock phonk music",
        "Punjabi": "aggressive punjabi rap sidhu moosewala drill phonk",
        "Lo-Fi / Instrumental": "aggressive phonk drift heavy bass workout",
        "Spanish": "rock espanol rap agresivo motivacion gym"
    },
    "surprise": {
        "All": "upbeat synthwave electronic dynamic dance grooves",
        "Hindi": "dynamic hindi dance party war ghungroo grooving songs",
        "English": "upbeat synthwave electronic dynamic dance grooves",
        "Punjabi": "dynamic modern punjabi pop party dance tracks",
        "Lo-Fi / Instrumental": "synthwave electronic future bass dynamic beats",
        "Spanish": "electro latino dance pop energetico"
    },
    "neutral": {
        "All": "chillhop lofi beats study relax focus ambient music",
        "Hindi": "soothing hindi sufi calm acoustic songs focus",
        "English": "chillhop lofi beats study relax focus ambient music",
        "Punjabi": "soothing calm punjabi acoustic melodies slow",
        "Lo-Fi / Instrumental": "ambient meditation lofi study focus background music",
        "Spanish": "musica relajante espanol guitarra acustica suave"
    }
}

# Curated High-Quality Multilingual YouTube Tracks (Zero-Config fallback)
CURATED_YOUTUBE_TRACKS: Dict[str, List[Dict[str, Any]]] = {
    "happy": [
        # Hindi
        {
            "id": "ZbZSe6N_BXs", "title": "Pharrell Williams - Happy", "artist": "Pharrell Williams",
            "language": "English", "genre": "Pop / Feel Good", "duration": 233.0
        },
        {
            "id": "ru0K8uYEZWw", "title": "Justin Timberlake - CAN'T STOP THE FEELING!", "artist": "Justin Timberlake",
            "language": "English", "genre": "Dance-Pop", "duration": 285.0
        },
        {
            "id": "OPf0YbXqDm0", "title": "Mark Ronson - Uptown Funk ft. Bruno Mars", "artist": "Mark Ronson",
            "language": "English", "genre": "Funk / Pop", "duration": 270.0
        },
        {
            "id": "NTHz9e-Nq7k", "title": "Kar Gayi Chull - Kapoor & Sons", "artist": "Badshah, Neha Kakkar",
            "language": "Hindi", "genre": "Bollywood Party / Upbeat", "duration": 187.0
        },
        {
            "id": "jCEdTq3j-0U", "title": "Gallan Goodiyaan - Dil Dhadakne Do", "artist": "Yashita Sharma, Shankar Mahadevan",
            "language": "Hindi", "genre": "Bollywood Celebration", "duration": 296.0
        },
        {
            "id": "II2EO3NwUr8", "title": "Badtameez Dil - Yeh Jawaani Hai Deewani", "artist": "Benny Dayal, Pritam",
            "language": "Hindi", "genre": "Bollywood Dance", "duration": 252.0
        },
        {
            "id": "cl0a3i2wFcc", "title": "Brown Munde - AP Dhillon & Gurinder Gill", "artist": "AP Dhillon",
            "language": "Punjabi", "genre": "Punjabi Pop / Swagger", "duration": 266.0
        },
        {
            "id": "V1bFr2KGq1g", "title": "Born to Shine - Diljit Dosanjh", "artist": "Diljit Dosanjh",
            "language": "Punjabi", "genre": "Punjabi Upbeat", "duration": 214.0
        },
        {
            "id": "kJQP7kiw5Fk", "title": "Luis Fonsi - Despacito ft. Daddy Yankee", "artist": "Luis Fonsi",
            "language": "Spanish", "genre": "Latin Pop / Dance", "duration": 282.0
        },
        {
            "id": "7NOSDKb0HlU", "title": "Sunny Day Chill - Upbeat Lo-Fi", "artist": "Lofi Fruits Music",
            "language": "Lo-Fi / Instrumental", "genre": "Chillhop / Feel Good", "duration": 190.0
        }
    ],
    "sad": [
        {
            "id": "hLQl3WQQoQ0", "title": "Adele - Someone Like You", "artist": "Adele",
            "language": "English", "genre": "Soul / Piano Ballad", "duration": 285.0
        },
        {
            "id": "450p7goxZqg", "title": "John Legend - All of Me", "artist": "John Legend",
            "language": "English", "genre": "Acoustic Piano", "duration": 307.0
        },
        {
            "id": "rgXOX89g0pM", "title": "Dean Lewis - Be Alright", "artist": "Dean Lewis",
            "language": "English", "genre": "Indie Acoustic", "duration": 204.0
        },
        {
            "id": "284Ov7ysmfA", "title": "Channa Mereya - Ae Dil Hai Mushkil", "artist": "Arijit Singh, Pritam",
            "language": "Hindi", "genre": "Sufi Melancholy / Acoustic", "duration": 289.0
        },
        {
            "id": "sAzlWScHTc4", "title": "Agar Tum Saath Ho - Tamasha", "artist": "Arijit Singh, Alka Yagnik",
            "language": "Hindi", "genre": "Emotional Ballad", "duration": 341.0
        },
        {
            "id": "Umqb9KENgmk", "title": "Tum Hi Ho - Aashiqui 2", "artist": "Arijit Singh, Mithoon",
            "language": "Hindi", "genre": "Romantic Melancholy", "duration": 262.0
        },
        {
            "id": "O_np3m6Jv2E", "title": "Qismat - Ammy Virk & Sargun Mehta", "artist": "Ammy Virk, B Praak",
            "language": "Punjabi", "genre": "Heartbreak Ballad", "duration": 250.0
        },
        {
            "id": "VAt0j_nfxlE", "title": "Filhall - B Praak & Akshay Kumar", "artist": "B Praak, Jaani",
            "language": "Punjabi", "genre": "Emotional Punjabi", "duration": 331.0
        },
        {
            "id": "kXYiU_JCYtU", "title": "The Girl I Haven't Met - Sad Lo-Fi Beats", "artist": "Kudasai",
            "language": "Lo-Fi / Instrumental", "genre": "Lo-Fi / Melancholy", "duration": 210.0
        }
    ],
    "angry": [
        {
            "id": "kXYiU_JCYtU", "title": "Linkin Park - Numb", "artist": "Linkin Park",
            "language": "English", "genre": "Alternative Rock / Cathartic", "duration": 187.0
        },
        {
            "id": "1V_xRb0x9aw", "title": "Imagine Dragons - Believer", "artist": "Imagine Dragons",
            "language": "English", "genre": "Heavy Beat / Rock Energy", "duration": 216.0
        },
        {
            "id": "_Yhyp-_hX2s", "title": "Eminem - 'Till I Collapse", "artist": "Eminem ft. Nate Dogg",
            "language": "English", "genre": "Aggressive Hip-Hop", "duration": 298.0
        },
        {
            "id": "j0q4_k24kIQ", "title": "Apna Time Aayega - Gully Boy", "artist": "Ranveer Singh, DIVINE",
            "language": "Hindi", "genre": "Desi Hip-Hop / Cathartic", "duration": 150.0
        },
        {
            "id": "vM_o8XFpW_o", "title": "Zinda - Bhaag Milkha Bhaag", "artist": "Siddharth Mahadevan, Shankar-Ehsaan-Loy",
            "language": "Hindi", "genre": "High Octane Rock", "duration": 211.0
        },
        {
            "id": "n_FCrCQ6-9U", "title": "295 - Sidhu Moose Wala", "artist": "Sidhu Moose Wala",
            "language": "Punjabi", "genre": "Aggressive Drill / Hip-Hop", "duration": 270.0
        },
        {
            "id": "cl0a3i2wFcc", "title": "G.O.A.T. - Diljit Dosanjh", "artist": "Diljit Dosanjh",
            "language": "Punjabi", "genre": "Hard Punjabi Trap", "duration": 223.0
        },
        {
            "id": "N3oCS85HvpY", "title": "Murder In My Mind - Drift Phonk", "artist": "Kordhell",
            "language": "Lo-Fi / Instrumental", "genre": "Phonk / Heavy Bass", "duration": 145.0
        }
    ],
    "surprise": [
        {
            "id": "4NRXx6U8ABQ", "title": "The Weeknd - Blinding Lights", "artist": "The Weeknd",
            "language": "English", "genre": "Synthwave / Electric", "duration": 260.0
        },
        {
            "id": "TUVcZfQe-Kw", "title": "Dua Lipa - Levitating", "artist": "Dua Lipa",
            "language": "English", "genre": "Nu-Disco / Dynamic", "duration": 203.0
        },
        {
            "id": "qFkNATtc3mc", "title": "Ghungroo - War", "artist": "Arijit Singh, Shilpa Rao, Vishal-Shekhar",
            "language": "Hindi", "genre": "Dynamic Funk / Pop", "duration": 302.0
        },
        {
            "id": "k4yXQkG2s1E", "title": "Kala Chashma - Baar Baar Dekho", "artist": "Amar Arshi, Badshah, Neha Kakkar",
            "language": "Hindi", "genre": "High Dynamic Energy", "duration": 184.0
        },
        {
            "id": "vX2c7k07EwU", "title": "Excuses - Intense", "artist": "AP Dhillon, Gurinder Gill",
            "language": "Punjabi", "genre": "Modern Grooves", "duration": 176.0
        },
        {
            "id": "8ZhnJmXmRvw", "title": "Resonance - Synthwave Retro", "artist": "HOME",
            "language": "Lo-Fi / Instrumental", "genre": "Synthwave / Uplifting", "duration": 212.0
        }
    ],
    "neutral": [
        {
            "id": "5yx6BWlEVcY", "title": "Chillhop Essentials - Relaxing Lofi Beats", "artist": "Chillhop Music",
            "language": "Lo-Fi / Instrumental", "genre": "Chillhop / Ambient", "duration": 3600.0
        },
        {
            "id": "DWcJFNfaw9c", "title": "Weightless - Marconi Union (Ambient Chill)", "artist": "Marconi Union",
            "language": "Lo-Fi / Instrumental", "genre": "Ambient / Zen Frequency", "duration": 360.0
        },
        {
            "id": "b_KjhpYq5-c", "title": "Iktara - Wake Up Sid", "artist": "Kavita Seth, Amit Trivedi",
            "language": "Hindi", "genre": "Sufi Chill / Acoustic", "duration": 253.0
        },
        {
            "id": "0B1v_H2lK2o", "title": "Kun Faya Kun - Rockstar", "artist": "A.R. Rahman, Mohit Chauhan, Javed Ali",
            "language": "Hindi", "genre": "Spiritual Ambient / Calm", "duration": 473.0
        },
        {
            "id": "H7HPA2xHw8E", "title": "Lover - Diljit Dosanjh (Chill Acoustic)", "artist": "Diljit Dosanjh",
            "language": "Punjabi", "genre": "Smooth Acoustic Punjabi", "duration": 200.0
        },
        {
            "id": "tPkqgD2c_1Y", "title": "Don't Know Why - Norah Jones", "artist": "Norah Jones",
            "language": "English", "genre": "Smooth Jazz / Acoustic", "duration": 186.0
        }
    ]
}


class YouTubeMusicProvider(BaseMusicProvider):
    """
    Multilingual YouTube Music Provider with auto-search and language filtering.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self.track_cache: Dict[str, List[Track]] = {e: [] for e in EMOTIONS}
        self.is_initialized: bool = False
        self.selected_language: str = "All"

    def initialize(self) -> bool:
        """Populates baseline curated multilingual tracks."""
        self.track_cache = {e: [] for e in EMOTIONS}
        for emotion, track_list in CURATED_YOUTUBE_TRACKS.items():
            for item in track_list:
                vid = item["id"]
                track_lang = item.get("language", "English")
                track = Track(
                    id=vid,
                    title=item["title"],
                    artist=item["artist"],
                    emotion_tag=emotion,
                    duration_seconds=float(item.get("duration", 200.0)),
                    stream_url=f"https://www.youtube.com/watch?v={vid}",
                    cover_art_url=f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
                    genre=f"{track_lang} • {item.get('genre', 'Soundtrack')}",
                    extra_metadata={
                        "embed_url": f"https://www.youtube.com/embed/{vid}?autoplay=1&enablejsapi=1",
                        "language": track_lang
                    }
                )
                self.track_cache[emotion].append(track)

        self.is_initialized = True
        logger.info("YouTubeMusicProvider initialized with %d curated multilingual tracks.", sum(len(v) for v in self.track_cache.values()))
        return True

    def set_language_filter(self, language: str):
        """Sets active language filter: 'All', 'Hindi', 'English', 'Punjabi', 'Lo-Fi / Instrumental', 'Spanish'."""
        self.selected_language = language if language in MUSIC_LANGUAGES else "All"

    def search_youtube_api(self, query: str, emotion: str, language: str = "All", max_results: int = 5) -> List[Track]:
        """Queries Google YouTube Data API v3 for live multilingual search results."""
        if not self.api_key:
            return []

        try:
            full_query = f"{query} {language}" if language != "All" else query
            params = {
                "part": "snippet",
                "q": full_query,
                "type": "video",
                "videoCategoryId": "10",
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
                            title = snippet.get("title", f"{emotion.capitalize()} Music")
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
                                genre=f"{language} • {emotion.capitalize()}",
                                extra_metadata={
                                    "embed_url": f"https://www.youtube.com/embed/{vid}?autoplay=1&enablejsapi=1",
                                    "language": language
                                }
                            )
                            new_tracks.append(t)
                    if new_tracks:
                        logger.info("Fetched %d live YouTube tracks for '%s'", len(new_tracks), full_query)
                        return new_tracks
        except Exception as e:
            logger.warning("YouTube API live fetch failed (%s), using curated library.", e)

        return []

    def get_tracks_by_emotion(self, emotion: str, language: Optional[str] = None) -> List[Track]:
        """Returns YouTube tracks matching emotion and optional language filter."""
        if not self.is_initialized:
            self.initialize()

        emotion_key = emotion.lower()
        if emotion_key not in EMOTIONS:
            emotion_key = "neutral"

        active_lang = language or self.selected_language
        cached_tracks = self.track_cache.get(emotion_key, [])

        if active_lang and active_lang != "All":
            filtered = [t for t in cached_tracks if t.extra_metadata.get("language") == active_lang]
            if filtered:
                return filtered
            # If no cached tracks match language, query live API if key is present
            if self.api_key:
                queries = EMOTION_SEARCH_QUERIES.get(emotion_key, {})
                q_text = queries.get(active_lang, f"{active_lang} {emotion_key} songs")
                live_tracks = self.search_youtube_api(q_text, emotion_key, language=active_lang)
                if live_tracks:
                    return live_tracks

        return cached_tracks

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
