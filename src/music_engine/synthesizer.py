"""
Procedural Harmonic Audio Synthesizer.
Generates pleasant, royalty-free WAV musical tracks for each of the 5 emotions
so the application has complete out-of-the-box local playback without external audio files.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import wavfile

from src.config import EMOTIONS, MUSIC_DIR
from src.utils import logger


class ProceduralAudioSynthesizer:
    """
    Synthesizes rich melodic and ambient audio tracks matching each emotion's
    harmonic mood, tempo, and acoustic texture.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def _generate_tone(self, freq: float, duration: float, envelope: str = "smooth") -> np.ndarray:
        """Generates a tone with harmonic richness and an amplitude envelope."""
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        # Fundamental + harmonics
        signal = 0.6 * np.sin(2 * np.pi * freq * t)
        signal += 0.25 * np.sin(2 * np.pi * (freq * 2) * t)
        signal += 0.15 * np.sin(2 * np.pi * (freq * 3) * t)

        # Envelopes
        n_samples = len(t)
        if envelope == "pluck":
            env = np.exp(-3.5 * t / duration)
        elif envelope == "pad":
            attack = int(n_samples * 0.2)
            release = int(n_samples * 0.2)
            env = np.ones(n_samples)
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)
        else:  # smooth
            attack = int(n_samples * 0.05)
            release = int(n_samples * 0.1)
            env = np.ones(n_samples)
            if attack > 0:
                env[:attack] = np.linspace(0, 1, attack)
            if release > 0:
                env[-release:] = np.linspace(1, 0, release)

        return signal * env

    def _chord(self, freqs: List[float], duration: float, envelope: str = "pad") -> np.ndarray:
        """Layers multiple frequencies into a harmonic chord."""
        chord_signal = np.zeros(int(self.sr * duration))
        for f in freqs:
            chord_signal += self._generate_tone(f, duration, envelope=envelope)
        return chord_signal / len(freqs)

    def synthesize_happy_track(self, duration: float = 12.0) -> np.ndarray:
        """Major key (C - G - Am - F) upbeat progression with joyful arpeggios."""
        chord_prog = [
            [261.63, 329.63, 392.00],  # C Major
            [196.00, 246.94, 293.66],  # G Major
            [220.00, 261.63, 329.63],  # A Minor
            [174.61, 220.00, 261.63],  # F Major
        ]
        beats = []
        step_dur = 0.35
        for chord in chord_prog:
            for _ in range(4):  # 4 pulses per chord
                # Arpeggiate
                for freq in chord:
                    beats.append(self._generate_tone(freq * 1.5, step_dur * 0.5, envelope="pluck"))
                beats.append(self._chord(chord, step_dur, envelope="smooth"))
        
        full_track = np.concatenate(beats)
        # Repeat to reach duration
        repeats = int(np.ceil(duration / (len(full_track) / self.sr)))
        full_track = np.tile(full_track, repeats)[:int(self.sr * duration)]
        return full_track

    def synthesize_sad_track(self, duration: float = 14.0) -> np.ndarray:
        """Melancholic minor progression (Am - F - C - Em) with slow soft resonance."""
        chord_prog = [
            [220.00, 261.63, 329.63],  # Am
            [174.61, 220.00, 261.63],  # F
            [261.63, 329.63, 392.00],  # C
            [164.81, 196.00, 246.94],  # Em
        ]
        bars = []
        bar_dur = 3.5
        for chord in chord_prog:
            pad = self._chord(chord, bar_dur, envelope="pad")
            melody_note = self._generate_tone(chord[2] * 2.0, bar_dur * 0.7, envelope="pluck")
            padded_melody = np.zeros(len(pad))
            padded_melody[:len(melody_note)] = melody_note * 0.4
            bars.append(pad * 0.7 + padded_melody)
        
        full_track = np.concatenate(bars)
        repeats = int(np.ceil(duration / (len(full_track) / self.sr)))
        return np.tile(full_track, repeats)[:int(self.sr * duration)]

    def synthesize_angry_track(self, duration: float = 10.0) -> np.ndarray:
        """Driving bass pulse and aggressive minor intervals with fast rhythmic cadence."""
        bass_notes = [110.00, 116.54, 110.00, 130.81]  # A2, Bb2, A2, C3
        pulses = []
        step_dur = 0.2
        for root in bass_notes:
            # Low punchy bass
            bass = self._generate_tone(root, step_dur, envelope="pluck")
            # Aggressive overtone
            high = self._generate_tone(root * 3.0, step_dur, envelope="pluck") * 0.4
            pulses.append(bass + high)

        full_pattern = np.concatenate(pulses)
        repeats = int(np.ceil(duration / (len(full_pattern) / self.sr)))
        return np.tile(full_pattern, repeats)[:int(self.sr * duration)]

    def synthesize_surprise_track(self, duration: float = 12.0) -> np.ndarray:
        """Vibrant synthwave arpeggiation with uplifting energetic chord shifts."""
        chords = [
            [293.66, 369.99, 440.00],  # D Major
            [329.63, 415.30, 493.88],  # E Major
            [277.18, 349.23, 415.30],  # C#m
            [369.99, 440.00, 554.37],  # F#m
        ]
        steps = []
        step_dur = 0.25
        for ch in chords:
            for freq in ch:
                steps.append(self._generate_tone(freq, step_dur, envelope="pluck"))
                steps.append(self._generate_tone(freq * 1.5, step_dur * 0.5, envelope="smooth"))

        full_pattern = np.concatenate(steps)
        repeats = int(np.ceil(duration / (len(full_pattern) / self.sr)))
        return np.tile(full_pattern, repeats)[:int(self.sr * duration)]

    def synthesize_neutral_track(self, duration: float = 15.0) -> np.ndarray:
        """Calm 432Hz ambient chord drone with relaxing gentle textures."""
        # 432 Hz tuning base
        drone_freqs = [216.0, 324.0, 432.0, 648.0]
        drone = self._chord(drone_freqs, duration, envelope="pad")
        return drone

    def generate_full_sample_library(self, target_dir: Path = MUSIC_DIR) -> Dict[str, List[Path]]:
        """
        Generates and writes a complete out-of-the-box local music library with
        categorized WAV files for all 5 emotions.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        created_files: Dict[str, List[Path]] = {e: [] for e in EMOTIONS}

        catalog = {
            "happy": [
                ("sunshine_groove.wav", self.synthesize_happy_track, "Sunshine Groove", 15.0),
                ("uplifting_morning.wav", self.synthesize_happy_track, "Uplifting Morning", 14.0),
            ],
            "sad": [
                ("rainy_window_reflections.wav", self.synthesize_sad_track, "Rainy Window", 16.0),
                ("melancholy_echoes.wav", self.synthesize_sad_track, "Melancholy Echoes", 15.0),
            ],
            "angry": [
                ("thunder_strike_rhythm.wav", self.synthesize_angry_track, "Thunder Strike", 12.0),
                ("driving_intensity.wav", self.synthesize_angry_track, "Driving Intensity", 14.0),
            ],
            "surprise": [
                ("electric_wonder.wav", self.synthesize_surprise_track, "Electric Wonder", 14.0),
                ("neon_discovery.wav", self.synthesize_surprise_track, "Neon Discovery", 13.0),
            ],
            "neutral": [
                ("ambient_zen_flow.wav", self.synthesize_neutral_track, "Ambient Zen Flow", 18.0),
                ("chill_focus_space.wav", self.synthesize_neutral_track, "Chill Focus Space", 16.0),
            ]
        }

        for emotion, track_list in catalog.items():
            emotion_folder = target_dir / emotion
            emotion_folder.mkdir(parents=True, exist_ok=True)

            for filename, generator_func, title, dur in track_list:
                file_path = emotion_folder / filename
                # Generate signal
                raw_audio = generator_func(duration=dur)
                # Normalize and convert to 16-bit PCM WAV
                max_val = np.max(np.abs(raw_audio)) + 1e-9
                normalized = (raw_audio / max_val * 32767.0).astype(np.int16)
                wavfile.write(str(file_path), self.sr, normalized)
                created_files[emotion].append(file_path)
                logger.info("Generated sample audio track: %s", file_path)

        return created_files
