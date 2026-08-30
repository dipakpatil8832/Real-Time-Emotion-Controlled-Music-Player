"""
Temporal Emotion Smoothing Module.
Implements Exponential Moving Average (EMA) and sliding window voting
to prevent frame-to-frame emotion jitter, facial micro-twitch flicker, and erratic song switching.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.config import EMOTIONS, SmoothingConfig, smoothing_config
from src.utils import logger


class EmotionSmoother:
    """
    Stabilizes raw frame-by-frame emotion probabilities using a combination of:
    1. Exponential Moving Average (EMA) on probability vectors.
    2. Fixed-length FIFO sliding window buffer.
    3. Hysteresis switching logic to require sustained emotional shift before triggering music changes.
    """

    def __init__(
        self,
        window_size: int = smoothing_config.window_size,
        decay_factor: float = smoothing_config.decay_factor,
        emotions: Optional[List[str]] = None,
        switch_confidence_gain: float = smoothing_config.switch_confidence_gain
    ):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.emotions = emotions or EMOTIONS
        self.switch_confidence_gain = switch_confidence_gain

        # History structures
        self.history: Deque[str] = deque(maxlen=window_size)
        self.prob_history: Deque[Dict[str, float]] = deque(maxlen=window_size)
        
        # Current smoothed state
        self.current_smoothed_probs: Dict[str, float] = {e: 1.0 / len(self.emotions) for e in self.emotions}
        self.current_dominant_emotion: str = "neutral"
        self.current_confidence: float = 0.5
        self.stability_index: float = 1.0  # Fraction of recent buffer matching dominant emotion

    def update(
        self,
        raw_emotion: str,
        raw_confidence: float,
        raw_probs: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, float], float]:
        """
        Ingests a new raw detection frame and returns stabilized values.

        Args:
            raw_emotion: Emotion with max probability in the latest frame.
            raw_confidence: Confidence score of the latest frame (0.0 - 1.0).
            raw_probs: Full dictionary of class probabilities.

        Returns:
            Tuple of:
            - smoothed_emotion (str): Stabilized dominant emotion
            - smoothed_confidence (float): Stabilized confidence score
            - smoothed_probs (dict): Stabilized probability distribution
            - stability_index (float): Stability metric [0.0 - 1.0] indicating consistency
        """
        # Append to sliding window
        self.history.append(raw_emotion)
        self.prob_history.append(raw_probs)

        # Fast initialization on first observation to prevent initial neutral lock-in
        if len(self.history) == 1:
            self.current_smoothed_probs = raw_probs.copy()
            self.current_dominant_emotion = raw_emotion
            self.current_confidence = raw_confidence
            self.stability_index = 1.0
            return (
                self.current_dominant_emotion,
                self.current_confidence,
                self.current_smoothed_probs.copy(),
                self.stability_index
            )

        # 1. Update Exponential Moving Average (EMA) for each class probability
        for emotion in self.emotions:
            latest_p = raw_probs.get(emotion, 0.0)
            prev_p = self.current_smoothed_probs.get(emotion, 1.0 / len(self.emotions))
            self.current_smoothed_probs[emotion] = (
                self.decay_factor * prev_p + (1.0 - self.decay_factor) * latest_p
            )

        # Normalize probability vector sum to 1.0
        total_p = sum(self.current_smoothed_probs.values()) + 1e-9
        for emotion in self.emotions:
            self.current_smoothed_probs[emotion] /= total_p

        # 2. Find candidate dominant emotion from smoothed probabilities
        candidate_emotion = max(self.current_smoothed_probs, key=self.current_smoothed_probs.get)
        candidate_confidence = self.current_smoothed_probs[candidate_emotion]

        # 3. Calculate Stability Index (Agreement ratio in sliding window)
        votes = list(self.history)
        candidate_votes = votes.count(candidate_emotion)
        self.stability_index = candidate_votes / max(1, len(votes))

        # 4. Hysteresis switching logic:
        # If candidate differs from current emotion, require either:
        # - High buffer consensus (>= 50%), OR
        # - Substantially higher confidence than current
        if candidate_emotion != self.current_dominant_emotion:
            current_conf = self.current_smoothed_probs.get(self.current_dominant_emotion, 0.0)
            threshold_conf = current_conf * self.switch_confidence_gain
            
            if (self.stability_index >= 0.50) or (candidate_confidence > threshold_conf and len(votes) >= 2):
                logger.info(
                    "Emotion transitioned: %s (%.2f) -> %s (%.2f) [Stability: %.2f]",
                    self.current_dominant_emotion, current_conf,
                    candidate_emotion, candidate_confidence, self.stability_index
                )
                self.current_dominant_emotion = candidate_emotion
        else:
            self.current_dominant_emotion = candidate_emotion

        self.current_confidence = self.current_smoothed_probs[self.current_dominant_emotion]

        return (
            self.current_dominant_emotion,
            self.current_confidence,
            self.current_smoothed_probs.copy(),
            self.stability_index
        )

    def reset(self) -> None:
        """Clears smoothing history buffers."""
        self.history.clear()
        self.prob_history.clear()
        self.current_smoothed_probs = {e: 1.0 / len(self.emotions) for e in self.emotions}
        self.current_dominant_emotion = "neutral"
        self.current_confidence = 0.5
        self.stability_index = 1.0
