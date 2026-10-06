"""
Gemini Multi-Modal & NLP Emotion Classification Engine.
Features:
1. Powered by ChatGoogleGenerativeAI (Gemini Flash / Pro) with temperature=0.
2. Supports Deep Semantic Thought & Text Analysis with multilingual support (English, Hindi, Hinglish, Spanish, etc.).
3. Supports Multimodal Facial Image / Vision Emotion Analysis directly from facial image bytes / PIL.
4. Returns calibrated JSON probabilities, dominant emotion, confidence, semantic rationale, and music themes.
5. Provides seamless fallback if API key is not configured or offline.
"""

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from src.config import EMOTIONS
from src.utils import logger


class GeminiEmotionClassifier:
    """
    LLM and Multimodal Emotion Classifier leveraging Google Gemini via ChatGoogleGenerativeAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
        temperature: float = 0.0,
    ):
        from src.config import DEFAULT_GOOGLE_API_KEY, GEMINI_DEFAULT_MODEL
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or DEFAULT_GOOGLE_API_KEY
            or ""
        ).strip()
        self.model_name = model_name or "gemini-3.6-flash"
        self.temperature = temperature
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """Initializes ChatGoogleGenerativeAI instance if API key is present."""
        if not self.api_key:
            logger.info("GeminiEmotionClassifier: No API Key provided yet. Ready for key input.")
            self.llm = None
            return

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            target_model = self.model_name or "gemini-3.6-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=target_model,
                temperature=self.temperature,
                google_api_key=self.api_key,
                max_retries=2,
            )
            logger.info("GeminiEmotionClassifier successfully initialized with model: %s", target_model)
        except Exception as e:
            logger.error("Failed to initialize ChatGoogleGenerativeAI: %s", e)
            self.llm = None

    def set_api_key(self, api_key: str):
        """Updates API key and re-initializes the Gemini model."""
        clean_key = (api_key or "").strip()
        if clean_key != self.api_key:
            self.api_key = clean_key
            self._init_llm()

    def set_model_name(self, model_name: str):
        """Updates model name and re-initializes."""
        if model_name != self.model_name:
            self.model_name = model_name
            self._init_llm()

    def is_configured(self) -> bool:
        """Returns True if Gemini LLM instance is ready with an API key."""
        return self.llm is not None and bool(self.api_key)

    @staticmethod
    def _parse_content_str(content: Any) -> str:
        """Converts response content (str or list of parts) into a clean string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content or "")

    def _invoke_llm(self, messages: List[Any]) -> str:
        """Invokes Gemini LLM with automatic failover across models if a preview model is busy."""
        candidate_models = [self.model_name, "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
        seen = set()
        models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

        last_error = None
        for m_name in models_to_try:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=m_name,
                    temperature=self.temperature,
                    google_api_key=self.api_key,
                    max_retries=1,
                )
                response = llm.invoke(messages)
                return self._parse_content_str(response.content if hasattr(response, "content") else response)
            except Exception as e:
                last_error = e
                logger.warning("Gemini model '%s' request encountered issue (%s). Retrying next candidate...", m_name, e)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("All Gemini model invocations failed.")

    def _extract_json_payload(self, text_output: Any) -> Optional[Dict[str, Any]]:
        """Safely parses structured JSON output from Gemini response."""
        clean_str = self._parse_content_str(text_output)
        if not clean_str:
            return None

        # Look for ```json ... ``` markdown fence
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_str)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception:
                pass

        # Look for raw { ... } block
        brace_match = re.search(r"\{[\s\S]*\}", clean_str)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except Exception:
                pass

        return None

    def predict_text_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyzes emotion in user text, thought journal, or statement using Gemini LLM.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return {
                "dominant_emotion": "neutral",
                "confidence": 0.50,
                "probabilities": {e: 0.20 for e in EMOTIONS},
                "detected_keywords": [],
                "explanation": "No text provided.",
                "music_theme": "Peaceful Ambient",
                "engine": f"Gemini ({self.model_name})",
            }

        if not self.is_configured():
            return {
                "dominant_emotion": "neutral",
                "confidence": 0.50,
                "probabilities": {e: 0.20 for e in EMOTIONS},
                "detected_keywords": [],
                "explanation": "Google Gemini API Key not configured. Please enter your Gemini API Key in the sidebar.",
                "music_theme": "Ambient Zen",
                "engine": "Gemini (ChatGoogleGenerativeAI) - Awaiting API Key",
            }

        prompt = f"""You are an expert AI psychologist and music affective computing engine.
Analyze the emotional sentiment and affective state of the following user thought / journal entry:

\"\"\"{clean_text}\"\"\"

Allowed 5 emotion categories:
1. "happy" (Joy, celebration, excitement, triumph, optimism, feeling blessed)
2. "sad" (Grief, sorrow, crying, loneliness, heartbreak, disappointment, depression)
3. "angry" (Frustration, rage, annoyance, irritation, resentment)
4. "surprise" (Shock, astonishment, unexpected discovery, awe, wonder)
5. "neutral" (Calm, relaxed, thoughtful, everyday routine, peace, balanced)

Return ONLY a valid JSON object matching this exact schema:
{{
  "dominant_emotion": "happy" | "sad" | "angry" | "surprise" | "neutral",
  "confidence": 0.95,
  "probabilities": {{
    "happy": 0.05,
    "sad": 0.85,
    "angry": 0.05,
    "surprise": 0.00,
    "neutral": 0.05
  }},
  "detected_keywords": ["lonely", "miss", "tears"],
  "explanation": "Brief 1-2 sentence rationale for the emotional diagnosis.",
  "music_theme": "Acoustic healing & comforting melancholy melodies"
}}
"""

        try:
            from langchain_core.messages import HumanMessage

            response_text = self._invoke_llm([HumanMessage(content=prompt)])
            parsed = self._extract_json_payload(response_text)
            if parsed and "dominant_emotion" in parsed:
                dom_emo = str(parsed.get("dominant_emotion", "neutral")).lower().strip()
                if dom_emo not in EMOTIONS:
                    dom_emo = "neutral"

                conf = float(parsed.get("confidence", 0.85))
                probs = parsed.get("probabilities", {})
                
                # Normalize probabilities
                clean_probs = {}
                total_p = 0.0
                for e in EMOTIONS:
                    val = float(probs.get(e, 0.20 if e == dom_emo else 0.05))
                    clean_probs[e] = val
                    total_p += val
                if total_p > 0:
                    clean_probs = {k: round(v / total_p, 3) for k, v in clean_probs.items()}

                return {
                    "dominant_emotion": dom_emo,
                    "confidence": conf,
                    "probabilities": clean_probs,
                    "detected_keywords": parsed.get("detected_keywords", []),
                    "explanation": parsed.get("explanation", f"Detected {dom_emo} via Gemini reasoning."),
                    "music_theme": parsed.get("music_theme", f"{dom_emo.capitalize()} Melodies"),
                    "engine": f"Google Gemini ({self.model_name})",
                    "raw_response": response_text
                }
        except Exception as e:
            logger.error("Gemini text emotion inference failed: %s", e)

        return {
            "dominant_emotion": "neutral",
            "confidence": 0.50,
            "probabilities": {e: 0.20 for e in EMOTIONS},
            "detected_keywords": [],
            "explanation": f"Gemini query encountered an error. Please verify your API Key.",
            "music_theme": "Ambient",
            "engine": "Gemini (ChatGoogleGenerativeAI)",
        }

    def predict_vision_emotion(self, image_input: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Multimodal visual facial emotion classification using Gemini Flash Vision.
        """
        if not self.is_configured():
            return {
                "dominant_emotion": "neutral",
                "confidence": 0.50,
                "probabilities": {e: 0.20 for e in EMOTIONS},
                "explanation": "Gemini Vision requires a valid Google API Key configured in the sidebar.",
                "backend": "Gemini Vision (ChatGoogleGenerativeAI) - Awaiting API Key"
            }

        # Convert input to PIL Image
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume BGR from OpenCV
                pil_img = Image.fromarray(image_input[:, :, ::-1])
            else:
                pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise ValueError("Unsupported image input type for Gemini Vision")

        # Encode PIL image to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = """Analyze the facial expression and emotional mood of the person in this photo.
Carefully distinguish between genuine happiness, neutral expressions, painful grimaces (distress/sadness), and anger.

Target categories: 'happy', 'sad', 'angry', 'surprise', 'neutral'.

Return ONLY a valid JSON object matching:
{
  "dominant_emotion": "happy" | "sad" | "angry" | "surprise" | "neutral",
  "confidence": 0.92,
  "probabilities": {
    "happy": 0.05,
    "sad": 0.85,
    "angry": 0.05,
    "surprise": 0.00,
    "neutral": 0.05
  },
  "explanation": "1-2 sentence description of facial micro-expressions (eyebrows, mouth, eyes)."
}
"""

        try:
            from langchain_core.messages import HumanMessage

            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]

            response_text = self._invoke_llm([HumanMessage(content=message_content)])
            parsed = self._extract_json_payload(response_text)
            if parsed and "dominant_emotion" in parsed:
                dom_emo = str(parsed.get("dominant_emotion", "neutral")).lower().strip()
                if dom_emo not in EMOTIONS:
                    dom_emo = "neutral"

                conf = float(parsed.get("confidence", 0.85))
                probs = parsed.get("probabilities", {})

                clean_probs = {}
                total_p = 0.0
                for e in EMOTIONS:
                    val = float(probs.get(e, 0.20 if e == dom_emo else 0.05))
                    clean_probs[e] = val
                    total_p += val
                if total_p > 0:
                    clean_probs = {k: round(v / total_p, 3) for k, v in clean_probs.items()}

                return {
                    "dominant_emotion": dom_emo,
                    "display_emotion": dom_emo,
                    "confidence": conf,
                    "probabilities": clean_probs,
                    "is_uncertain": conf < 0.35,
                    "status_label": f"{dom_emo.capitalize()} ({conf * 100:.1f}%)",
                    "explanation": parsed.get("explanation", ""),
                    "backend": f"Google Gemini Flash ({self.model_name})"
                }
        except Exception as e:
            logger.error("Gemini Vision classification failed: %s", e)

        return {
            "dominant_emotion": "neutral",
            "display_emotion": "neutral",
            "confidence": 0.50,
            "probabilities": {e: 0.20 for e in EMOTIONS},
            "is_uncertain": True,
            "status_label": "Neutral (Fallback)",
            "explanation": "Gemini Vision inference error. Falling back to baseline.",
            "backend": "Gemini Vision (ChatGoogleGenerativeAI)"
        }
