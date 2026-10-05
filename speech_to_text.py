"""Speech-to-text module using Groq Whisper for English Conversation Trainer.

Sends recorded WAV audio to Groq Whisper and returns structured transcription results.
"""

import os
from typing import Any, Dict
from groq import Groq


class SpeechToText:
    """Wrapper for Groq Whisper speech transcription."""

    def __init__(self, client: Groq, model: str | None = None):
        self.client = client
        self.model = model or os.environ.get("GROQ_STT_MODEL", "whisper-large-v3").strip()

    def transcribe_file(self, wav_path: str) -> Dict[str, Any]:
        """Transcribes a WAV file using Groq Whisper.

        Returns a dictionary with:
        {"text": str, "success": bool, "error": Optional[str]}
        """
        if not wav_path or not os.path.exists(wav_path):
            return {
                "text": "",
                "success": False,
                "error": "Audio file does not exist or was empty.",
            }

        try:
            with open(wav_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language="en",
                    temperature=0.0,
                    response_format="json",
                )

            text = transcription.text.strip()
            if not text:
                return {
                    "text": "",
                    "success": False,
                    "error": "No speech detected in recording.",
                }

            return {
                "text": text,
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "text": "",
                "success": False,
                "error": str(e),
            }
