"""Text-to-speech module using Groq TTS for English Conversation Trainer.

Converts bot dialogue lines to speech using Groq TTS and plays audio locally via sounddevice.
"""

import ctypes
import ctypes.util
import os
import tempfile
from typing import Optional
from groq import Groq

# Ensure local portaudio / jack libraries can be found dynamically on Linux
_local_lib = os.path.expanduser("~/.local/usr/lib/x86_64-linux-gnu")
_jack = os.path.join(_local_lib, "libjack.so.0")
if os.path.exists(_jack):
    try:
        ctypes.CDLL(_jack, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

_orig_find = ctypes.util.find_library


def _custom_find_library(name: str):
    if "portaudio" in name:
        local_p = os.path.join(_local_lib, "libportaudio.so.2")
        if os.path.exists(local_p):
            return local_p
    return _orig_find(name)


ctypes.util.find_library = _custom_find_library

from scipy.io import wavfile
import sounddevice as sd


class TextToSpeech:
    """Wrapper for Groq Text-to-Speech synthesis and local playback."""

    def __init__(
        self,
        client: Groq,
        model: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        self.client = client
        self.model = model or os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english").strip()
        self.voice = voice or os.environ.get("GROQ_TTS_VOICE", "autumn").strip()
        self._warned_terms = False

    def speak(self, text: str) -> bool:
        """Synthesize text into speech via Groq TTS and play audio locally.

        Waits for playback to complete and cleans up temporary audio file.
        Returns True if audio played successfully, False otherwise.
        """
        if not text or not text.strip():
            return False

        temp_wav_path = None
        try:
            # Request audio from Groq TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                input=text,
                voice=self.voice,
                response_format="wav",
            )

            audio_bytes = response.content
            if not audio_bytes:
                return False

            # Write temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav_path = temp_file.name
            temp_file.write(audio_bytes)
            temp_file.close()

            # Read and play audio locally
            sr, audio_data = wavfile.read(temp_wav_path)
            sd.play(audio_data, sr)
            sd.wait()
            return True

        except Exception as e:
            err_str = str(e)
            if "model_terms_required" in err_str:
                if not self._warned_terms:
                    print(
                        f"\n[Voice synthesis notice: Model `{self.model}` requires terms acceptance at:\n"
                        f" https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english ]"
                    )
                    self._warned_terms = True
            else:
                print(f"\n[Voice synthesis note: {e}]")
            return False

        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass
