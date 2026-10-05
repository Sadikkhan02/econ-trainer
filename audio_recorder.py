"""Microphone audio recorder using sounddevice for English Conversation Trainer.

Captures mono audio from the microphone with clean start/stop controls
and saves to a temporary WAV file for Groq Whisper transcription.
"""

import ctypes
import ctypes.util
import os
import queue
import tempfile
from typing import Optional

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

import numpy as np
from scipy.io import wavfile
import sounddevice as sd


class AudioRecorder:
    """Manages interactive microphone recording sessions."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def record_interactive(self) -> Optional[str]:
        """Prompts user to press Enter to start, records until Enter is pressed again.

        Returns path to a temporary WAV file, or None if cancelled/failed.
        """
        print("\n----------------------------------------")
        print("🎤 YOUR TURN")
        print("Press Enter to start recording.")
        print("----------------------------------------")

        try:
            input_val = input().strip()
            if input_val.lower() == "q":
                return "QUIT"
        except (KeyboardInterrupt, EOFError):
            return "QUIT"

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass  # Ignore audio buffer status warnings
            audio_queue.put(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=audio_callback,
            )
        except Exception as e:
            print(f"\n[Microphone error: {e}]")
            return None

        print("\n🔴 Recording... Speak now. (Press Enter to stop)")

        try:
            with stream:
                try:
                    input()
                except (KeyboardInterrupt, EOFError):
                    print("\nRecording interrupted.")
                    return None
        except Exception as e:
            print(f"\n[Recording stream error: {e}]")
            return None

        print("⏹ Recording stopped.")

        frames = []
        while not audio_queue.empty():
            frames.append(audio_queue.get())

        if not frames:
            print("⚠️ No audio data was recorded.")
            return None

        audio_data = np.concatenate(frames, axis=0)

        # Check if recording is virtually empty or very short (< 0.2s)
        total_samples = len(audio_data)
        duration_sec = total_samples / self.sample_rate
        if duration_sec < 0.2:
            print("⚠️ Recording was too short.")
            return None

        # Check energy / RMS to detect completely silent audio
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 100:  # Int16 threshold for dead silence
            print("⚠️ No sound detected from microphone.")
            return None

        # Save to temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = temp_wav.name
        temp_wav.close()

        try:
            wavfile.write(wav_path, self.sample_rate, audio_data)
            return wav_path
        except Exception as e:
            print(f"⚠️ Failed to write audio file: {e}")
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return None

    @staticmethod
    def cleanup_file(wav_path: Optional[str]) -> None:
        """Safely delete temporary WAV file."""
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
