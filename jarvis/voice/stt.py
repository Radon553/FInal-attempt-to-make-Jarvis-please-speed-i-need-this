"""
voice/stt.py — Speech-to-Text for Jarvis

Priority:
  1. faster-whisper (local) — runs Whisper on-device, zero API calls, no limits
  2. Groq Whisper (remote)  — fast cloud fallback, uses GROQ_API_KEY
  3. Google Speech (remote) — last resort, no key needed

Local Whisper model sizes (downloaded on first use):
  "base.en"   ~140 MB — fast, good enough for clear speech
  "small.en"  ~460 MB — better accuracy, still fast on Apple Silicon
  "medium.en" ~1.5 GB — best accuracy, slightly slower

Set WHISPER_MODEL in jarvis/.env to change, e.g.: WHISPER_MODEL=small.en

Key tuning:
  pause_threshold = 1.2s  — waits 1.2s silence before ending phrase.
                            0.8 cut off mid-sentence; 2.0 felt sluggish.
"""

import io
import os
import tempfile
from pathlib import Path

import speech_recognition as sr
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_GROQ_KEY      = os.getenv("GROQ_API_KEY")
_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")

# faster-whisper model — loaded once and reused (takes a few seconds first time)
_fw_model = None

# ---------------------------------------------------------------------------
# Recognizer settings
# ---------------------------------------------------------------------------

_recognizer = sr.Recognizer()
_recognizer.dynamic_energy_threshold  = True
_recognizer.energy_threshold          = 300
_recognizer.pause_threshold           = 1.2   # seconds of silence = end of phrase
                                               # 0.8 cut off mid-sentence; 1.2 feels natural
_recognizer.non_speaking_duration     = 0.3   # silence buffer before phrase begins

# Calibrate microphone once at startup — not on every listen() call
_calibrated = False


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibrate_once() -> None:
    """Adjust for ambient noise one time. No-op on subsequent calls."""
    global _calibrated
    if _calibrated:
        return
    with sr.Microphone() as source:
        print("[STT] Calibrating mic for ambient noise...")
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)
    _calibrated = True
    print("[STT] Ready.")


def recalibrate() -> None:
    """Force a fresh ambient-noise calibration. Call if you move rooms."""
    global _calibrated
    _calibrated = False
    _calibrate_once()


# ---------------------------------------------------------------------------
# Main listen function
# ---------------------------------------------------------------------------

def listen(timeout: int = 6, phrase_time_limit: int = 20) -> str | None:
    """
    Open the microphone, wait for speech, return transcribed text.

    Args:
        timeout:           Seconds to wait for speech to start (returns None if silent).
        phrase_time_limit: Max recording length in seconds before forcing transcription.

    Returns:
        Transcribed string, or None on silence/error.
    """
    _calibrate_once()
    with sr.Microphone() as source:
        print("[STT] Listening...")
        try:
            audio = _recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit,
            )
        except sr.WaitTimeoutError:
            print("[STT] No speech detected.")
            return None

    return _transcribe(audio)


def listen_continuously(callback, stop_phrase: str = "goodbye jarvis") -> None:
    """
    Listen in a loop, calling callback(text) for each recognized phrase.
    Stops when the stop_phrase is spoken.
    """
    _calibrate_once()
    print(f"[STT] Continuous mode — say '{stop_phrase}' to stop.")
    while True:
        text = listen()
        if text is None:
            continue
        if stop_phrase.lower() in text.lower():
            print("[STT] Stop phrase heard.")
            break
        callback(text)


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------

def _transcribe(audio: sr.AudioData) -> str | None:
    """Try backends in order: local faster-whisper → Groq → Google."""
    result = _transcribe_local(audio)
    if result:
        return result
    print("[STT] Local Whisper failed — trying Groq...")
    if _GROQ_KEY:
        result = _transcribe_groq(audio)
        if result:
            return result
        print("[STT] Groq failed — falling back to Google.")
    return _transcribe_google(audio)


def _transcribe_local(audio: sr.AudioData) -> str | None:
    """
    faster-whisper — runs Whisper entirely on-device, no API, no limits.
    Model is downloaded once (~140MB for base.en) and cached locally.
    """
    global _fw_model
    try:
        from faster_whisper import WhisperModel

        if _fw_model is None:
            print(f"[STT] Loading local Whisper model ({_WHISPER_MODEL})...")
            _fw_model = WhisperModel(
                _WHISPER_MODEL,
                device="cpu",
                compute_type="int8",   # int8 is fast and accurate on CPU
            )
            print("[STT] Local Whisper ready.")

        # Write audio to a temp WAV file (faster-whisper needs a file path)
        wav_bytes = audio.get_wav_data()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        segments, _ = _fw_model.transcribe(tmp_path, language="en")
        text = " ".join(s.text.strip() for s in segments).strip()
        os.unlink(tmp_path)

        print(f"[STT] Heard (local): {text}")
        return text or None

    except ImportError:
        print("[STT] faster-whisper not installed (pip install faster-whisper)")
        return None
    except Exception as e:
        print(f"[STT] Local Whisper error: {e}")
        return None


def _transcribe_groq(audio: sr.AudioData) -> str | None:
    """Groq Whisper — cloud fallback, fast but uses API quota."""
    try:
        from groq import Groq
        client = Groq(api_key=_GROQ_KEY)
        wav_bytes  = audio.get_wav_data()
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"
        print("[STT] Transcribing via Groq Whisper...")
        result = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",
        )
        text = result.strip() if isinstance(result, str) else result.text.strip()
        print(f"[STT] Heard (Groq): {text}")
        return text or None
    except Exception as e:
        print(f"[STT] Groq Whisper error: {e}")
        return None


def _transcribe_google(audio: sr.AudioData) -> str | None:
    """Google Web Speech — last resort, no key needed, least accurate."""
    print("[STT] Transcribing via Google...")
    try:
        text = _recognizer.recognize_google(audio)
        print(f"[STT] Heard (Google): {text}")
        return text
    except sr.UnknownValueError:
        print("[STT] Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"[STT] Google Speech unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== STT Test === (local Whisper: {_WHISPER_MODEL})")
    result = listen()
    print(f"\nTranscription: {result}" if result else "\nNothing transcribed.")
