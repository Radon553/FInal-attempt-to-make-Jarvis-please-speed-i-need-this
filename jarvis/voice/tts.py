"""
voice/tts.py — Text-to-Speech module for Jarvis

Primary:  ElevenLabs Neural TTS  — human-quality voice, Iron Man Jarvis-like
Fallback: edge-tts (Microsoft)   — used automatically if no ElevenLabs key set

Audio playback uses macOS's built-in `afplay` — no pygame, no SDL conflicts.

Install:
    pip install elevenlabs edge-tts python-dotenv
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

import edge_tts
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")

# ---------------------------------------------------------------------------
# ElevenLabs voice configuration
#
# Best Jarvis-like voices (all free tier):
#   "Daniel"  onwK4e9ZLuTAKqWW03F9  British, deep, calm, authoritative ← DEFAULT
#   "Adam"    pNInz6obpgDQGcFmaJgB  American, deep, professional
#   "Josh"    TxGEqnHWrfWFTfGW9XjX  American, deep, composed
# ---------------------------------------------------------------------------

VOICE_ID = "onwK4e9ZLuTAKqWW03F9"  # Daniel — British, authoritative, Jarvis-like
MODEL_ID  = "eleven_turbo_v2_5"     # Fastest model, lowest latency

_VOICE_SETTINGS = dict(
    stability=0.72,
    similarity_boost=0.82,
    style=0.38,
    use_speaker_boost=True,
)

# ---------------------------------------------------------------------------
# edge-tts fallback
# ---------------------------------------------------------------------------

_FALLBACK_VOICE = "en-GB-RyanNeural"
_FALLBACK_RATE  = "+8%"
_FALLBACK_PITCH = "-6Hz"

# ---------------------------------------------------------------------------
# Internal: audio playback via afplay (macOS built-in, no SDL)
# ---------------------------------------------------------------------------

def _play_mp3_file(filepath: str) -> None:
    """Play an MP3 file and block until playback finishes."""
    subprocess.run(["afplay", filepath], check=True)


def _play_mp3_bytes(audio_bytes: bytes) -> None:
    """Write audio bytes to a temp file and play it via afplay."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmpfile = f.name
    try:
        _play_mp3_file(tmpfile)
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Internal: ElevenLabs
# ---------------------------------------------------------------------------

def _speak_elevenlabs(text: str) -> None:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings

    client = ElevenLabs(api_key=_ELEVENLABS_KEY)
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=MODEL_ID,
        voice_settings=VoiceSettings(**_VOICE_SETTINGS),
        output_format="mp3_44100_128",
    )
    _play_mp3_bytes(b"".join(audio_stream))


# ---------------------------------------------------------------------------
# Internal: edge-tts fallback
# ---------------------------------------------------------------------------

async def _synthesize_edge(text: str, filepath: str) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=_FALLBACK_VOICE,
        rate=_FALLBACK_RATE,
        pitch=_FALLBACK_PITCH,
    )
    await communicate.save(filepath)


def _speak_edge(text: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmpfile = f.name
    try:
        asyncio.run(_synthesize_edge(text, tmpfile))
        _play_mp3_file(tmpfile)
    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(text: str) -> None:
    """
    Speak text aloud using ElevenLabs (or edge-tts as fallback).
    Blocks until audio finishes playing.

    Args:
        text: The string to speak. Empty strings are silently ignored.
    """
    if not text or not text.strip():
        return

    print(f"[TTS] Speaking: {text}")

    if _ELEVENLABS_KEY:
        try:
            _speak_elevenlabs(text)
            return
        except Exception as e:
            print(f"[TTS] ElevenLabs failed ({e}), falling back to edge-tts.")

    _speak_edge(text)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    speak(
        "Good evening. I am Jarvis, your personal AI assistant. "
        "All systems are online and fully operational. "
        "How may I assist you today?"
    )
