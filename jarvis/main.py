"""
main.py — Jarvis entry point

Run from the jarvis/ directory:
    python main.py

Flow:
    Microphone → STT → LLM (Groq) → TTS → Speaker
"""

from voice.stt import listen
from voice.tts import speak
from brain.agent import get_response, clear_history

EXIT_PHRASES = {"exit", "quit", "goodbye jarvis", "bye jarvis"}


def main() -> None:
    print("[JARVIS] Starting up...")
    speak("Jarvis online. How can I help you?")

    while True:
        # 1. Capture voice input
        user_text = listen()

        # No speech detected — try again
        if not user_text:
            continue

        # 2. Check for exit command
        if any(phrase in user_text.lower() for phrase in EXIT_PHRASES):
            speak("Goodbye. Shutting down.")
            print("[JARVIS] Exiting.")
            break

        # 3. Get LLM response
        response = get_response(user_text)

        # 4. Speak the response
        speak(response)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[JARVIS] Interrupted. Shutting down.")
