"""
vision/screen_vision.py — Screen capture and visual understanding for Jarvis

Vision backends (tried in order, first available wins):

  1. Groq vision   — FREE, uses your existing GROQ_API_KEY
                     Model: llama-3.2-11b-vision-preview
                     Fast, no new sign-up needed.

  2. Gemini Flash  — FREE tier (Google), needs GEMINI_API_KEY
                     Get one free at https://aistudio.google.com/
                     pip install google-generativeai

  3. OCR fallback  — No API key, works offline
                     Unreliable on dark UIs (Spotify, etc.)
                     brew install tesseract && pip install pytesseract

Install:
    pip install mss pillow groq
    pip install google-generativeai   # optional, Gemini backup
    brew install tesseract            # optional, OCR fallback
"""

import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_GROQ_KEY   = os.getenv("GROQ_API_KEY")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Groq vision models to try in order — first one that works is used
_GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout — best, free
    "llama-3.2-11b-vision-preview",                # Llama 3.2 Vision — reliable fallback
]

# Max width sent to vision model — keeps latency + cost low
_VISION_MAX_WIDTH = 1280

# ---------------------------------------------------------------------------
# Screenshot capture — mss (fast) or pyautogui (fallback)
# ---------------------------------------------------------------------------

try:
    import mss
    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False

from PIL import Image


def capture_screenshot(region: Optional[dict] = None) -> Image.Image:
    """
    Capture the screen and return an RGB PIL Image.

    Args:
        region: Optional {top, left, width, height} in pixels.
                None → full primary monitor.
    """
    if _MSS_AVAILABLE:
        return _capture_mss(region)
    return _capture_pyautogui(region)


def _capture_mss(region: Optional[dict]) -> Image.Image:
    with mss.mss() as sct:
        monitor = region if region else sct.monitors[1]
        raw = sct.grab(monitor)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def _capture_pyautogui(region: Optional[dict]) -> Image.Image:
    import pyautogui
    if region:
        return pyautogui.screenshot(
            region=(region["left"], region["top"], region["width"], region["height"])
        )
    return pyautogui.screenshot()


def _image_to_base64(image: Image.Image) -> str:
    """Resize to _VISION_MAX_WIDTH if needed, encode as base64 PNG string."""
    w, h = image.size
    if w > _VISION_MAX_WIDTH:
        scale = _VISION_MAX_WIDTH / w
        image = image.resize((_VISION_MAX_WIDTH, int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Local Ollama vision model — already downloaded, no limits
_OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:latest")
_OLLAMA_BASE         = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Vision model — primary sight capability
# Priority: Ollama (local, free, no limits) → Groq → Gemini → error
# ---------------------------------------------------------------------------

def _ask_vision_model(image_b64: str, prompt: str) -> str:
    """
    Send a screenshot + prompt to the best available vision model.

    Priority:
      1. Ollama llava  — local, completely free, no rate limits, works offline
      2. Groq vision   — remote, free tier, needs GROQ_API_KEY
      3. Gemini Flash  — remote, free tier, needs GEMINI_API_KEY
    """
    # Try local Ollama first
    result = _ask_ollama_vision(image_b64, prompt)
    if not result.startswith("[ERROR]"):
        return result
    print(f"[VISION] Ollama vision failed: {result}")

    # Groq fallback
    if _GROQ_KEY:
        result = _ask_groq_vision(image_b64, prompt)
        if not result.startswith("[ERROR]"):
            return result
        print(f"[VISION] Groq vision failed: {result}")

    # Gemini fallback
    if _GEMINI_KEY:
        result = _ask_gemini_vision(image_b64, prompt)
        if not result.startswith("[ERROR]"):
            return result
        print(f"[VISION] Gemini vision failed: {result}")

    return "[ERROR] All vision backends failed. Make sure Ollama is running (`ollama serve`)."


def _ask_ollama_vision(image_b64: str, prompt: str) -> str:
    """
    Use local Ollama llava for vision — no API key, no rate limits.
    llava:latest is already downloaded on this machine.
    """
    try:
        import urllib.request
        # Quick reachability check before trying
        urllib.request.urlopen(f"{_OLLAMA_BASE}/api/tags", timeout=1)
    except Exception:
        return "[ERROR] Ollama not running. Start it with: ollama serve"

    try:
        from openai import OpenAI
        client = OpenAI(base_url=f"{_OLLAMA_BASE}/v1", api_key="ollama")

        print(f"[VISION] Asking local {_OLLAMA_VISION_MODEL}...")
        response = client.chat.completions.create(
            model=_OLLAMA_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[ERROR] Ollama vision: {e}"


def _ask_groq_vision(image_b64: str, prompt: str) -> str:
    """Use Groq's vision model — tries each model in _GROQ_VISION_MODELS until one works."""
    try:
        from groq import Groq
        client = Groq(api_key=_GROQ_KEY)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        # Groq accepts base64 data URLs (same format as OpenAI)
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }]

        last_error = None
        for model in _GROQ_VISION_MODELS:
            try:
                print(f"[VISION] Trying Groq model: {model}")
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                )
                return response.choices[0].message.content.strip()
            except Exception as model_err:
                print(f"[VISION] {model} failed: {model_err}")
                last_error = model_err

        return f"[ERROR] All Groq vision models failed. Last error: {last_error}"

    except Exception as e:
        return f"[ERROR] Groq vision client error: {e}"


def _ask_gemini_vision(image_b64: str, prompt: str) -> str:
    """Use Google Gemini Flash (free tier) as a backup vision model."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=_GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Gemini accepts PIL images directly
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes))

        response = model.generate_content([prompt, image])
        return response.text.strip()

    except Exception as e:
        return f"[ERROR] Gemini vision: {e}"


# ---------------------------------------------------------------------------
# Public vision functions used by the agent
# ---------------------------------------------------------------------------

def vision_describe_screen(region: Optional[dict] = None) -> str:
    """
    Take a screenshot and ask the vision model to describe what's on screen.

    Returns a natural language description: which app is open, visible text,
    sidebar items, buttons, playlist names, etc.

    Works on dark themes, graphical UIs, custom fonts — anything visible.
    """
    image = capture_screenshot(region)
    b64   = _image_to_base64(image)

    prompt = (
        "You are helping an AI assistant understand what's on a computer screen. "
        "Describe concisely: which app is open, what's in the main area, "
        "what's in the sidebar or panels, any visible text labels, playlist or file names, "
        "buttons, and the overall UI state. Be specific about every text label you can read."
    )
    return _ask_vision_model(b64, prompt)


def vision_find_element(
    description: str,
    region: Optional[dict] = None,
) -> Optional[tuple[int, int]]:
    """
    Ask the vision model to find a UI element and return click coordinates.

    Works with: dark themes, custom fonts, icons, Spotify playlists, anything visual.

    Args:
        description: What to find — e.g. "metal playlist in sidebar",
                     "Play button", "Search bar", "Liked Songs".
        region:      Optional screen region to restrict the search.

    Returns:
        (x, y) pixel coordinates of the element center, or None if not found.
    """
    image = capture_screenshot(region)
    w, h  = image.size
    b64   = _image_to_base64(image)

    prompt = f"""You are a UI locator assistant. Examine this screenshot carefully.

Find the element described as: "{description}"

Reply with ONLY a JSON object — no explanation, no markdown code block:
- If found:   {{"found": true, "x_pct": 0.45, "y_pct": 0.62, "label": "exact text you see"}}
- If missing: {{"found": false, "reason": "brief note on what you see instead"}}

x_pct and y_pct are the element's CENTER position as decimal fractions of image width/height.
  0.0 = left/top edge,   1.0 = right/bottom edge
Be as precise as possible — this value is used to move the mouse and click."""

    raw = _ask_vision_model(b64, prompt)
    print(f"[VISION] Model response: {raw}")

    # Pull the JSON object out of the response (model sometimes adds extra text)
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if not match:
        print("[VISION] Could not parse JSON from vision response.")
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"[VISION] JSON parse error: {e}")
        return None

    if not data.get("found"):
        print(f"[VISION] Not found: {data.get('reason', '?')}")
        return None

    x     = int(data["x_pct"] * w)
    y     = int(data["y_pct"] * h)
    label = data.get("label", description)
    print(f"[VISION] Found '{label}' at ({x}, {y})")
    return (x, y)


def vision_query(question: str, region: Optional[dict] = None) -> str:
    """
    Ask a specific question about the current screen and get an answer.

    Example:
        vision_query("What playlists are in the Spotify sidebar?")
        vision_query("Is there a play button visible?")
    """
    image = capture_screenshot(region)
    b64   = _image_to_base64(image)
    return _ask_vision_model(b64, question)


# ---------------------------------------------------------------------------
# OCR fallback — lightweight, no API key, unreliable on dark/graphical UIs
# ---------------------------------------------------------------------------

try:
    import pytesseract
    _TESS_CONFIG    = "--psm 11 --oem 3"
    _SCALE_FACTOR   = 1.5
    _MIN_CONFIDENCE = 40
    _OCR_AVAILABLE  = True
except ImportError:
    _OCR_AVAILABLE  = False


def extract_text_blocks(image: Image.Image) -> list[dict]:
    """OCR — returns word-level text blocks with positions. Fallback only."""
    if not _OCR_AVAILABLE:
        return []

    orig_w, orig_h = image.size
    scaled = image.resize(
        (int(orig_w * _SCALE_FACTOR), int(orig_h * _SCALE_FACTOR)),
        Image.LANCZOS,
    )
    data = pytesseract.image_to_data(
        scaled, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT,
    )
    blocks = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < _MIN_CONFIDENCE:
            continue
        x  = int(data["left"][i]   / _SCALE_FACTOR)
        y  = int(data["top"][i]    / _SCALE_FACTOR)
        bw = int(data["width"][i]  / _SCALE_FACTOR)
        bh = int(data["height"][i] / _SCALE_FACTOR)
        blocks.append({
            "text": text, "x": x, "y": y, "w": bw, "h": bh,
            "center": (x + bw // 2, y + bh // 2), "conf": conf,
        })
    return blocks


def get_screen_text(region: Optional[dict] = None) -> str:
    """OCR text dump. Used only when no vision API is available."""
    if not _OCR_AVAILABLE:
        return "[No OCR available — install pytesseract]"
    image = capture_screenshot(region)
    return pytesseract.image_to_string(image, config=_TESS_CONFIG).strip()


def analyze_screen(region: Optional[dict] = None) -> dict:
    """OCR-based structured screen analysis. Kept for simple text-UI cases."""
    t0 = time.perf_counter()
    image  = capture_screenshot(region)
    blocks = extract_text_blocks(image)
    return {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "resolution":  image.size,
        "text_blocks": blocks,
        "all_text":    "\n".join(b["text"] for b in blocks),
        "word_count":  len(blocks),
        "elapsed_ms":  round((time.perf_counter() - t0) * 1000),
    }


def find_text_on_screen(
    query: str,
    region: Optional[dict] = None,
) -> Optional[tuple[int, int]]:
    """
    Find text on screen → return click coordinates.
    Tries OCR first (fast), falls back to vision model (accurate).
    """
    # Pass 1: OCR
    if _OCR_AVAILABLE:
        info = analyze_screen(region)
        q = query.lower()
        matches = [b for b in info["text_blocks"] if q in b["text"].lower()]
        if not matches:
            # Try multi-word sliding window
            words = q.split()
            if len(words) >= 2:
                for i in range(len(info["text_blocks"]) - len(words) + 1):
                    window = info["text_blocks"][i: i + len(words)]
                    if q in " ".join(b["text"].lower() for b in window):
                        matches.append(window[0])
        if matches:
            best = max(matches, key=lambda b: b["conf"])
            print(f"[VISION] OCR found '{best['text']}' at {best['center']}")
            return best["center"]
        print(f"[VISION] OCR missed '{query}' — trying vision model...")

    # Pass 2: Vision model
    return vision_find_element(query, region)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Jarvis Vision Test ===\n")

    if _GROQ_KEY:
        print(f"Vision backend: Groq ({_GROQ_VISION_MODELS[0]})")
    elif _GEMINI_KEY:
        print("Vision backend: Gemini Flash")
    else:
        print("WARNING: No vision API key found. Only OCR will work.")

    print("\nDescribing screen...")
    desc = vision_describe_screen()
    print(f"\n[Vision model says]\n{desc}\n")

    query = input("Find and locate an element (Enter to skip): ").strip()
    if query:
        coords = find_text_on_screen(query)
        print(f"Result: {coords if coords else 'Not found'}")
