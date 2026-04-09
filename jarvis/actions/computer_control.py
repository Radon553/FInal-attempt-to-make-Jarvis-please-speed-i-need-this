"""
actions/computer_control.py — Computer control module for Jarvis

macOS security note:
    Grant Accessibility + Screen Recording permissions to your terminal app in:
    System Settings → Privacy & Security → Accessibility
    System Settings → Privacy & Security → Screen Recording
"""

import subprocess
import platform
import time

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

_OS = platform.system()  # "Darwin" = macOS, "Windows", "Linux"


# ---------------------------------------------------------------------------
# App / system
# ---------------------------------------------------------------------------

def open_app(app_name: str) -> str:
    """Open an application by name (e.g. "Spotify", "Terminal")."""
    print(f"[CONTROL] Opening app: {app_name}")
    try:
        if _OS == "Darwin":
            subprocess.Popen(["open", "-a", app_name])
        elif _OS == "Windows":
            subprocess.Popen(["start", app_name], shell=True)
        else:
            subprocess.Popen([app_name])
        time.sleep(1)
        return f"Opened {app_name}."
    except Exception as e:
        return f"Could not open '{app_name}': {e}"


def run_command(command: str, capture_output: bool = False) -> str:
    """Execute a shell command, optionally returning its output."""
    print(f"[CONTROL] Running: {command}")
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=capture_output)
        if capture_output:
            return result.stdout.strip()
        return "Done."
    except Exception as e:
        return f"Command failed: {e}"


# ---------------------------------------------------------------------------
# Keyboard / mouse
# ---------------------------------------------------------------------------

def type_text(text: str) -> None:
    """
    Type text at the current cursor position using clipboard paste.
    Clipboard-based so all characters (URLs, symbols, etc.) work in any app.
    """
    print(f"[CONTROL] Typing: {text}")
    try:
        import pyperclip
        pyperclip.copy(text)
    except ImportError:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    time.sleep(0.05)
    paste_key = "command" if _OS == "Darwin" else "ctrl"
    pyautogui.hotkey(paste_key, "v")


def press_key(key: str) -> None:
    """Press a key or combo, e.g. "enter", "cmd+space", "ctrl+c"."""
    print(f"[CONTROL] Key: {key}")
    keys = key.lower().split("+")
    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)


def click_at(x: int, y: int, button: str = "left", clicks: int = 1) -> None:
    """Click at screen coordinates (x, y)."""
    print(f"[CONTROL] Click ({x}, {y}) {button} x{clicks}")
    pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=0.1)


def scroll(direction: str, amount: int = 3) -> None:
    """Scroll up or down at the current cursor position."""
    print(f"[CONTROL] Scroll {direction} x{amount}")
    units = amount if direction == "up" else -amount
    pyautogui.scroll(units)


# ---------------------------------------------------------------------------
# Spotify (macOS — AppleScript, no API key needed)
# ---------------------------------------------------------------------------

def spotify_search(query: str) -> str:
    """Open Spotify and search for a query (songs, artists, playlists)."""
    print(f"[CONTROL] Spotify search: {query}")
    encoded = query.replace(" ", "%20")
    subprocess.run(["open", f"spotify:search:{encoded}"])
    time.sleep(1.5)
    return f"Searching Spotify for: {query}"


def control_spotify(action: str) -> str:
    """Control Spotify playback. action = play|pause|next|previous|shuffle"""
    scripts = {
        "play":      'tell application "Spotify" to play',
        "pause":     'tell application "Spotify" to pause',
        "playpause": 'tell application "Spotify" to playpause',
        "next":      'tell application "Spotify" to next track',
        "previous":  'tell application "Spotify" to previous track',
        "shuffle":   'tell application "Spotify" to set shuffling to not shuffling',
    }
    script = scripts.get(action.lower())
    if not script:
        return f"Unknown action '{action}'. Use: play, pause, next, previous, shuffle."
    print(f"[CONTROL] Spotify: {action}")
    subprocess.run(["osascript", "-e", script], capture_output=True)
    return f"Spotify: {action}"


def spotify_set_volume(level: int) -> str:
    """Set Spotify volume 0–100."""
    level = max(0, min(100, level))
    subprocess.run(
        ["osascript", "-e", f'tell application "Spotify" to set sound volume to {level}'],
        capture_output=True,
    )
    return f"Volume: {level}"


def spotify_now_playing() -> str:
    """Return the currently playing Spotify track and artist."""
    script = '''
    tell application "Spotify"
        set t to name of current track
        set a to artist of current track
        return t & " by " & a
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    info = result.stdout.strip()
    return info if info else "Nothing playing."


# ---------------------------------------------------------------------------
# Vision — screen reading and element clicking
# ---------------------------------------------------------------------------

def read_screen() -> str:
    """Take a screenshot and describe what's on screen using the vision model."""
    print("[CONTROL] Reading screen...")
    try:
        from vision.screen_vision import vision_describe_screen
        return vision_describe_screen() or "Nothing detected."
    except Exception as e:
        return f"Screen read error: {e}"


def find_and_click(text: str) -> str:
    """Find a UI element on screen by description and click it."""
    print(f"[CONTROL] Finding '{text}'...")
    try:
        from vision.screen_vision import find_text_on_screen
        coords = find_text_on_screen(text)
        if coords:
            click_at(*coords)
            return f"Clicked '{text}'."
        return f"Could not find '{text}' on screen."
    except Exception as e:
        return f"find_and_click error: {e}"


def vision_describe(query: str = "") -> str:
    """Ask the vision model a specific question about the current screen."""
    print(f"[CONTROL] Vision query: {query}")
    try:
        from vision.screen_vision import vision_query
        return vision_query(query or "Describe what you see on this screen.")
    except Exception as e:
        return f"vision_describe error: {e}"
