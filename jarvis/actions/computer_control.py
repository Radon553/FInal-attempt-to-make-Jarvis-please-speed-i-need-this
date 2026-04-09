"""
actions/computer_control.py — Computer control module for Jarvis

Gives Jarvis the ability to interact with the operating system:
open apps, run commands, type text, move/click the mouse.

Install dependencies:
    pip install pyautogui keyboard

macOS security note:
    You must grant Accessibility and Screen Recording permissions to your
    terminal app (or Python) in:
    System Settings → Privacy & Security → Accessibility
    System Settings → Privacy & Security → Screen Recording
"""

import subprocess
import platform
import time

import pyautogui

# Prevent pyautogui from crashing the script on accidental mouse slam to corner
pyautogui.FAILSAFE = True

# Small delay between pyautogui actions to let the OS keep up
pyautogui.PAUSE = 0.05

_OS = platform.system()  # "Darwin" = macOS, "Windows", "Linux"


# ---------------------------------------------------------------------------
# Open an application
# ---------------------------------------------------------------------------

def open_app(app_name: str) -> bool:
    """
    Open an application by name.

    On macOS uses `open -a`, on Windows uses `start`, on Linux uses the
    app name directly as a command.

    Args:
        app_name: Application name as it appears on the system,
                  e.g. "Safari", "Calculator", "Spotify", "Terminal".

    Returns:
        True if the launch command succeeded, False otherwise.

    Example:
        open_app("Safari")
        open_app("Spotify")
    """
    print(f"[CONTROL] Opening app: {app_name}")
    try:
        if _OS == "Darwin":
            subprocess.Popen(["open", "-a", app_name])
        elif _OS == "Windows":
            subprocess.Popen(["start", app_name], shell=True)
        else:
            subprocess.Popen([app_name])
        time.sleep(1)  # Give the app a moment to launch
        return True
    except Exception as e:
        print(f"[CONTROL] Failed to open '{app_name}': {e}")
        return False


# ---------------------------------------------------------------------------
# Run a terminal command
# ---------------------------------------------------------------------------

def run_command(command: str, capture_output: bool = False) -> str | None:
    """
    Execute a shell command and optionally return its output.

    Args:
        command:        The shell command string to run, e.g. "ls -la".
        capture_output: If True, returns stdout as a string instead of
                        printing it to the terminal.

    Returns:
        Command output string if capture_output=True, else None.

    Example:
        run_command("open /Applications")
        output = run_command("date", capture_output=True)
    """
    print(f"[CONTROL] Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=capture_output,
        )
        if capture_output:
            output = result.stdout.strip()
            print(f"[CONTROL] Output: {output}")
            return output
        return None
    except Exception as e:
        print(f"[CONTROL] Command failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Type text
# ---------------------------------------------------------------------------

def type_text(text: str, interval: float = 0.05) -> None:
    """
    Type text at the current cursor position using clipboard paste.

    WHY clipboard instead of pyautogui.typewrite():
      typewrite() simulates individual key presses and silently drops
      special characters (/, :, @, etc.) in most macOS apps — browser
      address bars, Electron apps, Spotify, etc. Clipboard paste is
      instant and works with ANY character in ANY app.

    Click the target field first if it isn't already focused.

    Args:
        text: The string to type (supports all Unicode, URLs, symbols).

    Example:
        press_key("cmd+l")          # focus browser address bar
        type_text("https://youtube.com")
        press_key("enter")
    """
    print(f"[CONTROL] Typing (clipboard): {text}")

    # Copy text to clipboard — try pyperclip first, fall back to pbcopy (macOS)
    try:
        import pyperclip
        pyperclip.copy(text)
    except ImportError:
        # pbcopy is built into macOS, no extra install needed
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)

    time.sleep(0.05)

    # Paste with the OS-appropriate shortcut
    paste_key = "command" if _OS == "Darwin" else "ctrl"
    pyautogui.hotkey(paste_key, "v")


def press_key(key: str) -> None:
    """
    Press a single key or key combination.

    Args:
        key: Key name or combo string, e.g. "enter", "ctrl+c", "cmd+space".
             Uses pyautogui hotkey notation — modifiers joined with "+".

    Example:
        press_key("enter")
        press_key("cmd+space")   # Spotlight on macOS
        press_key("ctrl+c")      # Copy
    """
    print(f"[CONTROL] Pressing key: {key}")
    keys = key.lower().split("+")
    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)


# ---------------------------------------------------------------------------
# Mouse control
# ---------------------------------------------------------------------------

def move_mouse(x: int, y: int, duration: float = 0.3) -> None:
    """
    Smoothly move the mouse cursor to screen coordinates (x, y).

    Args:
        x:        Horizontal screen coordinate in pixels.
        y:        Vertical screen coordinate in pixels.
        duration: Seconds the movement takes. Slower = more human-like.

    Example:
        move_mouse(960, 540)   # Move to centre of a 1920×1080 screen
    """
    print(f"[CONTROL] Moving mouse to ({x}, {y})")
    pyautogui.moveTo(x, y, duration=duration)


def click_at(x: int, y: int, button: str = "left", clicks: int = 1) -> None:
    """
    Move the mouse to (x, y) and click.

    Args:
        x:       Horizontal screen coordinate in pixels.
        y:       Vertical screen coordinate in pixels.
        button:  "left", "right", or "middle".
        clicks:  Number of clicks (2 for double-click).

    Example:
        click_at(200, 300)              # Single left-click
        click_at(200, 300, clicks=2)    # Double-click
        click_at(200, 300, button="right")  # Right-click
    """
    print(f"[CONTROL] Clicking at ({x}, {y}) — {button} x{clicks}")
    pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=0.1)


def scroll(direction: str, amount: int = 3) -> None:
    """
    Scroll the mouse wheel at the current cursor position.

    Args:
        direction: "up" or "down".
        amount:    Number of scroll units. Higher = more scrolling.

    Example:
        scroll("down", 5)
        scroll("up", 2)
    """
    print(f"[CONTROL] Scrolling {direction} by {amount}")
    units = amount if direction == "up" else -amount
    pyautogui.scroll(units)


def get_mouse_position() -> tuple[int, int]:
    """
    Return the current mouse cursor position as (x, y).

    Useful for finding coordinates to use with click_at or move_mouse.

    Returns:
        Tuple of (x, y) screen coordinates.
    """
    pos = pyautogui.position()
    print(f"[CONTROL] Mouse is at ({pos.x}, {pos.y})")
    return pos.x, pos.y


# ---------------------------------------------------------------------------
# Spotify control (macOS — uses AppleScript, no API key needed)
# ---------------------------------------------------------------------------

def spotify_search(query: str) -> str:
    """
    Open Spotify and navigate to a search for the given query.
    Works for playlists, songs, artists, albums.

    Args:
        query: Search term, e.g. "lofi hip hop" or "Drake".

    Example:
        spotify_search("lofi hip hop playlist")
    """
    print(f"[CONTROL] Spotify search: {query}")
    encoded = query.replace(" ", "%20")
    # Spotify URI scheme — opens the search results page inside the app
    subprocess.run(["open", f"spotify:search:{encoded}"])
    time.sleep(1.5)
    return f"Opened Spotify search for: {query}"


def control_spotify(action: str) -> str:
    """
    Control Spotify playback via AppleScript.

    Args:
        action: One of — play, pause, playpause, next, previous, shuffle.

    Example:
        control_spotify("play")
        control_spotify("next")
        control_spotify("shuffle")
    """
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
    """
    Set Spotify's volume (0–100).

    Args:
        level: Integer from 0 (mute) to 100 (full volume).

    Example:
        spotify_set_volume(50)
    """
    level = max(0, min(100, level))
    script = f'tell application "Spotify" to set sound volume to {level}'
    subprocess.run(["osascript", "-e", script], capture_output=True)
    print(f"[CONTROL] Spotify volume: {level}")
    return f"Spotify volume set to {level}."


def spotify_now_playing() -> str:
    """
    Return the currently playing track and artist from Spotify.

    Example:
        info = spotify_now_playing()
    """
    script = '''
    tell application "Spotify"
        set t to name of current track
        set a to artist of current track
        return t & " by " & a
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    info = result.stdout.strip()
    print(f"[CONTROL] Now playing: {info}")
    return info if info else "Could not retrieve current track."


# ---------------------------------------------------------------------------
# Browser / web navigation
# ---------------------------------------------------------------------------

def open_url(url: str) -> str:
    """
    Open a URL in the default browser.

    The simplest way to navigate anywhere on the web. On macOS, `open`
    respects the user's default browser (Chrome, Safari, Firefox, etc.).

    Args:
        url: Full URL including scheme, e.g. "https://www.youtube.com".

    Returns:
        Confirmation string.

    Example:
        open_url("https://www.youtube.com")
        open_url("https://www.youtube.com/results?search_query=metal+music")
    """
    print(f"[CONTROL] Opening URL: {url}")
    if _OS == "Darwin":
        subprocess.Popen(["open", url])
    elif _OS == "Windows":
        subprocess.Popen(["start", url], shell=True)
    else:
        subprocess.Popen(["xdg-open", url])
    time.sleep(2)  # Give browser time to load
    return f"Opened: {url}"


def browser_navigate(url: str) -> str:
    """
    Navigate the currently open browser window to a URL.

    Uses Cmd+L (macOS) / Ctrl+L (Windows/Linux) to focus the address bar,
    pastes the URL via clipboard (reliable with all characters), then presses Enter.

    Use open_url() if no browser is open yet.
    Use this when a browser is already open and you want to go somewhere new.

    Args:
        url: Full URL, e.g. "https://www.youtube.com/watch?v=abc123"

    Example:
        browser_navigate("https://www.youtube.com")
    """
    print(f"[CONTROL] Browser navigate: {url}")

    # Focus the address bar
    focus_key = "command" if _OS == "Darwin" else "ctrl"
    pyautogui.hotkey(focus_key, "l")
    time.sleep(0.3)

    # Select all existing content and replace with the URL
    pyautogui.hotkey(focus_key, "a")
    time.sleep(0.1)

    # Paste URL via clipboard (typewrite fails on : and / in URLs)
    try:
        import pyperclip
        pyperclip.copy(url)
    except ImportError:
        subprocess.run(["pbcopy"], input=url.encode("utf-8"), check=True)
    pyautogui.hotkey(focus_key, "v")
    time.sleep(0.2)

    pyautogui.press("enter")
    time.sleep(1.5)  # Give page a moment to start loading
    return f"Navigated to: {url}"


def youtube_search(query: str) -> str:
    """
    Open YouTube and search for a video, channel, or playlist.

    Directly navigates to YouTube search results — no need to interact
    with the YouTube UI at all.

    Args:
        query: Search terms, e.g. "metal music playlist", "lofi hip hop".

    Returns:
        Confirmation string.

    Example:
        youtube_search("metal playlist 2024")
        youtube_search("lofi hip hop study music")
    """
    print(f"[CONTROL] YouTube search: {query}")
    encoded = query.replace(" ", "+")
    url = f"https://www.youtube.com/results?search_query={encoded}"
    return open_url(url)


# ---------------------------------------------------------------------------
# Vision tools — wrappers around vision/screen_vision.py
# The agent dispatches to these the same way it dispatches to any other tool.
# ---------------------------------------------------------------------------

def read_screen() -> str:
    """
    Take a screenshot and describe what's on screen using the vision model.

    Uses Groq vision (free, already configured) or Gemini Flash (free backup).
    Works on dark themes, graphical UIs, Spotify, icons — anything visible.

    Returns:
        Natural language description of the current screen state.

    Example:
        text = read_screen()
        # → "Spotify is open. The sidebar shows: Home, Search, Your Library,
        #    and playlists: My Metal Mix, Lofi Chill, Workout..."
    """
    print("[CONTROL] Reading screen via vision model...")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vision.screen_vision import vision_describe_screen
        result = vision_describe_screen()
        print(f"[CONTROL] Screen: {result[:150]}...")
        return result or "Nothing detected on screen."
    except Exception as e:
        return f"Screen read error: {e}"


def find_and_click(text: str) -> str:
    """
    Find a UI element on screen by description and click it.

    Uses Groq vision (free) to visually locate the element — works on dark
    themes, Spotify playlists, icons, custom fonts, anything visible.

    Args:
        text: Description of what to click, e.g. "My Metal Mix playlist",
              "Play button", "Search bar", "Liked Songs".

    Returns:
        Success/failure message.

    Example:
        find_and_click("My Metal Mix playlist in the sidebar")
        find_and_click("green Play button")
    """
    print(f"[CONTROL] Vision-searching for '{text}'...")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vision.screen_vision import find_text_on_screen
        coords = find_text_on_screen(text)
        if coords:
            click_at(*coords)
            return f"Clicked '{text}' at {coords}."
        return f"Could not find '{text}' on screen."
    except Exception as e:
        return f"find_and_click error: {e}"


def vision_describe(query: str = "") -> str:
    """
    Ask the vision model a specific question about the current screen.

    More focused than read_screen() when you want a targeted answer:
    "What playlists are in the sidebar?", "Is Spotify playing?", etc.

    Args:
        query: The question to ask. If empty, describes everything visible.

    Returns:
        Vision model's answer about the screen.

    Example:
        vision_describe("What playlists are visible in the Spotify sidebar?")
        vision_describe("Is there a play button visible?")
    """
    print(f"[CONTROL] Vision query: {query}")
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from vision.screen_vision import vision_query
        prompt = query if query else "Describe what you see on this screen in detail."
        return vision_query(prompt)
    except Exception as e:
        return f"vision_describe error: {e}"


# ---------------------------------------------------------------------------
# Convenience: screenshot
# ---------------------------------------------------------------------------

def take_screenshot(filepath: str = "screenshot.png") -> str:
    """
    Capture the entire screen and save it to a file.

    Args:
        filepath: Destination path for the PNG file.

    Returns:
        The filepath where the screenshot was saved.

    Example:
        path = take_screenshot("vision/last_frame.png")
    """
    print(f"[CONTROL] Taking screenshot → {filepath}")
    pyautogui.screenshot(filepath)
    return filepath


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Computer Control Test ===")
    print("Current mouse position:", get_mouse_position())

    # Safe tests — only reads/prints, no clicks or keypresses
    output = run_command("echo 'Jarvis computer control is online'", capture_output=True)
    print("Command output:", output)

    screenshot_path = take_screenshot("/tmp/jarvis_test_screenshot.png")
    print("Screenshot saved to:", screenshot_path)
