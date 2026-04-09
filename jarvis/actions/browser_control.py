"""
actions/browser_control.py — Playwright controls your real Chrome browser

Jarvis attaches to the Chrome you already use (via Chrome DevTools Protocol):
  - No separate browser window — Jarvis drives YOUR Chrome
  - All your tabs, logins, bookmarks, extensions stay intact
  - Open / close / switch tabs
  - Navigate to any URL
  - Click elements by text or ARIA role (DOM-aware, no pixel-guessing)
  - Fill forms, press keys, scroll pages
  - Read page text for AI reasoning
  - Take browser screenshots → send to vision model

How it works:
  On first use, Jarvis launches Chrome with --remote-debugging-port=9222
  and Playwright connects to it over CDP. Chrome stays open after Jarvis
  disconnects — use browser_close() only to disconnect Playwright, not to
  close your browser.

Install (done once):
    pip install playwright
    playwright install chromium   # only needed as fallback
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# CDP port — Chrome must be started with --remote-debugging-port=9222
_CDP_URL = "http://localhost:9222"

# macOS Chrome paths (tried in order)
_CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]

# ---------------------------------------------------------------------------
# Global browser state — one Playwright instance, multiple tabs
# ---------------------------------------------------------------------------

_playwright_handle = None   # sync_playwright() result
_browser           = None   # Browser object (CDP-connected or launched)
_context           = None   # BrowserContext
_pages: list       = []     # All open Page objects (tabs)
_current_idx: int  = 0      # Which tab is active
_cdp_mode: bool    = False  # True when connected via CDP to real Chrome


def _page():
    """Return the currently active tab, auto-connecting if needed."""
    _ensure_browser()
    # Refresh page list in case tabs were opened/closed outside Jarvis
    if _cdp_mode and _context:
        live = list(_context.pages)
        if live != _pages:
            _sync_pages(live)
    return _pages[_current_idx]


def _sync_pages(live: list) -> None:
    """Keep _pages in sync with the browser's actual tab list."""
    global _pages, _current_idx
    _pages = live
    _current_idx = min(_current_idx, max(0, len(_pages) - 1))


def _ensure_browser() -> None:
    """
    Connect to (or launch) a Chrome window Jarvis controls.

    Priority:
      1. Already connected — do nothing.
      2. Chrome already running with --remote-debugging-port=9222 — attach via CDP.
      3. Chrome installed — launch a Jarvis-owned Chrome window with a separate
         profile (bypasses macOS singleton) + CDP. This IS real Chrome, just a
         fresh profile so Jarvis has full control.
      4. Fallback — Playwright's bundled Chromium.
    """
    global _playwright_handle, _browser, _context, _pages, _current_idx, _cdp_mode

    # Already connected and healthy?
    if _browser and _browser.is_connected():
        return

    # Stale connection — reset
    _playwright_handle = None
    _browser = _context = None
    _pages = []

    from playwright.sync_api import sync_playwright
    _playwright_handle = sync_playwright().start()

    # --- Attempt 1: attach to Chrome already running with debug port ---
    try:
        _browser = _playwright_handle.chromium.connect_over_cdp(_CDP_URL)
        _context = _browser.contexts[0]
        _context.set_default_timeout(10_000)
        _pages = list(_context.pages) or [_context.new_page()]
        _current_idx = max(0, len(_pages) - 1)
        _cdp_mode = True
        print(f"[BROWSER] Attached to existing Chrome ({len(_pages)} tab(s)).")
        return
    except Exception:
        pass  # Chrome not running with debug port

    # --- Attempt 2: launch a Jarvis-owned Chrome window ---
    # macOS Chrome is a singleton — relaunching it just opens a tab in the
    # existing instance (without the debug port). Using --user-data-dir forces
    # a genuinely new, independent Chrome process that Jarvis fully controls.
    chrome = next((p for p in _CHROME_PATHS if Path(p).exists()), None)
    if chrome:
        profile_dir = Path.home() / ".jarvis_chrome_profile"
        profile_dir.mkdir(exist_ok=True)
        try:
            print("[BROWSER] Launching Jarvis Chrome window...")
            proc = subprocess.Popen(
                [
                    chrome,
                    f"--remote-debugging-port=9222",
                    f"--user-data-dir={profile_dir}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--start-maximized",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,   # capture errors so we can log them
            )

            # Poll until CDP port is ready (up to 8 seconds)
            for attempt in range(16):
                time.sleep(0.5)
                try:
                    _browser = _playwright_handle.chromium.connect_over_cdp(_CDP_URL)
                    break
                except Exception:
                    # Check if the process crashed
                    if proc.poll() is not None:
                        err = proc.stderr.read().decode(errors="replace")[:300]
                        print(f"[BROWSER] Chrome exited early: {err}")
                        break

            if _browser and _browser.is_connected():
                _context = _browser.contexts[0]
                _context.set_default_timeout(10_000)
                _pages = list(_context.pages) or [_context.new_page()]
                _current_idx = 0
                _cdp_mode = True
                print("[BROWSER] Jarvis Chrome window ready.")
                return

            print("[BROWSER] Chrome launched but CDP connect failed — falling back.")

        except Exception as e:
            print(f"[BROWSER] Chrome launch error: {e}")

    # --- Fallback: Playwright's bundled Chromium ---
    print("[BROWSER] Falling back to Playwright Chromium.")
    try:
        _browser = _playwright_handle.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--start-maximized"],
        )
        _context = _browser.new_context(viewport=None, no_viewport=True)
        _context.set_default_timeout(10_000)
        _pages = [_context.new_page()]
        _current_idx = 0
        _cdp_mode = False
        print("[BROWSER] Playwright Chromium launched.")
    except Exception as e:
        print(f"[BROWSER] FATAL: Cannot launch any browser: {e}")
        raise


# ---------------------------------------------------------------------------
# Browser lifecycle
# ---------------------------------------------------------------------------

def browser_start() -> str:
    """
    Connect to (or launch) your Chrome browser.
    Called automatically by other browser tools if not already connected.
    """
    _ensure_browser()
    mode = "your Chrome" if _cdp_mode else "Playwright Chromium"
    return f"Connected to {mode}."


def browser_close() -> str:
    """
    Disconnect Jarvis from Chrome.
    Does NOT close Chrome itself — your tabs stay open.
    """
    global _playwright_handle, _browser, _context, _pages, _current_idx
    try:
        if _playwright_handle:
            _playwright_handle.stop()
    except Exception:
        pass
    _playwright_handle = _browser = _context = None
    _pages = []
    _current_idx = 0
    return "Disconnected from browser."


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def browser_goto(url: str) -> str:
    """
    Navigate the current tab to any URL.

    Automatically adds https:// if the scheme is missing.

    Args:
        url: e.g. "https://youtube.com" or just "youtube.com"

    Example:
        browser_goto("https://www.youtube.com")
        browser_goto("reddit.com/r/programming")
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    print(f"[BROWSER] → {url}")
    _page().goto(url, wait_until="domcontentloaded", timeout=15_000)
    return f"Opened: {_page().title()} — {url}"


def browser_search(query: str) -> str:
    """
    Search Google for a query and open the results page.

    Args:
        query: What to search for, e.g. "best Python tutorials"

    Example:
        browser_search("latest AI news")
    """
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    return browser_goto(url)


def browser_youtube(query: str) -> str:
    """
    Search YouTube for a video, playlist, or channel.

    Args:
        query: Search terms, e.g. "metal music playlist 2024"

    Example:
        browser_youtube("lofi hip hop study")
    """
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    return browser_goto(url)


def browser_back() -> str:
    """Go back one page in the current tab's history."""
    _page().go_back(wait_until="domcontentloaded")
    return f"Back → {_page().title()}"


def browser_forward() -> str:
    """Go forward one page in the current tab's history."""
    _page().go_forward(wait_until="domcontentloaded")
    return f"Forward → {_page().title()}"


def browser_reload() -> str:
    """Reload the current page."""
    _page().reload(wait_until="domcontentloaded")
    return "Page reloaded."


# ---------------------------------------------------------------------------
# Tab management
# ---------------------------------------------------------------------------

def browser_new_tab(url: str = "") -> str:
    """
    Open a new tab in the Jarvis browser, optionally navigating to a URL.

    Args:
        url: Optional URL to open immediately in the new tab.

    Example:
        browser_new_tab("https://www.github.com")
        browser_new_tab()   # blank tab
    """
    global _current_idx
    _ensure_browser()
    page = _context.new_page()
    _pages.append(page)
    _current_idx = len(_pages) - 1
    print(f"[BROWSER] New tab #{_current_idx + 1}")
    if url:
        browser_goto(url)
        return f"New tab #{_current_idx + 1}: {url}"
    return f"Blank tab #{_current_idx + 1} opened."


def browser_close_tab() -> str:
    """
    Close the current tab and switch to the previous one.

    If only one tab is open, use browser_close() instead.
    """
    global _current_idx
    if len(_pages) <= 1:
        return "Only one tab open. Use browser_close() to close the whole browser."
    page = _pages.pop(_current_idx)
    page.close()
    _current_idx = max(0, _current_idx - 1)
    _pages[_current_idx].bring_to_front()
    return f"Tab closed. Now on tab {_current_idx + 1}: {_pages[_current_idx].title()}"


def browser_list_tabs() -> str:
    """
    List all open tabs with their index, title, and URL.
    The active tab is marked with →.

    Example response:
        → [1] YouTube — https://youtube.com
          [2] GitHub  — https://github.com
    """
    _ensure_browser()
    # In CDP mode, sync with the real browser's current tab list
    if _cdp_mode and _context:
        _sync_pages(list(_context.pages))
    lines = []
    for i, p in enumerate(_pages):
        marker = "→" if i == _current_idx else " "
        title  = (p.title() or "(no title)")[:50]
        url    = (p.url or "(blank)")[:80]
        lines.append(f"{marker} [{i + 1}] {title}  —  {url}")
    return "\n".join(lines) or "No tabs open."


def browser_switch_tab(index: int) -> str:
    """
    Switch to a tab by its number (1-based, as shown in browser_list_tabs).

    Args:
        index: Tab number, e.g. 2 for the second tab.

    Example:
        browser_switch_tab(2)
    """
    global _current_idx
    _ensure_browser()
    idx = index - 1
    if not (0 <= idx < len(_pages)):
        return f"Tab {index} doesn't exist. There are {len(_pages)} tabs open."
    _current_idx = idx
    _pages[_current_idx].bring_to_front()


def browser_find_tab(keyword: str) -> str:
    """
    Find and switch to a tab whose title or URL contains the keyword.

    Case-insensitive search. Switches to the first match.

    Args:
        keyword: Part of the page title or URL, e.g. "youtube", "github", "reddit".

    Example:
        browser_find_tab("youtube")   # switches to the YouTube tab
        browser_find_tab("reddit")
    """
    global _current_idx
    _ensure_browser()
    kw = keyword.lower()
    for i, p in enumerate(_pages):
        if kw in (p.title() or "").lower() or kw in (p.url or "").lower():
            _current_idx = i
            p.bring_to_front()
            return f"Switched to tab {i + 1}: {p.title()}"
    return f"No tab found matching '{keyword}'. Open tabs: {browser_list_tabs()}"


def browser_close_tabs_like(keyword: str) -> str:
    """
    Close all tabs whose title or URL contains the keyword.

    Keeps the current tab safe — if it matches, keeps it and closes the others.

    Args:
        keyword: Word to match against tab titles/URLs, e.g. "reddit", "gmail".

    Example:
        browser_close_tabs_like("reddit")   # close all Reddit tabs
        browser_close_tabs_like("youtube")  # close all YouTube tabs
    """
    global _pages, _current_idx
    _ensure_browser()
    kw = keyword.lower()
    to_close = [
        (i, p) for i, p in enumerate(_pages)
        if (kw in (p.title() or "").lower() or kw in (p.url or "").lower())
        and i != _current_idx   # never auto-close the active tab
    ]
    if not to_close:
        return f"No tabs matching '{keyword}' — nothing closed."
    for _, p in reversed(to_close):
        p.close()
    indices = {i for i, _ in to_close}
    _pages = [p for i, p in enumerate(_pages) if i not in indices]
    _current_idx = min(_current_idx, len(_pages) - 1)
    return f"Closed {len(to_close)} tab(s) matching '{keyword}'. {len(_pages)} tab(s) remaining."


def browser_close_other_tabs() -> str:
    """
    Close every tab except the one you're currently on.

    Example:
        browser_close_other_tabs()   # keep only the active tab
    """
    global _pages, _current_idx
    _ensure_browser()
    if len(_pages) <= 1:
        return "Only one tab open — nothing to close."
    current_page = _pages[_current_idx]
    count = len(_pages) - 1
    for i, p in enumerate(_pages):
        if i != _current_idx:
            p.close()
    _pages = [current_page]
    _current_idx = 0
    return f"Closed {count} tab(s). Just {current_page.title()} left."


# ---------------------------------------------------------------------------
# Page interaction — DOM-aware (no pixel-guessing needed for web pages)
# ---------------------------------------------------------------------------

def browser_click(text: str) -> str:
    """
    Click the first visible element on the page that contains this text.

    Playwright finds it in the DOM — works even if text is inside a link,
    button, span, or div. Much more reliable than screen coordinates for web.

    Args:
        text: Visible text of the element to click, e.g. "Sign in", "Subscribe".

    Example:
        browser_click("Subscribe")
        browser_click("Watch later")
    """
    print(f"[BROWSER] Clicking text: '{text}'")
    try:
        _page().get_by_text(text, exact=False).first.click(timeout=5_000)
        return f"Clicked '{text}'."
    except Exception as e:
        # Fallback: try as a link
        try:
            _page().get_by_role("link", name=text).first.click(timeout=3_000)
            return f"Clicked link '{text}'."
        except Exception:
            return f"Could not find '{text}' on page. Try browser_screenshot() to see what's visible."


def browser_click_button(name: str) -> str:
    """
    Click a button by its label text.

    Specifically targets <button> elements and role=button elements.

    Args:
        name: Button label, e.g. "Play", "Submit", "Search".

    Example:
        browser_click_button("Play all")
        browser_click_button("Accept cookies")
    """
    print(f"[BROWSER] Clicking button: '{name}'")
    try:
        _page().get_by_role("button", name=name).first.click(timeout=5_000)
        return f"Clicked button '{name}'."
    except Exception as e:
        return f"Button '{name}' not found: {e}"


def browser_fill(label: str, text: str) -> str:
    """
    Find an input field by its label or placeholder text and fill it.

    Args:
        label: The field's label or placeholder, e.g. "Search", "Email", "Password".
        text:  The value to type into the field.

    Example:
        browser_fill("Search", "metal music")
        browser_fill("username", "myuser@email.com")
    """
    print(f"[BROWSER] Filling '{label}' → '{text}'")
    p = _page()
    for attempt in [
        lambda: p.get_by_label(label, exact=False).first.fill(text),
        lambda: p.get_by_placeholder(label, exact=False).first.fill(text),
        lambda: p.get_by_role("searchbox").first.fill(text),
        lambda: p.get_by_role("textbox").first.fill(text),
    ]:
        try:
            attempt()
            return f"Filled '{label}' with '{text}'."
        except Exception:
            continue
    return f"Could not find input field '{label}' on this page."


def browser_press(key: str) -> str:
    """
    Press a keyboard key in the browser.

    Args:
        key: Key name — "Enter", "Escape", "Tab", "ArrowDown", etc.
             Playwright uses standard key names (capital first letter).

    Example:
        browser_press("Enter")     # submit a search
        browser_press("Escape")    # close a modal
    """
    _page().keyboard.press(key)
    return f"Pressed {key}."


def browser_scroll(direction: str, amount: int = 3) -> str:
    """
    Scroll the current page up or down.

    Args:
        direction: "up" or "down".
        amount:    Scroll intensity (1 = small, 5 = large). Default 3.

    Example:
        browser_scroll("down", 5)
        browser_scroll("up", 2)
    """
    delta = -400 * amount if direction.lower() == "up" else 400 * amount
    _page().mouse.wheel(0, delta)
    return f"Scrolled {direction}."


# ---------------------------------------------------------------------------
# Reading page content
# ---------------------------------------------------------------------------

def browser_read_page() -> str:
    """
    Return the visible text content of the current web page.

    Strips HTML — gives the AI the same text a human would read.
    Truncated to 3000 characters to fit in the agent's context window.

    Use this to understand what's on a page before deciding what to click.

    Example:
        text = browser_read_page()
        # → "YouTube  Search  Metal Music Playlist 2024 by MegaChannel..."
    """
    print("[BROWSER] Reading page text...")
    try:
        text = _page().inner_text("body")
        if len(text) > 3000:
            text = text[:3000] + "\n…[truncated — page has more content]"
        return text
    except Exception as e:
        return f"Could not read page text: {e}"


def browser_get_links() -> str:
    """
    Return all visible links on the current page (text + href).
    Useful for picking which link to follow next.

    Returns up to 30 links to keep the output manageable.
    """
    try:
        links = _page().eval_on_selector_all(
            "a[href]",
            """els => els
                .filter(e => e.offsetParent !== null)   // visible only
                .slice(0, 30)
                .map(e => e.innerText.trim() + '  →  ' + e.href)
                .filter(s => s.length > 5)"""
        )
        return "\n".join(links) or "No links found."
    except Exception as e:
        return f"Could not get links: {e}"


def browser_get_info() -> str:
    """Return the current tab's title and URL in one call."""
    _ensure_browser()
    return f"Title: {_page().title()}\nURL: {_page().url}"


# ---------------------------------------------------------------------------
# Vision-powered screenshot analysis
# ---------------------------------------------------------------------------

def browser_screenshot() -> str:
    """
    Take a screenshot of the browser and describe what's visible using the vision model.

    Use this when the page has complex visual content (thumbnails, carousels,
    video players) that browser_read_page() can't fully capture.

    Returns a natural language description of the page.
    """
    print("[BROWSER] Screenshot → vision analysis...")
    try:
        png_bytes = _page().screenshot()
        from vision.screen_vision import _image_to_base64, _ask_vision_model
        from PIL import Image
        image = Image.open(io.BytesIO(png_bytes))
        b64   = _image_to_base64(image)
        prompt = (
            "Describe this web page screenshot: what site is open, "
            "what's the main content area showing, what buttons or links are visible, "
            "and any important text or media on screen. Be specific about video titles, "
            "playlist names, or any interactive elements."
        )
        return _ask_vision_model(b64, prompt)
    except Exception as e:
        return f"Browser screenshot/vision failed: {e}"


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Browser Control Test ===\n")
    browser_start()
    print(browser_goto("https://www.youtube.com"))
    print("Tabs:", browser_list_tabs())
    print("\nPage text (first 300 chars):")
    print(browser_read_page()[:300])
    input("\nBrowser open. Press Enter to close...")
    browser_close()
