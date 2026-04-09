"""
brain/agent.py — Central AI brain for Jarvis

Runs locally via Ollama — no token limits, no internet required, completely free.
Falls back to Groq if Ollama isn't running (useful on other machines).

Local models used (already downloaded):
  LLM:    qwen2.5:7b   — great instruction/JSON following, fast on Apple Silicon
  Vision: llava:latest  — handled in screen_vision.py

To change the local model, set OLLAMA_MODEL in jarvis/.env:
  OLLAMA_MODEL=llama3:latest
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Client setup — Ollama first (local, free), Groq fallback (remote, rate-limited)
# ---------------------------------------------------------------------------

def _init_client() -> tuple[OpenAI, str]:
    """
    Try to connect to local Ollama. Fall back to Groq if unavailable.
    Returns (client, model_name).
    """
    import urllib.request

    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",          # required field but unused by Ollama
        )
        print(f"[AGENT] Using local Ollama — model: {ollama_model}")
        return client, ollama_model
    except Exception:
        pass

    # Groq fallback — uses the same OpenAI-compatible API
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
        )
        groq_model = "llama-3.3-70b-versatile"
        print(f"[AGENT] Ollama not running — falling back to Groq ({groq_model})")
        return client, groq_model

    raise EnvironmentError(
        "[AGENT] No LLM available.\n"
        "  Option 1 (recommended): run `ollama serve` in a terminal.\n"
        "  Option 2 (fallback): add GROQ_API_KEY to jarvis/.env"
    )


_client, MODEL = _init_client()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_TOKENS        = 512
TEMPERATURE_TOOL  = 0.0   # Deterministic for JSON tool calls
TEMPERATURE_REPLY = 0.7   # Natural variation for conversation
MAX_HISTORY       = 30

SYSTEM_PROMPT = """You are Jarvis, an AI butler that controls a computer. You have two modes:

════════════════════════════════════════════════════════════
TOOL MODE  —  use whenever the user wants you to DO something
════════════════════════════════════════════════════════════
Output ONLY a JSON object. One per action. Nothing before or after it.
{"tool": "tool_name", "args": {"param": "value"}}

Multiple actions in one reply — just put them one after another:
{"tool": "open_app", "args": {"app_name": "Spotify"}}{"tool": "control_spotify", "args": {"action": "play"}}

Few-shot examples (copy this exact format):
  "open youtube and search metal music"
  → {"tool": "browser_youtube", "args": {"query": "metal music"}}

  "go to github"
  → {"tool": "browser_goto", "args": {"url": "https://github.com"}}

  "open spotify"
  → {"tool": "open_app", "args": {"app_name": "Spotify"}}

  "new tab"
  → {"tool": "browser_new_tab", "args": {}}

  "what tabs do I have"
  → {"tool": "browser_list_tabs", "args": {}}

  "search google for AI news"
  → {"tool": "browser_search", "args": {"query": "AI news"}}

  "pause spotify"
  → {"tool": "control_spotify", "args": {"action": "pause"}}

  "what's on screen"
  → {"tool": "read_screen", "args": {}}

Available tools — use these exact names:
  open_app(app_name)
  run_command(command, capture_output)
  type_text(text)
  press_key(key)
  click_at(x, y)
  scroll(direction, amount)
  spotify_search(query)
  control_spotify(action)          action = play|pause|next|previous|shuffle
  spotify_set_volume(level)
  spotify_now_playing()
  read_screen()
  find_and_click(text)
  vision_describe(query)
  browser_start()
  browser_goto(url)
  browser_search(query)
  browser_youtube(query)
  browser_new_tab(url)
  browser_close_tab()
  browser_list_tabs()
  browser_switch_tab(index)
  browser_find_tab(keyword)
  browser_close_tabs_like(keyword)
  browser_close_other_tabs()
  browser_back()
  browser_click(text)
  browser_click_button(name)
  browser_fill(label, text)
  browser_press(key)
  browser_scroll(direction, amount)
  browser_read_page()
  browser_get_links()
  browser_screenshot()
  browser_close()

Rules:
- ANY web task → always use browser_* tools (never open_app for websites)
- Spotify music → open_app then control_spotify or spotify_search
- When you need to see the page before clicking → browser_read_page first

════════════════════════════════════════════════════════════
CHAT MODE  —  use for conversation, questions, status updates
════════════════════════════════════════════════════════════
1-2 sentences. Dry wit. Never say URLs or file paths. No filler words."""

_TOOL_NAMES = {
    "open_app", "run_command", "type_text", "press_key", "click_at", "scroll",
    "spotify_search", "control_spotify", "spotify_set_volume", "spotify_now_playing",
    "read_screen", "find_and_click", "vision_describe",
    "open_url", "browser_navigate", "youtube_search",
    # Playwright browser tools
    "browser_start", "browser_close",
    "browser_goto", "browser_search", "browser_youtube",
    "browser_back", "browser_forward", "browser_reload",
    "browser_new_tab", "browser_close_tab", "browser_list_tabs", "browser_switch_tab",
    "browser_find_tab", "browser_close_tabs_like", "browser_close_other_tabs",
    "browser_click", "browser_click_button", "browser_fill", "browser_press", "browser_scroll",
    "browser_read_page", "browser_get_links", "browser_get_info", "browser_screenshot",
}

# Pre-built spoken confirmations — natural butler-style, no URLs or technical strings
_CONFIRMATIONS = {
    # System
    "open_app":              lambda a: f"Opening {a.get('app_name', 'that')}.",
    "run_command":           lambda _: "Done.",
    "type_text":             lambda _: "Typed.",
    "press_key":             lambda _: "Done.",
    "click_at":              lambda _: "Done.",
    "scroll":                lambda a: f"Scrolled {a.get('direction', '')}.",
    # Spotify
    "spotify_search":        lambda a: f"Searching Spotify for {a.get('query', 'that')}.",
    "control_spotify":       lambda a: f"Spotify — {a.get('action', 'done')}.",
    "spotify_set_volume":    lambda a: f"Volume at {a.get('level', '?')}.",
    "spotify_now_playing":   lambda _: "Let me check.",
    # Vision / screen
    "read_screen":           lambda _: "Let me have a look.",
    "find_and_click":        lambda a: f"Found {a.get('text', 'it')}.",
    "vision_describe":       lambda _: "Taking a look.",
    # Web (legacy tools)
    "open_url":              lambda _: "On it.",
    "browser_navigate":      lambda _: "Navigating.",
    "youtube_search":        lambda a: f"Searching YouTube for {a.get('query', 'that')}.",
    # Playwright browser — never say the URL
    "browser_start":         lambda _: "Browser is up.",
    "browser_close":         lambda _: "Browser closed.",
    "browser_goto":          lambda _: "On it.",
    "browser_search":        lambda a: f"Googling {a.get('query', 'that')}.",
    "browser_youtube":       lambda a: f"Pulling up {a.get('query', 'that')} on YouTube.",
    "browser_back":          lambda _: "Going back.",
    "browser_forward":       lambda _: "Going forward.",
    "browser_reload":        lambda _: "Refreshed.",
    "browser_new_tab":       lambda a: f"New tab — {_site_name(a.get('url', ''))}." if a.get('url') else "New tab.",
    "browser_close_tab":     lambda _: "Tab closed.",
    "browser_list_tabs":     lambda _: "Here are your tabs.",
    "browser_switch_tab":    lambda a: f"Switching to tab {a.get('index', '?')}.",
    "browser_find_tab":      lambda a: f"Switched to the {a.get('keyword', '')} tab.",
    "browser_close_tabs_like": lambda a: f"Closed the {a.get('keyword', '')} tabs.",
    "browser_close_other_tabs": lambda _: "Cleared everything else.",
    "browser_click":         lambda a: f"Clicked {a.get('text', 'it')}.",
    "browser_click_button":  lambda a: f"Clicked {a.get('name', 'it')}.",
    "browser_fill":          lambda _: "Done.",
    "browser_press":         lambda _: "Done.",
    "browser_scroll":        lambda a: f"Scrolled {a.get('direction', '')}.",
    "browser_read_page":     lambda _: "Reading the page.",
    "browser_get_links":     lambda _: "Got the links.",
    "browser_get_info":      lambda _: "Got it.",
    "browser_screenshot":    lambda _: "Taking a look.",
}

# After these tools, append a brief follow-up if it was the last action in the chain
_FOLLOWUPS = {
    "browser_youtube":       "Want me to play the first one?",
    "browser_search":        "Want me to open one of those?",
    "youtube_search":        "Want me to play the first result?",
    "spotify_search":        "Which one should I play?",
    "browser_new_tab":       "What should I pull up?",
    "open_app":              None,   # too generic — skip
}


def _site_name(url: str) -> str:
    """Extract a readable site name from a URL. e.g. 'https://github.com/x' → 'GitHub'"""
    import re
    m = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
    if not m:
        return url
    domain = m.group(1).split('.')[0].capitalize()
    return domain


def _clean_for_speech(text: str) -> str:
    """
    Strip anything that sounds bad when spoken aloud:
      - Markdown links: [label](url) → label
      - Bare URLs: https://... → (removed)
      - File paths: /Users/... or ./... → (removed)
      - Markdown bold/italic: **text** → text
      - Backtick code spans: `code` → code
    """
    import re
    # [label](url) → label
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # bare URLs (http/https/ftp)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'ftp://\S+', '', text)
    # file paths starting with / or ./
    text = re.sub(r'(?<!\w)[./]{1,2}/\S+', '', text)
    # markdown bold/italic
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # backtick code spans
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # collapse extra whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

_history: list[dict] = []

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_response(text: str) -> str:
    """
    Send user input to the LLM and return Jarvis's spoken reply.

    Normal tools (open_app, click, etc.) execute immediately and return a
    pre-built spoken confirmation with no extra API call.

    Perception tools (read_screen) feed their output back to the LLM so it
    can decide what to do next — a minimal see → think → act loop.
    """
    _history.append({"role": "user", "content": text})
    _trim_history()

    try:
        # Step 1: Ask the LLM what to do
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + _history,
            temperature=TEMPERATURE_TOOL,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()

        # Step 2: Extract all tool calls (model may chain several in one response)
        tool_calls = _parse_all_tool_calls(raw)

        if tool_calls:
            reply = _execute_and_reply(raw, tool_calls)
        else:
            # Plain conversational response — scrub before speaking
            reply = _clean_for_speech(raw)
            _history.append({"role": "assistant", "content": reply})

        print(f"[AGENT] Jarvis: {reply}")
        return reply

    except Exception as e:
        print(f"[AGENT] Error: {e}")
        return "I'm sorry, I encountered an error processing that request."


def _execute_and_reply(raw: str, tool_calls: list[tuple[str, dict]]) -> str:
    """
    Execute a batch of tool calls and build a spoken reply.

    If read_screen is among the calls its output feeds back to the LLM
    for one follow-up reasoning step (see → think → act).
    All other tools just confirm immediately without a second API call.
    """
    action_confirmations: list[str] = []
    screen_text: str | None = None

    for name, args in tool_calls:
        print(f"[AGENT] Tool: {name}({args})")
        result = _execute_tool(name, args)
        print(f"[AGENT] Result: {result}")

        if name in ("read_screen", "browser_read_page", "browser_screenshot",
                    "browser_list_tabs", "browser_get_links", "browser_get_info"):
            # These tools return content the LLM needs to reason about
            screen_text = (screen_text or "") + f"\n[{name}]\n{result}"
        else:
            fn = _CONFIRMATIONS.get(name)
            action_confirmations.append(fn(args) if fn else "Done.")

    if screen_text is not None:
        # Feed screen observation back and let LLM decide the next action
        _history.append({"role": "assistant", "content": raw})
        _history.append({
            "role": "user",
            "content": (
                f"[Screen content]\n{screen_text[:1200]}\n[End]\n"
                "Based on what you see, issue the next tool call or reply."
            ),
        })

        follow = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + _history,
            temperature=TEMPERATURE_TOOL,
            max_tokens=MAX_TOKENS,
        )
        follow_raw = follow.choices[0].message.content.strip()
        follow_tools = _parse_all_tool_calls(follow_raw)

        if follow_tools:
            for name, args in follow_tools:
                print(f"[AGENT] Follow-up tool: {name}({args})")
                _execute_tool(name, args)
                fn = _CONFIRMATIONS.get(name)
                action_confirmations.append(fn(args) if fn else "Done.")
            _history.append({"role": "assistant", "content": follow_raw})
            # Proactive follow-up after the last tool in the chain
            last_name, _ = follow_tools[-1]
            followup = _FOLLOWUPS.get(last_name)
            if followup:
                action_confirmations.append(followup)
        else:
            action_confirmations.append(_clean_for_speech(follow_raw))
            _history.append({"role": "assistant", "content": follow_raw})
    else:
        _history.append({"role": "assistant", "content": raw})
        # Proactive follow-up after the last tool in the chain
        last_name, _ = tool_calls[-1]
        followup = _FOLLOWUPS.get(last_name)
        if followup:
            action_confirmations.append(followup)

    return " ".join(action_confirmations) if action_confirmations else "Done."


def clear_history() -> None:
    """Reset conversation history."""
    _history.clear()
    print("[AGENT] Conversation history cleared.")


def get_history() -> list[dict]:
    return list(_history)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_all_tool_calls(text: str) -> list[tuple[str, dict]]:
    """
    Extract every valid tool call JSON object from the model response.

    The model sometimes chains multiple calls in one reply like:
        {"tool": "open_app", ...}{"tool": "spotify_search", ...}
    We walk the string bracket-by-bracket so each top-level object is
    parsed independently and invalid JSON in one object doesn't corrupt others.
    """
    results = []
    i = 0
    while i < len(text):
        if text[i] != '{':
            i += 1
            continue
        # Walk forward tracking brace depth to find matching '}'
        depth = 0
        j = i
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[i:j + 1])
                        name = data.get("tool")
                        args = data.get("args", {})
                        if name in _TOOL_NAMES and isinstance(args, dict):
                            results.append((name, args))
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
            j += 1
        else:
            i += 1
    return results


def _execute_tool(name: str, args: dict) -> str:
    """
    Dispatch a tool call to the right module.
    Checks computer_control first, then browser_control.
    """
    try:
        import actions.computer_control as cc
        import actions.browser_control  as bc
        module = cc if hasattr(cc, name) else bc if hasattr(bc, name) else None
        if module is None:
            return f"Unknown tool: '{name}'"
        result = getattr(module, name)(**args)
        return str(result) if result is not None else "Done."
    except Exception as e:
        return f"Tool error ({name}): {e}"


def _trim_history() -> None:
    while len(_history) > MAX_HISTORY:
        _history.pop(0)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Jarvis Agent Test — type 'exit' to quit ===\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        print(f"Jarvis: {get_response(user_input)}\n")
