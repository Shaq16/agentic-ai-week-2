"""TOPIC 4 -- Memory, in two flavours.

SHORT-TERM  : the checkpointer. Scoped to one `thread_id`. Holds the whole BrewState
              (messages, cart, ...) so a conversation can be resumed. Two threads
              never see each other's messages or carts.
LONG-TERM   : the store. Scoped to a *namespace*, not a thread. A fact written while
              chatting in thread A is readable from brand-new thread B.

Both are backed by SQLite here, so they also survive an app restart.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from brewbot.domain import MENU, MILKS
from brewbot.state import Profile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DB = DATA_DIR / "short_term.sqlite"
STORE_DB = DATA_DIR / "long_term.sqlite"

# Namespaces inside the long-term store.
PROFILE_NS = ("brewbot", "profiles")
THREADS_NS = ("brewbot", "threads")


def _connect(path: Path, autocommit: bool = False) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Streamlit reruns callbacks on a worker thread, so the connection must not be
    # pinned to the thread that created it. SqliteStore issues its own BEGIN, so it
    # needs autocommit; SqliteSaver manages transactions the default way.
    return sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None if autocommit else "",
    )


def build_checkpointer() -> SqliteSaver:
    """Short-term memory. Keyed by thread_id at call time via config."""
    saver = SqliteSaver(_connect(CHECKPOINT_DB))
    saver.setup()
    return saver


def build_store() -> SqliteStore:
    """Long-term memory. Keyed by namespace, so it outlives any single thread."""
    store = SqliteStore(_connect(STORE_DB, autocommit=True))
    store.setup()
    return store


# --------------------------------------------------------------------------- profile


def load_profile(store: BaseStore, user_id: str) -> Profile:
    item = store.get(PROFILE_NS, user_id)
    return dict(item.value) if item else {}  # type: ignore[return-value]


def save_profile(store: BaseStore, user_id: str, profile: Profile) -> None:
    store.put(PROFILE_NS, user_id, dict(profile))


def forget_profile(store: BaseStore, user_id: str) -> None:
    store.delete(PROFILE_NS, user_id)


_NAME_PATTERNS = [
    r"\bmy name is ([a-z][a-z'\-]{1,20})",
    r"\bi am ([a-z][a-z'\-]{1,20})\b",
    r"\bi'm ([a-z][a-z'\-]{1,20})\b",
    r"\bcall me ([a-z][a-z'\-]{1,20})",
    r"\bthis is ([a-z][a-z'\-]{1,20}) (?:here|again)",
    r"\bit's ([a-z][a-z'\-]{1,20}) (?:here|again)",
]

# Words that follow "I'm ..." but are clearly not names.
_NOT_NAMES = {
    "back", "here", "good", "fine", "ok", "okay", "hungry", "thirsty", "late",
    "sorry", "just", "not", "so", "really", "in", "on", "a", "an", "the", "going",
    "looking", "trying", "thinking", "wondering", "still", "always", "never",
}

_DRINK_PATTERNS = [
    r"\bmy (?:usual|regular|go[- ]?to) is (?:an? )?([a-z ]{3,20})",
    r"\bi (?:always|usually|normally) (?:get|order|have) (?:an? )?([a-z ]{3,20})",
    r"\bi (?:love|like|prefer|adore) (?:an? )?([a-z ]{3,20})",
    r"\bfavou?rite (?:drink|coffee) is (?:an? )?([a-z ]{3,20})",
]


def extract_facts(text: str) -> Profile:
    """Pull a name / favourite drink / milk preference out of a sentence.

    Rule-based on purpose: the graph mechanics are what's being demonstrated, and a
    deterministic extractor keeps the memory demo reproducible with any model.
    """
    lowered = text.lower()
    facts: Profile = {}

    for pattern in _NAME_PATTERNS:
        if match := re.search(pattern, lowered):
            candidate = match.group(1).strip()
            if candidate not in _NOT_NAMES and candidate not in MENU:
                facts["name"] = candidate.capitalize()
                break

    for pattern in _DRINK_PATTERNS:
        if match := re.search(pattern, lowered):
            phrase = match.group(1).strip()
            # Longest menu name mentioned inside the captured phrase.
            hits = [m for m in MENU if m in phrase]
            if hits:
                facts["favorite_drink"] = max(hits, key=len)
                break

    if match := re.search(rf"\b({'|'.join(MILKS)}) milk\b", lowered):
        facts["milk"] = match.group(1)
    elif re.search(r"\b(oat|almond|soy|coconut)\b", lowered) and re.search(
        r"\b(always|usually|prefer|love|like|only)\b", lowered
    ):
        facts["milk"] = re.search(r"\b(oat|almond|soy|coconut)\b", lowered).group(1)

    return facts


def describe_profile(profile: Profile) -> str:
    if not profile:
        return "(nothing learned yet)"
    parts = []
    if name := profile.get("name"):
        parts.append(f"name={name}")
    if drink := profile.get("favorite_drink"):
        parts.append(f"favorite_drink={drink}")
    if milk := profile.get("milk"):
        parts.append(f"milk={milk}")
    return "; ".join(parts) or "(nothing learned yet)"


# --------------------------------------------------------------------------- threads


def register_thread(store: BaseStore, user_id: str, thread_id: str, label: str) -> None:
    """Thread names also live in the long-term store, so the sidebar survives restarts."""
    store.put(THREADS_NS, f"{user_id}::{thread_id}", {
        "thread_id": thread_id, "label": label, "created": time.time(),
    })


def list_threads(store: BaseStore, user_id: str) -> list[dict]:
    prefix = f"{user_id}::"
    items = store.search(THREADS_NS, limit=200)
    rows = [i.value for i in items if i.key.startswith(prefix)]
    rows.sort(key=lambda r: r.get("created", 0.0))
    return rows


def delete_thread(store: BaseStore, user_id: str, thread_id: str) -> None:
    store.delete(THREADS_NS, f"{user_id}::{thread_id}")
