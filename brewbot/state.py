"""TOPIC 2 -- State management: the typed graph state and its reducers.

A reducer answers one question: when a node returns a value for a channel, how does
that value combine with what is already there? BrewBot uses four different answers.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from brewbot.domain import CartItem

Intent = Literal["order", "menu", "remember", "recall", "smalltalk"]
INTENTS: tuple[Intent, ...] = ("order", "menu", "remember", "recall", "smalltalk")


class Profile(TypedDict, total=False):
    """The long-term memory record (Topic 4). Deliberately tiny."""

    name: str
    favorite_drink: str
    milk: str
    learned_in: dict[str, str]  # field -> the thread_id where we first heard it


RESET = "__reset__"


def merge_profile(left: Profile | None, right: Profile | None) -> Profile:
    """CUSTOM REDUCER: shallow-merge profile facts, never losing an old fact.

    `add_messages` appends and `operator.add` concatenates, but a profile wants
    neither -- a new value for `name` should *replace* the old name while leaving
    `favorite_drink` untouched. The nested `learned_in` provenance map is merged too.

    The `__reset__` sentinel clears the channel first, which is how `load_memory`
    makes the state an exact mirror of the long-term store at the start of a turn
    (otherwise a fact deleted from the store would linger in old checkpoints).
    """
    incoming = dict(right or {})
    start_fresh = bool(incoming.pop(RESET, False))
    merged: Profile = {} if start_fresh else dict(left or {})  # type: ignore[assignment]
    for key, value in incoming.items():
        if value in (None, "", {}):
            continue
        if key == "learned_in":
            provenance = dict(merged.get("learned_in", {}))
            provenance.update(value)
            merged["learned_in"] = provenance
        else:
            merged[key] = value  # type: ignore[literal-required]
    return merged


class BrewState(TypedDict):
    """The graph's typed State. Every key is a channel with its own reducer."""

    # 1. add_messages -- appends, de-duplicates by id, and lets a node overwrite an
    #    existing message by re-sending it with the same id.
    messages: Annotated[list[AnyMessage], add_messages]

    # 2. operator.add -- plain list concatenation. Each order node returns only the
    #    NEW lines and the channel accumulates them for the life of the thread.
    cart: Annotated[list[CartItem], operator.add]

    # 3. merge_profile -- our own reducer (see above).
    profile: Annotated[Profile, merge_profile]

    # 4. No annotation => LangGraph's default "last value wins" channel. These are
    #    per-turn scratch values that SHOULD be overwritten, not accumulated.
    intent: Intent
    router_note: str
    scratch: str
    trim_stats: dict[str, Any]


# Rendered as a table in the UI so the reducer choices are visible, not just claimed.
REDUCER_DOCS: list[dict[str, str]] = [
    {
        "channel": "messages",
        "type": "list[AnyMessage]",
        "reducer": "add_messages (built-in)",
        "behaviour": "Appends new messages; dedupes by message id.",
    },
    {
        "channel": "cart",
        "type": "list[CartItem]",
        "reducer": "operator.add (non-default)",
        "behaviour": "Concatenates. Nodes return only new lines; the cart accumulates.",
    },
    {
        "channel": "profile",
        "type": "Profile",
        "reducer": "merge_profile (custom)",
        "behaviour": "Shallow dict merge, so a new fact never erases an older one.",
    },
    {
        "channel": "intent / router_note / scratch / trim_stats",
        "type": "str / dict",
        "reducer": "default LastValue",
        "behaviour": "Overwritten every turn -- these are scratch, not history.",
    },
]


def empty_state() -> BrewState:
    return BrewState(
        messages=[], cart=[], profile={}, intent="smalltalk",
        router_note="", scratch="", trim_stats={},
    )
