"""TOPIC 3 -- Trimming and filtering the history before every model call.

The graph state keeps the *full* transcript forever (that is what short-term memory
is for). What we hand the model is a smaller, cheaper view of it, rebuilt on every
turn in two steps:

    filter  -- drop messages that are structurally useless (empty, or internal notes)
    trim    -- drop the oldest messages until the rest fit a token budget

`stats` records both steps so the UI can show before/after counts instead of just
asserting that trimming happened.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

from brewbot.domain import cart_total, render_cart
from brewbot.state import Profile

# Messages tagged with this are bookkeeping, never sent to the model.
INTERNAL_TAG = "internal"


def count_tokens(messages: list[BaseMessage]) -> int:
    """Approximate token counter (~4 chars/token) plus per-message overhead.

    Deliberately model-agnostic so the demo behaves identically offline and online.
    """
    return count_tokens_approximately(messages)


def build_system_prompt(intent: str, profile: Profile, cart: list, scratch: str) -> SystemMessage:
    """The one message that is always kept, no matter how tight the budget."""
    lines = [
        "You are BrewBot, the counter assistant at a small neighbourhood cafe.",
        "Be warm and brief -- two or three sentences, no bullet lists.",
        f"[TASK] {intent}",
    ]

    if profile:
        known = []
        if name := profile.get("name"):
            known.append(f"name={name}")
        if drink := profile.get("favorite_drink"):
            known.append(f"favorite_drink={drink}")
        if milk := profile.get("milk"):
            known.append(f"milk={milk}")
        lines.append("[PROFILE] " + ("; ".join(known) if known else "empty"))
        lines.append("Greet them by name when you know it, and use their usual order if relevant.")
    else:
        lines.append("[PROFILE] empty")

    lines.append(f"[CART] {render_cart(cart)}" if cart else "[CART] (empty)")
    if cart:
        lines.append(f"[CART_TOTAL] ${cart_total(cart):.2f}")
    if scratch:
        lines.append(f"[CONTEXT]\n{scratch}")

    return SystemMessage(content="\n".join(lines))


def _is_internal(message: BaseMessage) -> bool:
    tags = (message.additional_kwargs or {}).get("tags") or []
    return INTERNAL_TAG in tags


def prepare_model_input(
    history: list[AnyMessage],
    system: SystemMessage,
    token_budget: int,
) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Filter, then trim. Returns the messages to send plus stats for the UI."""
    before_count = len(history)
    before_tokens = count_tokens([system, *history])

    # --- step 1: filter -------------------------------------------------------
    # Keep only real conversational turns, then drop internal notes and blanks.
    kept = filter_messages(history, include_types=[HumanMessage, AIMessage])
    kept = [m for m in kept if not _is_internal(m) and str(m.content).strip()]
    filtered_out = before_count - len(kept)
    after_filter_tokens = count_tokens([system, *kept])

    # --- step 2: trim ---------------------------------------------------------
    # `include_system` keeps the system prompt; `start_on="human"` guarantees the
    # window still begins with a user turn so the model never sees a dangling reply.
    trimmed = trim_messages(
        [system, *kept],
        max_tokens=token_budget,
        token_counter=count_tokens,
        strategy="last",
        include_system=True,
        start_on="human",
        allow_partial=False,
    )

    # A very small budget can starve everything except the system prompt. Always
    # keep the newest user turn, otherwise the model has no question to answer.
    if not any(isinstance(m, HumanMessage) for m in trimmed):
        newest_human = next((m for m in reversed(kept) if isinstance(m, HumanMessage)), None)
        if newest_human is not None:
            trimmed = [system, newest_human]

    dropped_by_trim = len(kept) + 1 - len(trimmed)
    after_tokens = count_tokens(trimmed)

    # The system prompt is never droppable (include_system=True), so it is a floor
    # the budget cannot push below. Surfacing it keeps the UI numbers honest: with a
    # tiny budget and a big [CONTEXT] block, `sent_tokens` can legitimately exceed it.
    system_tokens = count_tokens([system])

    stats = {
        "budget": token_budget,
        "system_tokens": system_tokens,
        "floor_exceeds_budget": system_tokens > token_budget,
        "history_messages": before_count,
        "history_tokens": before_tokens,
        "after_filter_messages": len(kept),
        "after_filter_tokens": after_filter_tokens,
        "filtered_out": filtered_out,
        "sent_messages": len(trimmed),
        "sent_tokens": after_tokens,
        "dropped_by_trim": max(0, dropped_by_trim),
        "saved_tokens": max(0, before_tokens - after_tokens),
        "preview": [
            {
                "role": m.__class__.__name__.replace("Message", ""),
                "tokens": count_tokens([m]),
                "text": str(m.content),
            }
            for m in trimmed
        ],
    }
    return trimmed, stats
