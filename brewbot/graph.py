r"""TOPIC 1 -- The LangGraph graph: typed state, nodes, edges, one conditional edge.

    START -> load_memory -> route_intent -*- (conditional) -> take_order    -\
                                           |                  answer_menu    |
                                           |                  remember_fact  +-> respond -> END
                                           |                  recall_memory  |
                                           \-                  smalltalk    -/

`route_intent` classifies the turn and `choose_branch` is the conditional edge that
fans out to one of five handlers. They all converge on `respond`, which is the single
place a model is ever called -- and therefore the single place trimming happens.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from brewbot.domain import cart_total, describe_item, lookup_menu, parse_order, render_cart
from brewbot.memory import extract_facts, load_profile, save_profile
from brewbot.state import RESET, BrewState, Intent, merge_profile
from brewbot.trimming import INTERNAL_TAG, build_system_prompt, prepare_model_input

DEFAULT_BUDGET = 400


def _last_user_text(state: BrewState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _cfg(config: RunnableConfig, key: str, default: Any = None) -> Any:
    return (config or {}).get("configurable", {}).get(key, default)


# ------------------------------------------------------------------ node: memory


def load_memory(state: BrewState, config: RunnableConfig) -> dict:
    """Read long-term memory (Topic 4) into the state before anything else runs."""
    store = get_store()
    profile = load_profile(store, _cfg(config, "user_id", "default"))
    # RESET makes the channel an exact mirror of the store for this turn.
    return {"profile": {RESET: True, **profile}}


# ------------------------------------------------------------------ node: router

_RECALL_RE = re.compile(
    r"\b(my usual|the usual|do you remember|remember me|who am i|what.{0,12}my name"
    r"|what do you know about me|my favou?rite|remind me what)\b"
)
_MENU_RE = re.compile(
    r"\b(menu|how much|what.{0,5}(?:s| is| are) (?:the )?(?:price|cost)|price|prices|cost"
    r"|what do you have|what.{0,4}s good|do you (?:have|sell|do)|options|what.{0,4}s in)\b"
)
_ORDER_RE = re.compile(
    r"\b(can i (?:get|have)|i(?:'| wou)?ld like|i want|give me|make (?:it|me)|order"
    r"|add|grab|take|to go|for here|please)\b"
)
# A deliberately NARROWER set for the itemless case. When nothing on the menu was
# named we still want the order branch (so it can ask what they meant), but only on
# an unmistakable intent to order -- a bare "please" or "add" is far too weak to
# hijack the turn away from smalltalk.
_ORDER_INTENT_RE = re.compile(
    r"\b(can i (?:get|have)|i(?:'| wou)?ld like|i want|give me|take (?:an|my) order"
    r"|(?:place|make) an order|order(?:ing)?(?: something| now| please)?)\b"
)


def route_intent(state: BrewState, config: RunnableConfig) -> dict:
    """Classify the turn. Rule-based so the branch taken is explainable in the UI."""
    text = _last_user_text(state)
    lowered = text.lower()

    if match := _RECALL_RE.search(lowered):
        return {"intent": "recall", "router_note": f"matched recall phrase {match.group(0)!r}"}

    facts = extract_facts(text)
    if facts:
        return {
            "intent": "remember",
            "router_note": "user stated a personal fact: " + ", ".join(facts),
        }

    if match := _MENU_RE.search(lowered):
        return {"intent": "menu", "router_note": f"matched menu phrase {match.group(0)!r}"}

    items = parse_order(text)
    if items and _ORDER_RE.search(lowered):
        return {
            "intent": "order",
            "router_note": f"menu items {[i['name'] for i in items]} + an ordering phrase",
        }
    if items:
        return {"intent": "order", "router_note": f"menu items {[i['name'] for i in items]}"}

    # An intent to order with nothing recognisable on it -- e.g. "take an order".
    # `take_order` handles the empty cart by asking what they actually wanted.
    if match := _ORDER_INTENT_RE.search(lowered):
        return {
            "intent": "order",
            "router_note": f"ordering phrase {match.group(0)!r} but no menu item named",
        }

    return {"intent": "smalltalk", "router_note": "no rule matched -- falling through"}


def choose_branch(state: BrewState) -> Intent:
    """THE CONDITIONAL EDGE. Returns the name of the branch to run next."""
    return state["intent"]


BRANCHES: dict[str, str] = {
    "order": "take_order",
    "menu": "answer_menu",
    "remember": "remember_fact",
    "recall": "recall_memory",
    "smalltalk": "smalltalk",
}


# ------------------------------------------------------------------ nodes: branches


def take_order(state: BrewState, config: RunnableConfig) -> dict:
    items = parse_order(_last_user_text(state))
    if not items:
        return {"scratch": "They tried to order but nothing matched the menu. Ask what they meant."}
    added = ", ".join(describe_item(i) for i in items)
    running = cart_total(list(state.get("cart", [])) + items)

    # An append-only audit line for the transcript. It is tagged `internal` because
    # the cart is already rendered into the system prompt -- resending it as history
    # would just burn tokens. This is what the filter step in Topic 3 removes.
    ledger = AIMessage(
        content=f"[ledger] +{added} -> ticket ${running:.2f}",
        additional_kwargs={"tags": [INTERNAL_TAG]},
    )

    # Only the NEW lines are returned -- `operator.add` appends them to the cart.
    return {"cart": items, "messages": [ledger], "scratch": f"Added {added}."}


def answer_menu(state: BrewState, config: RunnableConfig) -> dict:
    return {"scratch": lookup_menu(_last_user_text(state))}


def remember_fact(state: BrewState, config: RunnableConfig) -> dict:
    """Write a fact to the LONG-TERM store, tagged with the thread it came from."""
    store = get_store()
    user_id = _cfg(config, "user_id", "default")
    where = _cfg(config, "thread_label") or _cfg(config, "thread_id", "unknown")

    facts = extract_facts(_last_user_text(state))
    stored = load_profile(store, user_id)

    provenance = dict(stored.get("learned_in", {}))
    for field in facts:
        provenance.setdefault(field, where)  # remember where we FIRST heard it

    update = {**facts, "learned_in": provenance}
    save_profile(store, user_id, merge_profile(stored, update))

    summary = ", ".join(f"{k}={v}" for k, v in facts.items()) or "nothing new"
    return {
        "profile": update,
        "scratch": f"Just learned: {summary}. Saved to long-term memory.",
    }


def _article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def recall_memory(state: BrewState, config: RunnableConfig) -> dict:
    """Answer from long-term memory, naming the thread the fact was learned in."""
    profile = state.get("profile") or {}
    known = {k: v for k, v in profile.items() if k != "learned_in" and v}
    if not known:
        return {"scratch": "Long-term memory is empty -- say you don't think you've met yet."}

    provenance = profile.get("learned_in", {})
    bits = []
    if name := profile.get("name"):
        bits.append(f"you're {name}")
    if drink := profile.get("favorite_drink"):
        milk = profile.get("milk")
        usual = f"{milk} milk {drink}" if milk else drink
        bits.append(f"your usual is {_article(usual)} {usual}")
    elif milk := profile.get("milk"):
        bits.append(f"you take {milk} milk")

    sources = sorted({v for k, v in provenance.items() if k in known})
    origin = f" I picked that up in {' and '.join(repr(s) for s in sources)}." if sources else ""
    return {"scratch": f"Of course -- {' and '.join(bits)}.{origin}"}


def smalltalk(state: BrewState, config: RunnableConfig) -> dict:
    return {"scratch": ""}


# ------------------------------------------------------------------ node: respond


def make_respond(model: BaseChatModel):
    """The only node that calls the model -- so trimming lives in exactly one place."""

    def respond(state: BrewState, config: RunnableConfig) -> dict:
        budget = int(_cfg(config, "token_budget", DEFAULT_BUDGET))
        system = build_system_prompt(
            state["intent"], state.get("profile", {}), state.get("cart", []), state.get("scratch", "")
        )
        # TOPIC 3: filter + trim the full history down to what actually gets sent.
        messages, stats = prepare_model_input(state["messages"], system, budget)
        try:
            reply = model.invoke(messages)
        except Exception as exc:  # a provider failure shouldn't lose the conversation
            reply = AIMessage(content=f"(model error: {type(exc).__name__}: {exc})")
        stats["system_prompt"] = str(system.content)
        return {"messages": [reply], "trim_stats": stats}

    return respond


# ------------------------------------------------------------------ assembly


def build_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """Wire the nodes and edges together and compile."""
    builder = StateGraph(BrewState)

    builder.add_node("load_memory", load_memory)
    builder.add_node("route_intent", route_intent)
    builder.add_node("take_order", take_order)
    builder.add_node("answer_menu", answer_menu)
    builder.add_node("remember_fact", remember_fact)
    builder.add_node("recall_memory", recall_memory)
    builder.add_node("smalltalk", smalltalk)
    builder.add_node("respond", make_respond(model))

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "route_intent")

    # The conditional edge: one source, five possible destinations.
    builder.add_conditional_edges("route_intent", choose_branch, BRANCHES)

    for branch in BRANCHES.values():
        builder.add_edge(branch, "respond")
    builder.add_edge("respond", END)

    # checkpointer = short-term memory (per thread); store = long-term (cross-thread).
    return builder.compile(checkpointer=checkpointer, store=store)
