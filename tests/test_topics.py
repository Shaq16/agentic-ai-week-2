"""End-to-end checks that each of the four topics actually works.

Run with:  .venv/bin/python tests/test_topics.py     (no pytest required)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

import brewbot.memory as memory
from brewbot.domain import cart_total
from brewbot.graph import build_graph
from brewbot.models import DemoChatModel
from brewbot.state import REDUCER_DOCS, merge_profile
from brewbot.trimming import INTERNAL_TAG, build_system_prompt, prepare_model_input


def fresh_backends(tmp: Path):
    """Point the sqlite files at a temp dir so tests never touch real data."""
    memory.DATA_DIR = tmp
    memory.CHECKPOINT_DB = tmp / "short_term.sqlite"
    memory.STORE_DB = tmp / "long_term.sqlite"
    return memory.build_checkpointer(), memory.build_store()


def chat(graph, text: str, thread: str, *, user="tester", label=None, budget=400):
    config = {
        "configurable": {
            "thread_id": thread,
            "user_id": user,
            "thread_label": label or thread,
            "token_budget": budget,
        }
    }
    return graph.invoke({"messages": [HumanMessage(text)]}, config)


# ------------------------------------------------------------------ topic 1


def test_graph_shape_and_routing(graph):
    drawn = graph.get_graph()
    nodes = set(drawn.nodes)
    assert {"load_memory", "route_intent", "respond", "take_order"} <= nodes, nodes
    assert len(nodes) >= 8, nodes

    cases = [
        ("can I get two large oat lattes", "order"),
        ("how much is a mocha?", "menu"),
        ("my name is Priya", "remember"),
        ("what's my usual?", "recall"),
        ("lovely weather today", "smalltalk"),
    ]
    for i, (text, expected) in enumerate(cases):
        out = chat(graph, text, f"route-{i}")
        assert out["intent"] == expected, f"{text!r} -> {out['intent']} (want {expected})"
        assert out["router_note"], "router should explain itself"
    print(f"  topic 1 OK  {len(nodes)} nodes, conditional edge routed {len(cases)}/{len(cases)} cases")


# ------------------------------------------------------------------ topic 2


def test_reducers(graph):
    # operator.add on `cart`: each turn returns only NEW lines, the channel accumulates.
    chat(graph, "can I get a large latte", "reducer-thread")
    after_one = chat(graph, "and an almond croissant please", "reducer-thread")
    assert len(after_one["cart"]) == 2, after_one["cart"]
    assert cart_total(after_one["cart"]) > 0

    # add_messages on `messages`: 2 order turns x (human + internal ledger + ai) = 6.
    kinds = [type(m).__name__ for m in after_one["messages"]]
    assert len(after_one["messages"]) == 6, kinds
    internal = [m for m in after_one["messages"]
                if INTERNAL_TAG in ((m.additional_kwargs or {}).get("tags") or [])]
    assert len(internal) == 2, "each order turn should append one ledger entry"

    # custom merge_profile: a second fact must not erase the first.
    merged = merge_profile({"name": "Priya"}, {"favorite_drink": "latte"})
    assert merged == {"name": "Priya", "favorite_drink": "latte"}, merged
    assert len(REDUCER_DOCS) >= 3
    print(f"  topic 2 OK  cart accumulated to {len(after_one['cart'])} lines "
          f"(${cart_total(after_one['cart']):.2f}), messages={len(after_one['messages'])}")


def test_thread_isolation(graph):
    chat(graph, "can I get three espressos", "thread-A")
    b = chat(graph, "hello there", "thread-B")
    a = graph.get_state({"configurable": {"thread_id": "thread-A"}}).values
    assert len(a["cart"]) == 1 and a["cart"][0]["qty"] == 3, a["cart"]
    assert b["cart"] == [], b["cart"]
    # thread-A: human + ledger + reply; thread-B: human + reply (no order, no ledger).
    assert len(a["messages"]) == 3, [type(m).__name__ for m in a["messages"]]
    assert len(b["messages"]) == 2, [type(m).__name__ for m in b["messages"]]
    print("  topic 2 OK  thread-A cart=1 line / thread-B cart=0 lines -> threads are isolated")


# ------------------------------------------------------------------ topic 3


def test_trimming_and_filtering(graph):
    history = []
    for i in range(15):
        history.append(HumanMessage(f"turn {i}: a fairly wordy question about the cafe menu"))
        history.append(AIMessage(f"turn {i}: a fairly wordy answer describing drinks and prices"))
    history.append(AIMessage(content="", additional_kwargs={"tags": ["internal"]}))

    system = build_system_prompt("menu", {"name": "Priya"}, [], "menu text")
    wide, wide_stats = prepare_model_input(history, system, 5000)
    narrow, narrow_stats = prepare_model_input(history, system, 200)

    assert wide_stats["filtered_out"] == 1, "the empty internal message should be filtered"
    assert narrow_stats["sent_messages"] < wide_stats["sent_messages"]
    assert narrow_stats["sent_tokens"] <= max(200, narrow_stats["system_tokens"])
    assert narrow_stats["saved_tokens"] > 0
    assert any(m.type == "system" for m in narrow), "system prompt must survive trimming"
    assert any(m.type == "human" for m in narrow), "newest user turn must survive trimming"

    # And the budget really reaches the graph.
    out = chat(graph, "what do you have?", "trim-thread", budget=120)
    stats = out["trim_stats"]
    assert stats["budget"] == 120
    # "what do you have?" injects the whole menu into the system prompt, which is a
    # floor trimming cannot go below -- but the history window is still squeezed.
    assert stats["floor_exceeds_budget"], stats["system_tokens"]
    assert stats["sent_messages"] <= 2, stats["sent_messages"]

    # The internal ledger an order writes must never reach the model.
    ordered = chat(graph, "can I get a large mocha", "filter-thread")
    ledger_stats = ordered["trim_stats"]
    assert ledger_stats["filtered_out"] >= 1, ledger_stats
    assert not any("[ledger]" in p["text"] for p in ledger_stats["preview"]), "ledger leaked"
    print(f"  topic 3 OK  {wide_stats['history_messages']} msgs/{wide_stats['history_tokens']} tok "
          f"-> budget 200 sent {narrow_stats['sent_messages']} msgs/{narrow_stats['sent_tokens']} tok "
          f"(filtered {narrow_stats['filtered_out']}, trimmed {narrow_stats['dropped_by_trim']})")


# ------------------------------------------------------------------ topic 4


def test_long_term_memory_across_threads(graph, store):
    # A dedicated user id so earlier tests' facts don't pollute this one.
    who = "memory-demo"

    # Learn two facts in one thread...
    chat(graph, "hi, my name is Priya", "monday", user=who, label="Monday morning")
    chat(graph, "I always get a chai latte with oat milk", "monday", user=who,
         label="Monday morning")

    saved = memory.load_profile(store, who)
    assert saved.get("name") == "Priya", saved
    assert saved.get("favorite_drink") == "chai latte", saved
    assert saved.get("milk") == "oat", saved
    assert saved["learned_in"]["name"] == "Monday morning", saved["learned_in"]

    # ...then start a completely new thread and ask.
    fresh = chat(graph, "what's my usual?", "friday", user=who, label="Friday visit")
    assert fresh["intent"] == "recall"
    assert len(fresh["messages"]) == 2, "the new thread has no history of its own"
    reply = str(fresh["messages"][-1].content)
    assert "Priya" in reply and "chai latte" in reply, reply
    assert "Monday morning" in reply, "recall should say where it learned the fact"
    print(f"  topic 4 OK  new thread 'friday' recalled: {reply.strip()}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        checkpointer, store = fresh_backends(Path(tmp))
        graph = build_graph(DemoChatModel(), checkpointer=checkpointer, store=store)
        print("Running BrewBot topic checks...")
        test_graph_shape_and_routing(graph)
        test_reducers(graph)
        test_thread_isolation(graph)
        test_trimming_and_filtering(graph)
        test_long_term_memory_across_threads(graph, store)
        print("\nAll four topics verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
