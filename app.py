"""BrewBot -- a cafe ordering assistant that shows its own LangGraph mechanics.

Run:  .venv/bin/streamlit run app.py

Every tab maps to one of the four topics:
    Graph    -> Topic 1  (state, nodes, edges, conditional edge)
    State    -> Topic 2  (reducers, checkpointer, thread isolation)
    Trimming -> Topic 3  (filter + trim before each model call)
    Memory   -> Topic 4  (short-term checkpointer vs long-term store)
"""

from __future__ import annotations

import html
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage, HumanMessage

from brewbot.domain import cart_total, describe_item
from brewbot.graph import BRANCHES, build_graph
from brewbot.memory import (
    DATA_DIR,
    build_checkpointer,
    build_store,
    delete_thread,
    describe_profile,
    forget_profile,
    list_threads,
    load_profile,
    register_thread,
)
from brewbot.models import PROVIDERS, available_providers, get_model
from brewbot.state import REDUCER_DOCS
from brewbot.trimming import (
    INTERNAL_TAG,
    build_system_prompt,
    count_tokens,
    prepare_model_input,
)

st.set_page_config(page_title="BrewBot -- LangGraph demo", page_icon="☕", layout="wide")


# ------------------------------------------------------------------ resources


@st.cache_resource(show_spinner=False)
def get_backends():
    """One checkpointer + one store for the whole app (both SQLite-backed)."""
    return build_checkpointer(), build_store()


@st.cache_resource(show_spinner=False)
def get_graph(provider: str, model_name: str):
    """Recompiled when the model changes; memory backends are reused."""
    checkpointer, store = get_backends()
    return build_graph(get_model(provider, model_name), checkpointer=checkpointer, store=store)


@st.cache_data(show_spinner=False)
def graph_ascii(provider: str, model_name: str) -> str:
    return get_graph(provider, model_name).get_graph().draw_ascii()


@st.cache_data(show_spinner=False)
def graph_mermaid(provider: str, model_name: str) -> str:
    return get_graph(provider, model_name).get_graph().draw_mermaid()


checkpointer, store = get_backends()


# ------------------------------------------------------------------ session


def new_thread_id(label: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-") or "thread"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


defaults = {
    "user_id": "priya",
    "thread_id": None,
    "traces": {},          # thread_id -> list of per-turn trim/route traces
    "pending": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

user_id = st.session_state.user_id
threads = list_threads(store, user_id)
if not threads:
    first = new_thread_id("Morning rush")
    register_thread(store, user_id, first, "Morning rush")
    threads = list_threads(store, user_id)
if st.session_state.thread_id not in {t["thread_id"] for t in threads}:
    st.session_state.thread_id = threads[0]["thread_id"]


def thread_label(thread_id: str) -> str:
    return next(
        (t["label"] for t in threads if t["thread_id"] == thread_id), thread_id
    )


def config_for(thread_id: str, budget: int) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "thread_label": thread_label(thread_id),
            "token_budget": budget,
        }
    }


def state_of(thread_id: str) -> dict:
    return get_graph(provider, model_name).get_state(config_for(thread_id, 400)).values or {}


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("☕ BrewBot")
    st.caption("A LangGraph cafe assistant that shows its own wiring.")

    st.subheader("Model")
    ready = available_providers()
    provider = st.selectbox(
        "Provider", ready, format_func=lambda p: PROVIDERS[p]["label"],
        help="DemoChatModel needs no API key and runs fully offline.",
    )
    model_name = st.selectbox("Model", PROVIDERS[provider]["models"])
    missing = [PROVIDERS[p]["env"] for p in PROVIDERS if p not in ready and PROVIDERS[p]["env"]]
    if missing:
        st.caption("Set " + " or ".join(missing) + " to unlock more providers.")

    st.subheader("Token budget (Topic 3)")
    budget = st.slider(
        "Max tokens sent to the model", min_value=80, max_value=2000, value=400, step=20,
        help="Lower this and watch older messages get dropped before the model call.",
    )

    st.subheader("Customer")
    st.text_input("user_id", key="user_id", help="Long-term memory is stored per user_id.")

    st.subheader("Threads (Topic 2 + 4)")
    labels = {t["thread_id"]: t["label"] for t in threads}
    choice = st.radio(
        "Conversation", list(labels), index=list(labels).index(st.session_state.thread_id),
        format_func=lambda t: labels[t], label_visibility="collapsed",
    )
    if choice != st.session_state.thread_id:
        st.session_state.thread_id = choice
        st.rerun()

    with st.form("new_thread", clear_on_submit=True):
        new_label = st.text_input("New conversation", placeholder="e.g. Friday visit")
        if st.form_submit_button("Start thread", use_container_width=True) and new_label.strip():
            tid = new_thread_id(new_label.strip())
            register_thread(store, user_id, tid, new_label.strip())
            st.session_state.thread_id = tid
            st.rerun()

    if len(threads) > 1 and st.button("Delete this thread", use_container_width=True):
        delete_thread(store, user_id, st.session_state.thread_id)
        st.session_state.thread_id = None
        st.rerun()

    st.subheader("Long-term memory (Topic 4)")
    profile = load_profile(store, user_id)
    st.info(describe_profile(profile))
    if profile and st.button("Forget me", use_container_width=True):
        forget_profile(store, user_id)
        st.rerun()

    with st.expander("Guided demo"):
        st.caption("Click these in order to exercise all four topics.")
        script = [
            "hi, my name is Priya",
            "I always get a chai latte with oat milk",
            "how much is a mocha?",
            "can I get two large oat lattes and a croissant",
            "what's my usual?",
        ]
        for i, line in enumerate(script):
            if st.button(line, key=f"demo-{i}", use_container_width=True):
                st.session_state.pending = line
                st.rerun()
        st.caption("Then start a NEW thread and click *what's my usual?* again.")


thread_id = st.session_state.thread_id
graph = get_graph(provider, model_name)
config = config_for(thread_id, budget)


# ------------------------------------------------------------------ turn handling

if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None
    with st.spinner("BrewBot is thinking..."):
        result = graph.invoke({"messages": [HumanMessage(prompt)]}, config)
    trace = {
        "prompt": prompt,
        "intent": result.get("intent"),
        "router_note": result.get("router_note"),
        "stats": result.get("trim_stats", {}),
    }
    st.session_state.traces.setdefault(thread_id, []).append(trace)
    st.rerun()


def md(text) -> str:
    """Escape text for Streamlit markdown.

    Streamlit reads `$...$` as LaTeX. Cafe prices are full of dollar signs, so
    "espresso $2.50, macchiato $3.25" had everything between two prices
    swallowed into a math span -- the menu rendered in the wrong font with the
    dollar signs missing. The model still receives the unescaped string; only
    the display copy is escaped.
    """
    return str(text).replace("$", r"\$")


def esc(text) -> str:
    """For blocks rendered with unsafe_allow_html, where markdown escaping does
    not apply and md()'s backslash would show up on screen as "\\$2.50"."""
    return html.escape(str(text))


def is_internal(message) -> bool:
    return INTERNAL_TAG in ((message.additional_kwargs or {}).get("tags") or [])


snapshot = graph.get_state(config)
state = snapshot.values or {}
all_messages = state.get("messages", [])
# Internal ledger lines live in state but are shown separately, never as chat turns.
messages = [m for m in all_messages if not is_internal(m)]
ledger = [m for m in all_messages if is_internal(m)]
cart = state.get("cart", [])
traces = st.session_state.traces.get(thread_id, [])
last_stats = traces[-1]["stats"] if traces else {}

# Fetched once and shared by the Graph and Memory tabs.
checkpoints = list(graph.get_state_history(config))

# `st.session_state.traces` only holds turns sent in THIS browser session, so a
# reload emptied the graph walkthrough even though the conversation was still
# there. The checkpointer knows better: it wrote a snapshot before every node,
# and the one whose `next` is a handler node records exactly which branch
# `choose_branch` picked. Reconstructing from that survives reloads and works
# for conversations this session never saw.
_handlers = set(BRANCHES.values())


def routed_turns() -> list[dict]:
    out = []
    for snap in reversed(checkpoints):
        nxt = snap.next or ()
        if len(nxt) == 1 and nxt[0] in _handlers:
            values = snap.values or {}
            asked = [m for m in values.get("messages", []) if isinstance(m, HumanMessage)]
            out.append({
                "branch": nxt[0],
                "intent": values.get("intent"),
                "router_note": values.get("router_note"),
                "prompt": str(asked[-1].content).strip() if asked else "",
            })
    return out


turns = routed_turns()


# ------------------------------------------------------------------ header

left, right = st.columns([3, 2])
with left:
    st.markdown(f"### {thread_label(thread_id)}")
    st.caption(f"thread_id `{thread_id}` · user_id `{user_id}` · model `{model_name}`")
with right:
    a, b, c = st.columns(3)
    a.metric("Messages", len(messages))
    b.metric("Cart lines", len(cart))
    c.metric("Ticket", f"${cart_total(cart):.2f}")

tab_chat, tab_graph, tab_state, tab_trim, tab_memory = st.tabs(
    ["💬 Chat", "① Graph", "② State & reducers", "③ Trimming", "④ Memory"]
)


# ------------------------------------------------------------------ chat tab

with tab_chat:
    if not messages:
        st.info("Say hello, ask for the menu, or order something. "
                "Use the sidebar's *Guided demo* to walk all four topics.")

    turn = 0
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(md(message.content))
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="☕"):
                st.markdown(md(message.content))
                if turn < len(traces):
                    t = traces[turn]
                    s = t["stats"]
                    with st.expander(
                        f"trace · intent **{t['intent']}** · "
                        f"sent {s.get('sent_messages', '?')}/{s.get('history_messages', '?')} msgs "
                        f"({s.get('sent_tokens', '?')} tok)",
                        expanded=(turn == len(traces) - 1),
                    ):
                        st.markdown(f"**Router:** {md(t['router_note'])}")
                        st.markdown(f"**Branch taken:** `{BRANCHES.get(t['intent'], '?')}`")
                        st.markdown(
                            f"**Filtered out:** {s.get('filtered_out', 0)} · "
                            f"**Dropped by trim:** {s.get('dropped_by_trim', 0)} · "
                            f"**Tokens saved:** {s.get('saved_tokens', 0)}"
                        )
                turn += 1

    if cart:
        with st.expander(f"🧾 Ticket -- {len(cart)} lines, ${cart_total(cart):.2f}", expanded=False):
            for item in cart:
                st.markdown("- " + md(describe_item(item)))
            st.caption("This channel uses `operator.add`: each order node returns only "
                       "the new lines and the reducer appends them.")
    if ledger:
        with st.expander(f"🔒 {len(ledger)} internal ledger entries (never sent to the model)"):
            for entry in ledger:
                st.code(entry.content, language=None)
            st.caption("These sit in the `messages` channel but carry an `internal` tag, so "
                       "the filter step in Topic 3 strips them before every model call.")


# ------------------------------------------------------------------ graph tab

with tab_graph:
    st.subheader("Topic 1 -- a compiled StateGraph")
    st.markdown(
        "`route_intent` ends in a **conditional edge** (`choose_branch`) that picks one of "
        "five handler nodes. They all converge on `respond`, the only node that calls a model."
    )
    st.code(graph_ascii(provider, model_name), language=None)

    # --- interactive: which path did a given turn actually take? -------------
    st.markdown("##### Walk an actual turn through the graph")
    if not turns:
        st.info("Send a message first — then you can step through the exact path it took.")
    else:
        pick = st.select_slider(
            "Turn",
            options=list(range(1, len(turns) + 1)),
            value=len(turns),
            format_func=lambda i: f"{i}. {turns[i-1]['prompt'][:38] or '(no prompt)'}",
            key="graph-turn",
        )
        t = turns[pick - 1]
        branch = t["branch"]
        path = ["load_memory", "route_intent", branch, "respond", "END"]
        chips = []
        for node in path:
            hot = node == branch
            chips.append(
                f"<span style=\"display:inline-block;padding:5px 12px;margin:3px 2px;"
                f"border-radius:6px;font-family:ui-monospace,monospace;font-size:12.5px;"
                f"background:{'#ff4b4b' if hot else 'rgba(128,128,128,.18)'};"
                f"color:{'#fff' if hot else 'inherit'};"
                f"font-weight:{'700' if hot else '400'}\">{node}</span>"
            )
        st.markdown(
            "&nbsp;<span style='opacity:.45'>→</span>&nbsp;".join(chips),
            unsafe_allow_html=True,
        )
        st.caption(
            f"`choose_branch` read intent **{t['intent']}** and sent this turn to "
            f"**{branch}** — the highlighted node. Every other handler was skipped."
        )
        if t["router_note"]:
            st.markdown(f"**Router said:** {md(t['router_note'])}")
        st.caption(f"Recovered from the checkpointer, not from this browser session — "
                   f"so all {len(turns)} turns in this thread are here after a reload.")

        # --- how often each branch has fired in this thread ------------------
        st.markdown("##### Which branches this conversation has actually used")
        counts = {node: 0 for node in dict.fromkeys(BRANCHES.values())}
        for tr in turns:
            if tr["branch"] in counts:
                counts[tr["branch"]] += 1
        widest = max(counts.values()) or 1
        for node, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            bar = "█" * round(14 * n / widest) if n else ""
            st.markdown(
                f"<code>{node:<16}</code> <span style='color:#ff4b4b'>{bar}</span> "
                f"<span style='opacity:.6'>{n} turn{'' if n == 1 else 's'}</span>",
                unsafe_allow_html=True,
            )
        unused = [n for n, c in counts.items() if c == 0]
        if unused:
            st.caption("Never taken yet: " + ", ".join(f"`{u}`" for u in unused)
                       + " — the conditional edge only ever runs one of these per turn.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nodes**")
        st.table([
            {"node": "load_memory", "does": "reads the long-term store into state"},
            {"node": "route_intent", "does": "classifies the turn, writes `intent`"},
            {"node": "take_order", "does": "parses drinks, appends to `cart`"},
            {"node": "answer_menu", "does": "looks up prices"},
            {"node": "remember_fact", "does": "writes a fact to the store"},
            {"node": "recall_memory", "does": "answers from the store"},
            {"node": "smalltalk", "does": "fallback"},
            {"node": "respond", "does": "trims history, calls the model"},
        ])
    with col2:
        st.markdown("**Conditional edge**")
        st.table([{"intent": k, "goes to node": v} for k, v in BRANCHES.items()])
        if st.button("Render PNG via mermaid.ink (needs internet)"):
            try:
                st.image(graph.get_graph().draw_mermaid_png())
            except Exception as exc:
                st.warning(f"Could not render PNG ({type(exc).__name__}). The ASCII "
                           "diagram above is generated the same way, offline.")

    with st.expander("Mermaid source"):
        st.code(graph_mermaid(provider, model_name), language="text")


# ------------------------------------------------------------------ state tab

with tab_state:
    st.subheader("Topic 2 -- channels, reducers, and thread isolation")
    st.markdown("**`BrewState` channels and the reducer each one uses**")
    st.table(REDUCER_DOCS)

    st.markdown("**Live channel values for this thread**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("`cart` -- reduced with `operator.add`")
        st.json(cart if cart else [], expanded=False)
        st.caption("`profile` -- reduced with the custom `merge_profile`")
        st.json(state.get("profile", {}), expanded=False)
    with c2:
        st.caption("`intent` / `router_note` / `scratch` -- default LastValue")
        st.json({
            "intent": state.get("intent"),
            "router_note": state.get("router_note"),
            "scratch": (state.get("scratch") or "")[:300],
        }, expanded=False)

    st.markdown("**Every thread for this user -- proof they stay separate**")
    rows = []
    for t in threads:
        values = state_of(t["thread_id"])
        rows.append({
            "thread": t["label"] + ("  ← current" if t["thread_id"] == thread_id else ""),
            "thread_id": t["thread_id"],
            "messages": len(values.get("messages", [])),
            "cart lines": len(values.get("cart", [])),
            "ticket": f"${cart_total(values.get('cart', [])):.2f}",
        })
    st.table(rows)
    st.caption("Same graph, same checkpointer -- only the `thread_id` in the config differs, "
               "so messages and carts never leak between conversations.")


# ------------------------------------------------------------------ trimming tab

with tab_trim:
    st.subheader("Topic 3 -- what actually reaches the model")

    # --- interactive: recompute the payload live, with no model call ---------
    # prepare_model_input is pure, so any budget can be previewed against the
    # current history for free. The sidebar slider only took effect on the NEXT
    # message, which made the whole mechanism invisible until you sent one.
    if all_messages:
        st.markdown("##### Drag the budget and watch the window close")
        live_budget = st.slider(
            "Token budget to preview", min_value=40, max_value=2000,
            value=budget, step=20, key="trim-live",
            help="Recomputed instantly against this thread's history. "
                 "No model is called, and nothing is saved.",
        )
        system = build_system_prompt(
            state.get("intent") or "smalltalk",
            state.get("profile", {}) or {},
            cart,
            state.get("scratch") or "",
        )
        sent, live = prepare_model_input(all_messages, system, live_budget)
        sent_ids = {id(m) for m in sent}

        gone = live["filtered_out"] + live["dropped_by_trim"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Would be sent", f"{live['sent_messages']} msgs",
                  f"{live['sent_tokens']} tok")
        k2.metric("Dropped", f"{gone} msgs",
                  f"-{live['saved_tokens']} tok" if live["saved_tokens"] else "nothing lost",
                  delta_color="inverse" if live["saved_tokens"] else "off")
        k3.metric("Budget", live_budget,
                  "system floor " + str(live["system_tokens"]))

        # Binary-search the budget at which trim_messages starts dropping, so the
        # slider has a target instead of sitting at "0 dropped".
        if prepare_model_input(all_messages, system, 2000)[1]["dropped_by_trim"] == 0:
            lo, hi = 40, 2000
            while lo < hi:
                mid = (lo + hi) // 2
                if prepare_model_input(all_messages, system, mid)[1]["dropped_by_trim"]:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 40:
                st.caption(f"With this history, `trim_messages` starts dropping turns "
                           f"below **{lo} tokens**. Drag under that to watch the oldest "
                           f"ones fall away first.")

        if live["floor_exceeds_budget"]:
            st.warning(
                f"The system prompt alone is {live['system_tokens']} tokens. "
                f"`include_system=True` means it cannot be dropped, so below this "
                f"budget only the history gets squeezed."
            )

        st.caption("Every message in this thread, newest last. "
                   "**Kept** rows go to the model; dimmed rows do not.")
        for m in all_messages:
            role = m.__class__.__name__.replace("Message", "")
            tok = count_tokens([m])
            body = str(m.content).strip().replace("\n", " ")[:96] or "(empty)"
            if is_internal(m):
                why, kept = "internal tag — filtered", False
            elif not str(m.content).strip():
                why, kept = "blank — filtered", False
            elif id(m) in sent_ids:
                why, kept = "sent", True
            else:
                why, kept = "too old for the budget — trimmed", False
            if kept:
                st.markdown(
                    f"<div style='padding:4px 0'><span style='background:#ff4b4b;color:#fff;"
                    f"padding:1px 7px;border-radius:4px;font-size:11px;font-weight:700'>KEPT</span> "
                    f"<code>{role}</code> <span style='opacity:.55;font-size:12px'>{tok} tok</span><br>"
                    f"<span style='font-size:13px'>{esc(body)}</span></div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='padding:4px 0;opacity:.42'><span style='border:1px solid currentColor;"
                    f"padding:1px 7px;border-radius:4px;font-size:11px'>{why}</span> "
                    f"<code>{role}</code> <span style='font-size:12px'>{tok} tok</span><br>"
                    f"<span style='font-size:13px;text-decoration:line-through'>{esc(body)}</span></div>",
                    unsafe_allow_html=True)
        st.caption("The full transcript stays in the checkpointer either way — "
                   "trimming changes what is *sent*, never what is *remembered*.")
        st.divider()

    st.markdown("##### What was actually sent on the last real turn")
    if not last_stats:
        st.info("Send a message first, then come back to see the before/after numbers.")
    else:
        s = last_stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("History in state", f"{s['history_messages']} msgs", f"{s['history_tokens']} tok")
        m2.metric("After filtering", f"{s['after_filter_messages']} msgs",
                  f"-{s['filtered_out']} msgs", delta_color="inverse")
        m3.metric("Sent to model", f"{s['sent_messages']} msgs", f"{s['sent_tokens']} tok")
        m4.metric("Tokens saved", s["saved_tokens"], f"budget {s['budget']}")

        if s.get("floor_exceeds_budget"):
            st.warning(
                f"The system prompt alone is {s['system_tokens']} tokens, above the "
                f"{s['budget']}-token budget. `trim_messages(include_system=True)` will not "
                "drop it, so it acts as a floor -- only the history gets squeezed."
            )

        st.markdown("**The exact payload sent on the last turn**")
        for part in s.get("preview", []):
            with st.expander(f"{part['role']} · {part['tokens']} tokens"):
                st.text(part["text"])

        with st.expander("How this is computed"):
            st.markdown(
                "1. **Filter** -- `filter_messages(include_types=[HumanMessage, AIMessage])`, "
                "then drop blanks and anything tagged `internal`.\n"
                "2. **Trim** -- `trim_messages(strategy='last', include_system=True, "
                "start_on='human')` drops the *oldest* turns until the rest fit the budget.\n\n"
                "The full transcript stays in the checkpointer either way -- trimming only "
                "changes what is sent, never what is remembered."
            )

        if len(traces) > 1:
            st.markdown("**Every turn in this thread**")
            st.table([
                {
                    "turn": i + 1,
                    "intent": t["intent"],
                    "history": t["stats"].get("history_messages"),
                    "sent": t["stats"].get("sent_messages"),
                    "tokens before": t["stats"].get("history_tokens"),
                    "tokens sent": t["stats"].get("sent_tokens"),
                }
                for i, t in enumerate(traces)
            ])


# ------------------------------------------------------------------ memory tab

with tab_memory:
    st.subheader("Topic 4 -- two kinds of memory")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Short-term · checkpointer")
        st.caption("`SqliteSaver`, scoped to one `thread_id`.")
        history = checkpoints
        st.metric("Checkpoints in this thread", len(history))
        st.write(f"Messages held: **{len(messages)}** · Cart lines: **{len(cart)}**")
        st.caption(f"Next node: `{snapshot.next or ('END',)}`")
        st.code(str(DATA_DIR / "short_term.sqlite"), language=None)

        # --- interactive: step back through saved checkpoints ---------------
        # get_state_history was already being fetched just to count it. Every
        # one of those snapshots is a full state the checkpointer can replay,
        # so let it be inspected rather than merely tallied.
        if len(history) > 1:
            st.markdown("**Rewind through the saved checkpoints**")
            oldest_first = list(reversed(history))
            idx = st.slider(
                "Checkpoint", 0, len(oldest_first) - 1, len(oldest_first) - 1,
                key="ckpt", help="0 is the oldest. The rightmost is now.",
            )
            snap = oldest_first[idx]
            vals = snap.values or {}
            snap_msgs = [m for m in vals.get("messages", []) if not is_internal(m)]
            snap_cart = vals.get("cart", [])
            d1, d2, d3 = st.columns(3)
            d1.metric("Messages", len(snap_msgs))
            d2.metric("Cart lines", len(snap_cart))
            d3.metric("Ticket", f"${cart_total(snap_cart):.2f}")
            st.caption(
                f"Written after `{', '.join(snap.next) if snap.next else 'END'}`"
                + (" · this is the current state" if idx == len(oldest_first) - 1 else "")
            )
            if snap_msgs:
                last = snap_msgs[-1]
                st.markdown(
                    f"Newest message at this point — `{last.__class__.__name__}`:  \n"
                    + md(str(last.content).strip()[:220] or "(empty)")
                )
            st.caption("Nothing here is mutated — this reads snapshots the "
                       "checkpointer already wrote after every node.")

    with c2:
        st.markdown("#### Long-term · store")
        st.caption("`SqliteStore`, scoped to a namespace -- visible from *every* thread.")
        current = load_profile(store, user_id)
        if current:
            st.json({k: v for k, v in current.items() if k != "learned_in"})
            st.markdown("**Where each fact was learned**")
            st.table([
                {"fact": k, "learned in thread": v}
                for k, v in current.get("learned_in", {}).items()
            ])
        else:
            st.info("Nothing learned yet. Try *my name is Priya* or *I always get a chai latte*.")
        st.code(str(DATA_DIR / "long_term.sqlite"), language=None)

    st.divider()
    st.markdown("**The cross-thread proof**")
    st.markdown(
        "1. In one thread, say **“my name is Priya”** — `remember_fact` writes it to the store.\n"
        "2. Start a **new thread** in the sidebar. Its checkpointer state is empty (0 messages).\n"
        "3. Ask **“what's my usual?”** — `load_memory` reads the store, so the new thread "
        "answers correctly and even names the thread it originally learned the fact in."
    )
    proof = []
    for t in threads:
        values = state_of(t["thread_id"])
        proof.append({
            "thread": t["label"],
            "own messages (short-term)": len(values.get("messages", [])),
            "profile visible (long-term)": describe_profile(load_profile(store, user_id)),
        })
    st.table(proof)


# ------------------------------------------------------------------ input

if prompt := st.chat_input("Order a drink, ask the menu, or tell BrewBot your name..."):
    st.session_state.pending = prompt
    st.rerun()
