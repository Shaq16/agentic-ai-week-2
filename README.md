# ☕ BrewBot — a LangGraph cafe assistant that shows its own wiring

A small chat app for one simple domain: **taking coffee orders at a neighbourhood cafe**.
It exists to demonstrate four LangGraph mechanics, and the front end is built so each one
is *visible* rather than merely claimed.

```
Topic 1  Graph                → the "① Graph" tab renders draw_ascii() of the compiled StateGraph
Topic 2  State & reducers     → the "② State" tab shows every channel, its reducer, and per-thread state
Topic 3  Trimming & filtering → the "③ Trimming" tab shows tokens/messages before vs. after, per turn
Topic 4  Memory               → the "④ Memory" tab contrasts the checkpointer with the store
```

## Run it

```bash
./run.sh                       # creates .venv on first run, then starts Streamlit
```

or manually:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app.py
```

No API key is required — the default **DemoChatModel** runs entirely offline. If
`GEMINI_API_KEY` or `OPENAI_API_KEY` is set, that provider appears in the sidebar
dropdown. Swapping the model changes the prose, never the mechanics.

Verify everything without opening a browser:

```bash
.venv/bin/python tests/test_topics.py
```

```
topic 1 OK  10 nodes, conditional edge routed 5/5 cases
topic 2 OK  cart accumulated to 2 lines ($9.10), messages=6
topic 2 OK  thread-A cart=1 line / thread-B cart=0 lines -> threads are isolated
topic 3 OK  31 msgs/635 tok -> budget 200 sent 7 msgs/185 tok (filtered 1, trimmed 24)
topic 4 OK  new thread 'friday' recalled: Of course -- you're Priya and your usual is
            an oat milk chai latte. I picked that up in 'Monday morning'.
```

## Guided demo (2 minutes)

The sidebar has a **Guided demo** expander with these as one-click buttons.

1. `hi, my name is Priya` → router picks **remember**, the fact goes to the long-term store.
2. `I always get a chai latte with oat milk` → a second fact is merged in.
3. `how much is a mocha?` → router picks **menu**.
4. `can I get two large oat lattes and a croissant` → router picks **order**; the `cart`
   channel grows via `operator.add`, and an *internal ledger* message is appended that is
   filtered out before the next model call.
5. `what's my usual?` → router picks **recall**.
6. **Start a new thread** in the sidebar, then ask `what's my usual?` again. The new thread
   has 0 messages of its own, yet answers correctly and names the thread it learned the fact in.
7. Drag the **token budget** slider from 2000 down to 120 and send another message — watch
   "Sent to model" collapse in the ③ Trimming tab.

---

## Topic 1 — the graph

[`brewbot/graph.py`](brewbot/graph.py) builds a `StateGraph` over the typed `BrewState`,
with eight nodes and one conditional edge.

```
                                                    +-----------+
                                                    | __start__ |
                                                    +-----------+
                                                          *
                                                   +-------------+
                                                   | load_memory |
                                                   +-------------+
                                                          *
                                                  +--------------+
                                              ....| route_intent |.....
                                    ..........  ..+--------------+...............
                           .........        ....           .           ....      .........
+-------------+           +---------------+           +---------------+           +-----------+           +------------+
| answer_menu |*****      | recall_memory |           | remember_fact |           | smalltalk |      *****| take_order |
+-------------+     ******+---------------+***        +---------------+      *****+-----------+******     +------------+
                              **********      ****            *          ****    **********
                                        **************       *      *************
                                                  *******    *   ******
                                                       +---------+
                                                       | respond |
                                                       +---------+
                                                            *
                                                       +---------+
                                                       | __end__ |
                                                       +---------+
```

`route_intent` classifies the turn with explainable rules and writes `intent`. The
**conditional edge** `choose_branch` then fans out to one of five handlers:

| intent | node | what it does |
|---|---|---|
| `order` | `take_order` | parses drinks, appends to `cart` |
| `menu` | `answer_menu` | looks up prices |
| `remember` | `remember_fact` | writes a fact to the long-term store |
| `recall` | `recall_memory` | answers from the long-term store |
| `smalltalk` | `smalltalk` | fallback |

All five converge on `respond` — the **only** node that calls a model, which is why
trimming lives in exactly one place. The UI renders the diagram with `draw_ascii()`
(offline) and offers `draw_mermaid_png()` behind a button.

## Topic 2 — state management with reducers

[`brewbot/state.py`](brewbot/state.py) defines four channels using **four different reducers**:

| channel | reducer | why |
|---|---|---|
| `messages` | `add_messages` (built-in) | appends turns, dedupes by message id |
| `cart` | `operator.add` (non-default) | order nodes return *only new lines*; the channel accumulates |
| `profile` | `merge_profile` (**custom**) | shallow dict-merge, so a new fact never erases an older one |
| `intent`, `router_note`, `scratch`, `trim_stats` | default `LastValue` | per-turn scratch that *should* be overwritten |

`merge_profile` also honours a `__reset__` sentinel, which `load_memory` uses to make the
channel an exact mirror of the store at the start of every turn (otherwise a fact deleted
from the store would linger in old checkpoints).

**Thread isolation** is proved in the ② State tab, which lists every thread with its own
message count and ticket total. Same graph, same checkpointer — only the `thread_id` in
the config differs:

| thread | messages | cart lines | ticket |
|---|---|---|---|
| Morning rush ← current | 22 | 2 | $14.95 |
| Friday visit | 2 | 0 | $0.00 |

## Topic 3 — trimming and filtering

[`brewbot/trimming.py`](brewbot/trimming.py) rebuilds the model payload on every turn, in
two steps, and records stats for the UI:

1. **Filter** — `filter_messages(include_types=[HumanMessage, AIMessage])`, then drop blanks
   and anything tagged `internal`. This fires in normal use: every order appends a
   `[ledger]` audit line to the transcript, which is deliberately *not* model context
   because the cart is already rendered into the system prompt.
2. **Trim** — `trim_messages(strategy="last", include_system=True, start_on="human")` drops
   the oldest turns until the rest fit the token budget from the sidebar slider.

The full transcript always stays in the checkpointer. Trimming changes what is *sent*,
never what is *remembered*.

The ③ Trimming tab shows the before/after for the latest turn, the exact payload that was
sent (expandable, with per-message token counts), and a table of every turn in the thread:

```
History in state   7 msgs / 247 tok
After filtering    5 msgs   (-2 msgs  ← the internal ledger entries)
Sent to model      6 msgs / 203 tok
Tokens saved       44       (budget 400)
```

One honest detail the UI surfaces: because `include_system=True` means the system prompt
can never be dropped, it acts as a **floor**. Ask "what do you have?" with a 120-token
budget and the whole menu lands in the system prompt — `sent_tokens` legitimately exceeds
the budget, and the tab says so rather than hiding it.

## Topic 4 — short-term and long-term memory

[`brewbot/memory.py`](brewbot/memory.py). Both are SQLite-backed, so they also survive an
app restart.

| | short-term | long-term |
|---|---|---|
| backend | `SqliteSaver` (checkpointer) | `SqliteStore` (store) |
| scope | one `thread_id` | a namespace — **every** thread |
| holds | the whole `BrewState`: messages, cart, intent | a small profile: `name`, `favorite_drink`, `milk` |
| file | `data/short_term.sqlite` | `data/long_term.sqlite` |

The profile schema is deliberately tiny, plus a `learned_in` provenance map recording which
thread each fact first came from — which is what makes the cross-thread demo unambiguous:

> **new thread, 0 messages of its own** → *"what's my usual?"*
> → *"Of course — you're Priya and your usual is an oat milk chai latte.
>    I picked that up in 'Morning rush'."*

`load_memory` runs first on every turn and reads the store into state; `remember_fact`
writes back to it. The sidebar shows the live profile and a **Forget me** button.

## Layout

```
app.py                  Streamlit front end (5 tabs, one per topic + chat)
brewbot/
  domain.py             the cafe: menu, prices, order parsing  (no LangGraph)
  state.py              TOPIC 2 — BrewState + the four reducers
  graph.py              TOPIC 1 — nodes, edges, the conditional edge, compile()
  trimming.py           TOPIC 3 — filter + trim + the stats the UI renders
  memory.py             TOPIC 4 — checkpointer, store, profile, thread registry
  models.py             DemoChatModel (offline) + Gemini / OpenAI
tests/test_topics.py    end-to-end proof of all four topics
```
