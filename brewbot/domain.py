"""BrewBot's little world: a cafe menu plus the parsers that turn free text into cart items.

Nothing in here is LangGraph-specific -- it is the "simple domain" the graph reasons about.
"""

from __future__ import annotations

import re
from typing import TypedDict


class CartItem(TypedDict):
    """One line on the order. Cart items are appended by the `operator.add` reducer."""

    name: str
    size: str
    milk: str | None
    qty: int
    unit_price: float
    line_total: float


DRINKS: dict[str, float] = {
    "espresso": 2.50,
    "macchiato": 3.25,
    "cortado": 3.50,
    "americano": 3.00,
    "drip coffee": 2.75,
    "latte": 4.00,
    "cappuccino": 4.00,
    "flat white": 4.25,
    "mocha": 4.75,
    "cold brew": 4.50,
    "iced latte": 4.50,
    "matcha latte": 5.00,
    "chai latte": 4.50,
    "hot chocolate": 4.00,
    "tea": 2.75,
}

FOOD: dict[str, float] = {
    "croissant": 3.25,
    "almond croissant": 4.00,
    "blueberry muffin": 3.50,
    "banana bread": 3.75,
    "bagel": 3.00,
    "cinnamon roll": 4.25,
    "avocado toast": 8.50,
}

MENU: dict[str, float] = {**DRINKS, **FOOD}

# Size surcharges. Food ignores size and always rings up as "regular".
SIZES: dict[str, float] = {"small": 0.00, "medium": 0.60, "large": 1.10}
SIZE_ALIASES = {
    "small": "small", "sm": "small", "short": "small", "tall": "small",
    "medium": "medium", "med": "medium", "regular": "medium", "grande": "medium",
    "large": "large", "lg": "large", "big": "large", "venti": "large",
}

MILKS = ["whole", "skim", "oat", "almond", "soy", "coconut"]
MILK_SURCHARGE = 0.75
NON_MILK_MILKS = {"oat", "almond", "soy", "coconut"}

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2, "pair": 2, "dozen": 12,
}

# Longest names first so "almond croissant" wins over "croissant" and
# "matcha latte" wins over "latte".
_ITEM_PATTERNS = sorted(MENU, key=len, reverse=True)

Span = tuple[int, int]


def _price(name: str, size: str, milk: str | None) -> float:
    base = MENU[name]
    if name in DRINKS:
        base += SIZES.get(size, 0.0)
        if milk in NON_MILK_MILKS:
            base += MILK_SURCHARGE
    return round(base, 2)


def _overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _find_items(lowered: str) -> list[tuple[Span, str]]:
    """Locate menu items, longest name first so short names cannot steal a span."""
    claimed: list[Span] = []
    found: list[tuple[Span, str]] = []
    for name in _ITEM_PATTERNS:
        for match in re.finditer(rf"\b{re.escape(name)}s?\b", lowered):
            span = match.span()
            if any(_overlap(span, c) for c in claimed):
                continue
            claimed.append(span)
            found.append((span, name))
    found.sort(key=lambda f: f[0][0])
    return found


def _modifier_hits(lowered: str, words: dict[str, str], item_spans: list[Span]) -> list[tuple[Span, str]]:
    """Find modifier words, skipping any that sit *inside* an item name.

    This is what keeps the "almond" of "almond croissant" from becoming milk.
    """
    hits: list[tuple[Span, str]] = []
    for word, value in words.items():
        for match in re.finditer(rf"\b{re.escape(word)}\b", lowered):
            if not any(_overlap(match.span(), s) for s in item_spans):
                hits.append((match.span(), value))
    hits.sort(key=lambda h: h[0][0])
    return hits


def _take(hits: list[tuple[Span, str]], window: Span, used: set[Span]) -> str | None:
    """Consume the last unused modifier lying in `window` (the gap before an item)."""
    for span, value in reversed(hits):
        if span in used:
            continue
        if window[0] <= span[0] and span[1] <= window[1]:
            used.add(span)
            return value
    return None


def _quantity(fragment: str) -> int:
    if digits := re.findall(r"\b(\d{1,2})\b", fragment):
        qty = int(digits[-1])
    elif words := [w for w in re.findall(r"[a-z]+", fragment) if w in NUMBER_WORDS]:
        qty = NUMBER_WORDS[words[-1]]
    else:
        qty = 1
    return max(1, min(qty, 20))


def parse_order(text: str) -> list[CartItem]:
    """Pull cart items out of a sentence like 'two large oat lattes and a croissant'.

    Modifiers are matched to the item that follows them; anything left over
    (e.g. a trailing "with oat milk") becomes the default for items that got none.
    """
    lowered = text.lower()
    found = _find_items(lowered)
    if not found:
        return []

    item_spans = [span for span, _ in found]
    size_hits = _modifier_hits(lowered, SIZE_ALIASES, item_spans)
    milk_hits = _modifier_hits(lowered, {m: m for m in MILKS}, item_spans)
    used: set[Span] = set()

    drafts: list[dict] = []
    previous_end = 0
    for span, name in found:
        window = (previous_end, span[0])
        drafts.append(
            {
                "name": name,
                "qty": _quantity(lowered[window[0]:window[1]]),
                "size": _take(size_hits, window, used),
                "milk": _take(milk_hits, window, used),
            }
        )
        previous_end = span[1]

    # Modifiers nobody claimed -- e.g. "a latte with oat milk", where the milk
    # trails the item -- become the default for the rest of the order.
    spare_size = next((v for s, v in size_hits if s not in used), None)
    spare_milk = next((v for s, v in milk_hits if s not in used), None)

    items: list[CartItem] = []
    for draft in drafts:
        name = draft["name"]
        size = draft["size"] or spare_size or "medium"
        milk = draft["milk"] or spare_milk
        if name in FOOD:
            size, milk = "regular", None
        unit = _price(name, size, milk)
        items.append(
            CartItem(
                name=name,
                size=size,
                milk=milk,
                qty=draft["qty"],
                unit_price=unit,
                line_total=round(unit * draft["qty"], 2),
            )
        )

    # Fold duplicate lines ("a latte and another latte") into one.
    merged: dict[tuple, CartItem] = {}
    for item in items:
        key = (item["name"], item["size"], item["milk"])
        if key in merged:
            merged[key]["qty"] += item["qty"]
            merged[key]["line_total"] = round(merged[key]["unit_price"] * merged[key]["qty"], 2)
        else:
            merged[key] = item
    return list(merged.values())


def describe_item(item: CartItem) -> str:
    bits = [str(item["qty"]) + "x"]
    if item["size"] != "regular":
        bits.append(item["size"])
    if item["milk"]:
        bits.append(item["milk"] + " milk")
    bits.append(item["name"])
    return " ".join(bits) + f" (${item['line_total']:.2f})"


def cart_total(cart: list[CartItem]) -> float:
    return round(sum(i["line_total"] for i in cart), 2)


def render_cart(cart: list[CartItem]) -> str:
    if not cart:
        return "(empty)"
    lines = [f"  - {describe_item(i)}" for i in cart]
    lines.append(f"  TOTAL: ${cart_total(cart):.2f}")
    return "\n".join(lines)


def lookup_menu(text: str) -> str:
    """Answer a menu question: either about specific items, or the whole board."""
    lowered = text.lower()
    hits = [n for n in _ITEM_PATTERNS if re.search(rf"\b{re.escape(n)}s?\b", lowered)]
    if hits:
        # Drop names that are substrings of a longer hit ("latte" inside "chai latte").
        hits = [h for h in hits if not any(h != o and h in o for o in hits)]
        lines = []
        for name in hits:
            base = MENU[name]
            if name in DRINKS:
                lines.append(
                    f"{name}: ${base:.2f} small / ${base + SIZES['medium']:.2f} medium "
                    f"/ ${base + SIZES['large']:.2f} large (+${MILK_SURCHARGE:.2f} for oat/almond/soy/coconut)"
                )
            else:
                lines.append(f"{name}: ${base:.2f}")
        return "\n".join(lines)

    drinks = ", ".join(f"{n} ${p:.2f}" for n, p in DRINKS.items())
    food = ", ".join(f"{n} ${p:.2f}" for n, p in FOOD.items())
    return f"DRINKS (small): {drinks}\nFOOD: {food}\nSizes: small / medium / large."
