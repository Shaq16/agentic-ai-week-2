"""Chat models. BrewBot runs fully offline by default; real providers are optional.

The graph never depends on which one is selected -- swapping the model changes the
prose, not the mechanics being demonstrated.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class DemoChatModel(BaseChatModel):
    """A deterministic, offline stand-in for a real chat model.

    It reads the tagged system prompt the graph builds ([TASK], [PROFILE], [CART],
    [CONTEXT]) and turns it into cafe-appropriate prose. Because it only ever sees
    the *trimmed* message list, it is a fair participant in the Topic 3 demo: shrink
    the budget far enough and it genuinely loses the earlier context.
    """

    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "brewbot-demo"

    # -- prompt parsing ----------------------------------------------------
    @staticmethod
    def _tag(system_text: str, tag: str) -> str:
        match = re.search(rf"^\[{tag}\](.*?)(?=^\[[A-Z_]+\]|\Z)", system_text, re.S | re.M)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _first_name(profile_line: str) -> str:
        match = re.search(r"name=([^;\n]+)", profile_line)
        return match.group(1).strip() if match else ""

    def _compose(self, messages: list[BaseMessage]) -> str:
        system_text = next(
            (str(m.content) for m in messages if isinstance(m, SystemMessage)), ""
        )
        last_user = next(
            (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        task = self._tag(system_text, "TASK") or "smalltalk"
        profile = self._tag(system_text, "PROFILE")
        context = self._tag(system_text, "CONTEXT")
        cart = self._tag(system_text, "CART")
        total = self._tag(system_text, "CART_TOTAL")
        name = self._first_name(profile)
        hey = f"{name}, " if name else ""
        # Deterministic but not repetitive: vary phrasing by the turn's content.
        pick = len(last_user) % 2

        if task == "order":
            added = context or "your order"
            opener = ["Perfect", "Lovely"][pick]
            reply = f"{opener}{', ' + name if name else ''} -- {added}"
            if total:
                reply += f" Your ticket is now {total}."
            return reply + " Anything else for you?"

        if task == "menu":
            lead = ["Here you go", "Sure thing"][pick]
            return f"{lead}{', ' + name if name else ''}:\n{context}\nWant me to start a ticket?"

        if task == "remember":
            return f"Noted{', ' + name if name else ''} -- {context} I'll keep that on file for next time."

        if task == "recall":
            return f"{context}"

        greeting = ["Morning", "Hi there"][pick]
        if name:
            body = f"{greeting}, {name}! Good to see you again."
            if "favorite_drink" in profile:
                drink = re.search(r"favorite_drink=([^;\n]+)", profile).group(1).strip()
                body += f" The usual {drink}?"
        else:
            body = f"{greeting}! Welcome to BrewBot."
            body += " I can read you the menu, take an order, or remember your usual."
        if cart and cart != "(empty)":
            body += f" You've still got a ticket open{' at ' + total if total else ''}."
        return body

    # -- BaseChatModel plumbing -------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._compose(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


# ------------------------------------------------------------------ providers

PROVIDERS: dict[str, dict[str, Any]] = {
    "demo": {
        "label": "DemoChatModel (offline)",
        "env": None,
        "models": ["brewbot-demo"],
    },
    "gemini": {
        "label": "Google Gemini",
        "env": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    },
    "openai": {
        "label": "OpenAI",
        "env": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4o"],
    },
}


def available_providers() -> list[str]:
    ready = ["demo"]
    for key, spec in PROVIDERS.items():
        if key == "demo":
            continue
        env = spec["env"]
        if env and os.environ.get(env):
            ready.append(key)
    return ready


def get_model(provider: str, model_name: str, temperature: float = 0.3) -> BaseChatModel:
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)
    return DemoChatModel()
