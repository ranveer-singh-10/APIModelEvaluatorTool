"""Provider registry.

Each provider module exposes the same interface: ``MODEL_PRICING``, ``chat()``, ``text_from_response()``,
``stop_reason()``, ``usage_summary()`` and ``user_content()``. A module may also define ``configured()``;
``default_models()`` leaves it out while that returns False (e.g. no API key).
"""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType

from . import anthropic_api, openai_api, openrouter_api

PROVIDERS: dict[str, ModuleType] = {
    "anthropic": anthropic_api,
    "openai": openai_api,
    "openrouter": openrouter_api,
}


def provider_for(model: str) -> str:
    """Return the name of the provider that serves ``model``.

    A priced model belongs to the provider whose ``MODEL_PRICING`` lists it. Unpriced models still run (with no cost
    estimate) if the name says which provider they belong to. ``web/log_viewer.html`` copies these rules in
    ``inferProvider``, so keep the two in step.
    """
    for name, provider in PROVIDERS.items():
        if model in provider.MODEL_PRICING:
            return name
    # OpenRouter ids are "vendor/model", so a "/" is checked first: "openai/gpt-oss-120b" is OpenRouter's, not OpenAI's
    if "/" in model:
        return "openrouter"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(f"Can't tell which provider model '{model}' belongs to")


def is_configured(name: str) -> bool:
    check = getattr(PROVIDERS[name], "configured", None)
    return check() if check else True


def default_models(providers: Iterable[str] | None = None) -> list[str]:
    """Every priced model of the given providers.

    With none given, providers that aren't configured are left out, so default runs don't fail on them; naming a
    provider includes it either way (its calls then fail clearly).
    """
    names = list(providers) if providers else [name for name in PROVIDERS if is_configured(name)]
    unknown = [name for name in names if name not in PROVIDERS]
    if unknown:
        raise ValueError(f"Unknown provider(s): {', '.join(unknown)}. Choose from {', '.join(PROVIDERS)}")
    return [model for name in names for model in PROVIDERS[name].MODEL_PRICING]
