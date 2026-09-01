from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


ModelInput = str | BaseChatModel


def create_chat_model(
    model: ModelInput | None = None,
    *,
    temperature: float = 0,
) -> BaseChatModel:
    """Resolve an injected model or initialize one from a LangChain identifier.

    Identifiers use LangChain's ``provider:model`` form. The default is the
    locally served ``ollama:llama3.2`` model. Any injected ``BaseChatModel``
    remains supported.
    """
    if isinstance(model, BaseChatModel):
        return model

    identifier = model or os.getenv("AGENT_MODEL", "ollama:llama3.2")
    if not identifier.strip():
        raise ValueError("A model identifier must be provided via AGENT_MODEL or model=")

    return init_chat_model(identifier, temperature=temperature)
