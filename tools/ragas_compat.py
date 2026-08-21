"""Compatibility shims required before importing ragas."""

from __future__ import annotations

import sys
import types


def ensure_ragas_langchain_compat() -> None:
    """
    ragas 0.4.x imports ChatVertexAI from langchain_community at module
    import time. langchain-community 0.4.x removed that module, so inject a
    stub before ragas is imported.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: N801 - match upstream symbol name
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "ChatVertexAI is not available in this environment."
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module

    parent_name = "langchain_community.chat_models"
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, "vertexai", module)
