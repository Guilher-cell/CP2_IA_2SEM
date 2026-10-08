"""
Compatibilidade do RAGAS com versões recentes do langchain-community.

Problema: o `ragas` (inclusive o 0.4.x) faz, em ragas/llms/base.py,
    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI
mas o langchain-community atual removeu essas classes do Vertex AI, e o
`import ragas` falha com ModuleNotFoundError — mesmo que o projeto nem use
Vertex AI. Aqui registramos classes vazias no lugar delas, só para o import
do ragas funcionar. Elas nunca são usadas (o juiz do RAGAS é o gemma4:cloud).

Se o import original funcionar (versões compatíveis), nada é alterado.
"""

from __future__ import annotations

import sys
import types


class _Stub:
    """Classe vazia que substitui classes removidas (Vertex AI)."""


def aplicar_compat_ragas() -> None:
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    except Exception:
        modulo = types.ModuleType("langchain_community.chat_models.vertexai")
        modulo.ChatVertexAI = type("ChatVertexAI", (_Stub,), {})
        sys.modules["langchain_community.chat_models.vertexai"] = modulo

    try:
        import langchain_community.llms as llms

        if not hasattr(llms, "VertexAI"):
            llms.VertexAI = type("VertexAI", (_Stub,), {})
    except Exception:
        pass
