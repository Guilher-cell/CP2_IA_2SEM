"""
Reranking com cross-encoder (diferencial, +0,5 na rubrica do CKP02).

Usa cross-encoder/ms-marco-MiniLM-L-6-v2 (biblioteca sentence-transformers),
igual ao exemplo da Aula 07, para reordenar os chunks recuperados pela busca
vetorial antes de passar ao LLM — a busca vetorial traz TOP_K_RERANKING
candidatos (mais que o necessário) e o cross-encoder reordena por relevância
real à pergunta, cortando para o TOP_K final.

Este módulo é opcional: sentence-transformers é uma dependência pesada
(baixa um modelo na primeira execução). Se não estiver instalada, o restante
do pipeline (retriever.py) continua funcionando normalmente sem reranking —
ver o try/except em retriever.py.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

_MODELO_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder_cache = {"modelo": None, "indisponivel": False}


def _carregar_cross_encoder():
    if _cross_encoder_cache["modelo"] is None and not _cross_encoder_cache["indisponivel"]:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder_cache["modelo"] = CrossEncoder(_MODELO_RERANKER)
        except Exception:
            _cross_encoder_cache["indisponivel"] = True
    return _cross_encoder_cache["modelo"]


def reranking_disponivel() -> bool:
    """Checa se o reranker consegue ser carregado neste ambiente, sem lançar erro."""
    return _carregar_cross_encoder() is not None


def rerankear(
    pergunta: str, documentos: List[Document], top_k: int
) -> List[Tuple[Document, float]]:
    """Reordena `documentos` pela relevância do cross-encoder à `pergunta` e
    retorna os `top_k` melhores, junto com o score de relevância.

    Se o cross-encoder não estiver disponível (sentence-transformers não
    instalado, ou falha ao baixar o modelo), devolve os documentos na ordem
    original (da busca vetorial), com score None substituído por 0.0, para
    que o pipeline continue funcionando sem o diferencial.
    """
    modelo = _carregar_cross_encoder()
    if modelo is None:
        return [(doc, 0.0) for doc in documentos[:top_k]]

    pares = [(pergunta, doc.page_content) for doc in documentos]
    scores = modelo.predict(pares)
    documentos_com_score = list(zip(documentos, (float(s) for s in scores)))
    documentos_com_score.sort(key=lambda par: par[1], reverse=True)
    return documentos_com_score[:top_k]
