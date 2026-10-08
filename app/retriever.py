"""
Etapa "retrieve" do pipeline — função `buscar(consulta)`.

Aviso do checkpoint: "o pipeline do CKP02 será reutilizado como @tool no
CKP03 — construa modular com uma função buscar(consulta) clara." É
exatamente isso que este módulo entrega: uma função pura, com assinatura
simples, que no CKP03 poderá ser decorada com @tool do LangChain sem
nenhuma mudança na lógica interna.

Inclui os dois diferenciais de recuperação do checkpoint:
    - metadata filtering (parâmetro `filtro_metadata`, usando `where` do Chroma)
    - reranking com cross-encoder (parâmetro `usar_reranking`)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import CONFIGURACAO_PADRAO, TOP_K_PADRAO, TOP_K_RERANKING
from app.reranker import rerankear
from app.vectorstore import carregar_vectorstore


def buscar(
    consulta: str,
    embeddings: Embeddings,
    k: int = TOP_K_PADRAO,
    config_chunking: str = CONFIGURACAO_PADRAO,
    filtro_metadata: Optional[Dict] = None,
    usar_reranking: bool = False,
) -> List[Document]:
    """Busca os `k` chunks mais relevantes para `consulta` na base indexada.

    Parâmetros:
        consulta: texto da busca (já deve vir contextualizado — ver
            app/context_manager.py para perguntas de seguimento).
        embeddings: instância de Embeddings (ver app/embeddings.py).
        k: quantos chunks retornar no final.
        config_chunking: qual configuração de chunking usar (cada uma tem
            sua própria coleção no Chroma — ver app/vectorstore.py).
        filtro_metadata: filtro opcional no formato `where` do Chroma, ex:
            {"documento_fonte": "doc3_recomendacoes_oms_dieta_saudavel.md"}
            (diferencial: metadata filtering).
        usar_reranking: se True, recupera mais candidatos e reordena com
            cross-encoder antes de cortar para `k` (diferencial: reranking).

    Retorna uma lista de `Document` do LangChain, cada um com metadata
    contendo chunk_id, documento_fonte, fonte_url, pagina — usados depois
    para montar as Citacoes da resposta (ver app/generation.py).
    """
    vectorstore = carregar_vectorstore(embeddings, config_chunking=config_chunking)

    k_busca = TOP_K_RERANKING if usar_reranking else k
    documentos = vectorstore.similarity_search(consulta, k=k_busca, filter=filtro_metadata)

    if usar_reranking:
        documentos_com_score = rerankear(consulta, documentos, top_k=k)
        return [doc for doc, _score in documentos_com_score]

    return documentos[:k]


def buscar_unindo(
    consultas: List[str],
    embeddings: Embeddings,
    k: int = TOP_K_PADRAO,
    config_chunking: str = CONFIGURACAO_PADRAO,
    usar_reranking: bool = False,
) -> List[Document]:
    """Busca com mais de uma consulta (ex: pergunta original + versão reescrita
    pelo context_manager) e une os resultados sem duplicar chunks.

    Evita que uma reescrita ruim de uma pergunta de seguimento faça o chunk
    certo ficar de fora: a pergunta original sempre participa da busca."""
    vistos, unidos = set(), []
    for consulta in dict.fromkeys(c.strip() for c in consultas if c and c.strip()):
        for doc in buscar(consulta, embeddings, k=k, config_chunking=config_chunking, usar_reranking=usar_reranking):
            chave = doc.metadata.get("chunk_id")
            if chave not in vistos:
                vistos.add(chave)
                unidos.append(doc)
    return unidos[: k + 2]


if __name__ == "__main__":
    from dotenv import load_dotenv

    from app.embeddings import get_embeddings

    load_dotenv()
    emb = get_embeddings()
    resultados = buscar("quanta água devo beber por dia?", embeddings=emb)
    for doc in resultados:
        print("-", doc.metadata["documento_fonte"], "|", doc.metadata["chunk_id"])
        print("  ", doc.page_content[:120].replace("\n", " "), "...")
