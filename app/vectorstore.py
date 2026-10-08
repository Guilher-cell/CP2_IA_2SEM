"""
Etapa "store" do pipeline — ChromaDB local (PersistentClient), com uma
coleção por configuração de chunking, para permitir comparar os resultados
de busca/retrieval entre as diferentes estratégias (ver app/evaluation.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import DIRETORIO_CHROMA, NOME_COLECAO_CHROMA
from app.schemas import Chunk


def _chunk_para_document(chunk: Chunk) -> Document:
    """Converte um Chunk (Pydantic, validado) em Document do LangChain, com
    todos os metadados necessários para citar a fonte depois (metadata
    filtering + proveniência na resposta)."""
    return Document(
        page_content=chunk.texto,
        metadata={
            "chunk_id": chunk.chunk_id,
            "documento_fonte": chunk.documento_fonte,
            "fonte_url": chunk.fonte_url,
            "pagina": chunk.pagina if chunk.pagina is not None else -1,
            "indice": chunk.indice,
            "chunk_size_config": chunk.chunk_size_config,
        },
    )


def nome_colecao_para(config_chunking: str) -> str:
    """Nome da coleção Chroma para uma configuração de chunking específica."""
    return f"{NOME_COLECAO_CHROMA}__{config_chunking}"


def construir_vectorstore(
    chunks: List[Chunk],
    embeddings: Embeddings,
    config_chunking: str,
    persist_directory: Path = DIRETORIO_CHROMA,
) -> Chroma:
    """Cria (ou recria) a coleção Chroma de uma configuração de chunking e
    indexa todos os chunks passados."""
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_directory))
    colecao = nome_colecao_para(config_chunking)

    # Remove a coleção antiga (se existir) para reindexar do zero — evita
    # chunks duplicados quando o script de ingestão é rodado mais de uma vez.
    try:
        client.delete_collection(colecao)
    except Exception:
        pass  # coleção ainda não existia, tudo bem

    vectorstore = Chroma(
        client=client,
        collection_name=colecao,
        embedding_function=embeddings,
    )

    documentos = [_chunk_para_document(c) for c in chunks]
    ids = [c.chunk_id for c in chunks]
    vectorstore.add_documents(documents=documentos, ids=ids)

    return vectorstore


def carregar_vectorstore(
    embeddings: Embeddings,
    config_chunking: str,
    persist_directory: Path = DIRETORIO_CHROMA,
) -> Chroma:
    """Abre uma coleção Chroma já existente (sem reindexar)."""
    client = chromadb.PersistentClient(path=str(persist_directory))
    colecao = nome_colecao_para(config_chunking)
    return Chroma(client=client, collection_name=colecao, embedding_function=embeddings)
