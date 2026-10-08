"""
Etapa "embed" do pipeline — OllamaEmbeddings com nomic-embed-text, único
modelo de embedding aprovado pelo checkpoint.

Dois caminhos possíveis para chegar ao MESMO modelo (nomic-embed-text):

  - "cloud": API hospedada em https://ollama.com, autenticada com OLLAMA_API_KEY.
    Atenção: dependendo da conta/chave, o Ollama Cloud pode responder 401 para
    o endpoint de embeddings (/api/embed) mesmo com a chave válida para chat.
  - "local": servidor Ollama instalado na máquina (http://localhost:11434),
    com `ollama pull nomic-embed-text`. Não precisa de chave.

Controle via variável de ambiente EMBEDDING_BACKEND no .env:
    auto  (padrão) -> tenta cloud; se falhar, tenta local; se ambos falharem,
                      levanta um erro explicando o que fazer.
    cloud -> só cloud.
    local -> só local.

Como o modelo é o mesmo nos dois casos, os vetores são equivalentes: não há
problema em indexar com um backend e consultar com o outro.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from typing import List

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.config import (
    MODELO_EMBEDDING,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_LOCAL_BASE_URL,
    obter_api_key,
)


class EmbeddingsNomic(Embeddings):
    """Aplica os prefixos que o nomic-embed-text exige para boa qualidade de
    busca: 'search_document: ' ao indexar e 'search_query: ' ao consultar.
    Sem eles a recuperação em português fica visivelmente pior."""

    def __init__(self, base: OllamaEmbeddings):
        self.base = base
        self.base_url = base.base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.base.embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self.base.embed_query(f"search_query: {text}")


def _criar_cloud(api_key: str) -> OllamaEmbeddings:
    # OllamaEmbeddings não aceita `headers` direto (ao contrário do ChatOllama):
    # o header vai em `client_kwargs`, repassado ao ollama.Client por baixo.
    return OllamaEmbeddings(
        model=MODELO_EMBEDDING,
        base_url=OLLAMA_CLOUD_BASE_URL,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    )


def _criar_local() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=OLLAMA_LOCAL_BASE_URL)


def _testar(emb: OllamaEmbeddings) -> Tuple[bool, str]:
    """Faz uma chamada mínima para saber se o backend realmente responde."""
    try:
        emb.embed_query("teste de conexão")
        return True, ""
    except Exception as exc:  # 401, conexão recusada, modelo não encontrado...
        return False, f"{type(exc).__name__}: {exc}"


def get_embeddings(backend: Optional[str] = None) -> EmbeddingsNomic:
    """Devolve um OllamaEmbeddings funcional (nomic-embed-text)."""
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "auto")).strip().lower()
    api_key = obter_api_key()
    erros = []

    if backend in ("cloud", "auto"):
        if api_key:
            emb = _criar_cloud(api_key)
            ok, erro = _testar(emb)
            if ok:
                return EmbeddingsNomic(emb)
            erros.append(f"cloud: {erro}")
        else:
            erros.append("cloud: OLLAMA_API_KEY não encontrada no .env")

    if backend in ("local", "auto"):
        emb = _criar_local()
        ok, erro = _testar(emb)
        if ok:
            if backend == "auto":
                print(
                    "Aviso: embeddings via Ollama Cloud indisponíveis; usando "
                    "nomic-embed-text no Ollama LOCAL (mesmo modelo)."
                )
            return EmbeddingsNomic(emb)
        erros.append(f"local: {erro}")

    raise RuntimeError(
        "Não foi possível obter embeddings com nomic-embed-text.\n  - "
        + "\n  - ".join(erros)
        + "\n\nComo resolver: rode `python -m app.diagnostico` para ver qual parte falha. "
        "Alternativa mais simples: instale o Ollama (https://ollama.com/download), "
        "rode `ollama pull nomic-embed-text` e use EMBEDDING_BACKEND=local no .env."
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    vetor = get_embeddings().embed_query("exemplo de consulta sobre alimentação saudável")
    print(f"Dimensão do embedding retornado: {len(vetor)}")
