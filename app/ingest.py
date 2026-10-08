"""
Script de ingestão: roda load -> split -> embed -> store para TODAS as
configurações de chunking definidas em app/config.py, criando uma coleção
Chroma por configuração (necessário para comparar o retrieval de cada uma
com RAGAS — ver app/evaluation.py).

Rodar sempre que:
    - novos documentos forem adicionados em data/docs/ (ver README.md), ou
    - as configurações de chunking em app/config.py mudarem.

Uso:
    python -m app.ingest
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from app.chunking import gerar_chunks_todas_configuracoes  # noqa: E402
from app.config import CONFIGURACOES_CHUNKING  # noqa: E402
from app.embeddings import get_embeddings  # noqa: E402
from app.loaders import carregar_documentos  # noqa: E402
from app.vectorstore import construir_vectorstore  # noqa: E402


def ingerir_tudo() -> None:
    print("Carregando documentos de data/docs/ ...")
    documentos = carregar_documentos()
    arquivos_unicos = sorted({d.metadata["arquivo"] for d in documentos})
    print(f"  {len(arquivos_unicos)} documento(s): {', '.join(arquivos_unicos)}")

    print("\nGerando chunks para cada configuração de chunking...")
    chunks_por_config = gerar_chunks_todas_configuracoes(documentos)
    for nome, chunks in chunks_por_config.items():
        print(f"  {nome}: {len(chunks)} chunks")

    print("\nCarregando embeddings (nomic-embed-text via Ollama Cloud)...")
    embeddings = get_embeddings()

    print("\nIndexando no ChromaDB (uma coleção por configuração)...")
    for nome in CONFIGURACOES_CHUNKING:
        chunks = chunks_por_config[nome]
        construir_vectorstore(chunks, embeddings, config_chunking=nome)
        print(f"  Coleção '{nome}' indexada com {len(chunks)} chunks.")

    print("\nIngestão concluída. Rode 'python -m app.main' para conversar com o DocMind RAG.")


if __name__ == "__main__":
    ingerir_tudo()
