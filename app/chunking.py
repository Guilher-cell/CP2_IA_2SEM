"""
Etapa "split" do pipeline: divide os documentos carregados em chunks, usando
RecursiveCharacterTextSplitter com os separadores exigidos pelo checkpoint
(["\\n\\n", "\\n", ". ", " ", ""]) e chunk_overlap de 10-15% do chunk_size.

Implementa >= 2 configurações de chunk_size (ver app/config.py) para a
comparação quantitativa via RAGAS exigida pelo CKP02.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CONFIGURACOES_CHUNKING, SEPARADORES_SPLITTER
from app.schemas import Chunk


def gerar_chunks(
    documentos: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Chunk]:
    """Aplica o RecursiveCharacterTextSplitter e devolve Chunks validados (Pydantic)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARADORES_SPLITTER,
    )

    chunks: List[Chunk] = []
    # Agrupamos por arquivo para numerar o índice de cada chunk dentro do seu
    # próprio documento de origem (mais legível do que um índice global).
    contador_por_arquivo: Dict[str, int] = {}

    for doc in documentos:
        arquivo = doc.metadata["arquivo"]
        pedacos_texto = splitter.split_text(doc.page_content)
        rotulo = doc.metadata.get("rotulo", "")
        for texto in pedacos_texto:
            # Chunk contextual: um chunk isolado como "Açúcares livres: devem
            # representar menos de 10%..." não diz de qual fonte vem (OMS?). O
            # prefixo [rótulo] entra no texto embedado e melhora a busca por
            # perguntas que citam a fonte ("segundo a OMS"). Aumenta cada chunk
            # em ~40 caracteres, além do chunk_size do splitter.
            if rotulo:
                texto = f"[{rotulo}] {texto}"
            indice = contador_por_arquivo.get(arquivo, 0)
            chunk = Chunk(
                chunk_id=f"{arquivo}::{chunk_size}::{indice}",
                texto=texto,
                documento_fonte=arquivo,
                fonte_url=doc.metadata.get("fonte_url", ""),
                pagina=doc.metadata.get("pagina"),
                indice=indice,
                chunk_size_config=chunk_size,
            )
            chunks.append(chunk)
            contador_por_arquivo[arquivo] = indice + 1

    return chunks


def gerar_chunks_todas_configuracoes(
    documentos: List[Document],
    configuracoes: Dict[str, dict] = CONFIGURACOES_CHUNKING,
) -> Dict[str, List[Chunk]]:
    """Roda gerar_chunks para cada configuração definida em app/config.py.

    Retorna um dict {nome_configuracao: [Chunk, ...]}, pronto para ser
    comparado via RAGAS em app/evaluation.py.
    """
    resultado = {}
    for nome, params in configuracoes.items():
        resultado[nome] = gerar_chunks(
            documentos, chunk_size=params["chunk_size"], chunk_overlap=params["chunk_overlap"]
        )
    return resultado


if __name__ == "__main__":
    from app.loaders import carregar_documentos

    docs = carregar_documentos()
    todas = gerar_chunks_todas_configuracoes(docs)
    for nome, chunks in todas.items():
        tamanhos = [len(c.texto) for c in chunks]
        media = sum(tamanhos) / len(tamanhos) if tamanhos else 0
        print(f"{nome}: {len(chunks)} chunks | tamanho médio: {media:.0f} caracteres")
