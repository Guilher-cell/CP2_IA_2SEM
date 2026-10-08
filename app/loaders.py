"""
Carregamento dos documentos reais da base de conhecimento (etapa "load" do
pipeline load -> split -> embed -> store -> retrieve -> generate).

Suporta .md, .txt e .pdf (conforme exigido pelo checkpoint). Para .md/.txt,
extrai metadados de um cabeçalho simples no topo do arquivo (linhas
"Fonte:", "URL:", "Categoria:") — é o formato usado pelos documentos em
data/docs/. Para .pdf, tenta extrair metadados equivalentes das primeiras
linhas do texto extraído, com fallback para valores genéricos.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.config import DIRETORIO_DOCUMENTOS
from app.schemas import DocumentoFonte

_PADRAO_CABECALHO = re.compile(
    r"^(Fonte|URL|Categoria|Rotulo|Pais|Resumo adaptado por):\s*(.+)$", re.MULTILINE
)


def _extrair_metadados_cabecalho(texto: str) -> dict:
    """Lê o cabeçalho tipo 'Fonte: ... / URL: ... / Categoria: ...' do topo do arquivo."""
    metadados = {}
    for chave, valor in _PADRAO_CABECALHO.findall(texto[:2000]):
        metadados[chave.strip().lower()] = valor.strip()
    return metadados


def _carregar_md_ou_txt(caminho: Path) -> Document:
    texto_completo = caminho.read_text(encoding="utf-8")
    metadados_cabecalho = _extrair_metadados_cabecalho(texto_completo)

    fonte_url = metadados_cabecalho.get("url", "")
    titulo = metadados_cabecalho.get("fonte", caminho.stem)
    categoria = metadados_cabecalho.get("categoria", "diretriz_oficial")

    # Valida o documento via Pydantic ANTES de aceitá-lo no pipeline — um
    # documento sem fonte_url válida levanta erro aqui, não silenciosamente
    # mais adiante no RAG.
    DocumentoFonte(arquivo=caminho.name, titulo=titulo, fonte_url=fonte_url, categoria=categoria)

    return Document(
        page_content=texto_completo,
        metadata={
            "arquivo": caminho.name,
            "titulo": titulo,
            "fonte_url": fonte_url,
            "categoria": categoria,
            "rotulo": metadados_cabecalho.get("rotulo", ""),
        },
    )


def _carregar_pdf(caminho: Path) -> List[Document]:
    """Carrega um PDF página a página, preservando o número de página no metadata."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "Carregar PDFs requer o pacote 'pypdf' (já incluído em requirements.txt). "
            "Rode: pip install pypdf"
        ) from exc

    leitor = PdfReader(str(caminho))
    primeira_pagina_texto = leitor.pages[0].extract_text() or "" if leitor.pages else ""
    metadados_cabecalho = _extrair_metadados_cabecalho(primeira_pagina_texto)
    fonte_url = metadados_cabecalho.get("url", "")
    titulo = metadados_cabecalho.get("fonte", caminho.stem)
    categoria = metadados_cabecalho.get("categoria", "diretriz_oficial")

    if not fonte_url:
        raise ValueError(
            f"O PDF '{caminho.name}' não tem uma linha 'URL: ...' nas primeiras linhas "
            "do texto extraído. Adicione a fonte de origem antes de incluir o PDF na base."
        )

    documentos = []
    for num_pagina, pagina in enumerate(leitor.pages, start=1):
        texto_pagina = pagina.extract_text() or ""
        if not texto_pagina.strip():
            continue
        documentos.append(
            Document(
                page_content=texto_pagina,
                metadata={
                    "arquivo": caminho.name,
                    "titulo": titulo,
                    "fonte_url": fonte_url,
                    "categoria": categoria,
                    "pagina": num_pagina,
                    "rotulo": metadados_cabecalho.get("rotulo", ""),
                },
            )
        )
    return documentos


def carregar_documentos(diretorio: Path = DIRETORIO_DOCUMENTOS) -> List[Document]:
    """Carrega todos os documentos suportados (.md, .txt, .pdf) de um diretório.

    Levanta erro se nenhum documento for encontrado, ou se o total for menor
    que 5 (requisito mínimo do checkpoint) — essa segunda checagem só é um
    aviso (warning impresso), não um erro bloqueante, para permitir testes
    incrementais durante o desenvolvimento (Aula 05 permite começar com 2-3).
    """
    diretorio = Path(diretorio)
    arquivos = sorted(
        [p for p in diretorio.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"}]
    )

    if not arquivos:
        raise FileNotFoundError(f"Nenhum documento encontrado em {diretorio}.")

    documentos: List[Document] = []
    for caminho in arquivos:
        if caminho.suffix.lower() in {".md", ".txt"}:
            documentos.append(_carregar_md_ou_txt(caminho))
        elif caminho.suffix.lower() == ".pdf":
            documentos.extend(_carregar_pdf(caminho))

    arquivos_unicos = {d.metadata["arquivo"] for d in documentos}
    if len(arquivos_unicos) < 5:
        print(
            f"Aviso: apenas {len(arquivos_unicos)} documento(s) único(s) na base. "
            "O checkpoint exige >= 5 documentos reais até a entrega final."
        )

    return documentos


if __name__ == "__main__":
    docs = carregar_documentos()
    print(f"{len(docs)} 'documentos' (ou páginas) carregados de {len({d.metadata['arquivo'] for d in docs})} arquivo(s).")
    for d in docs[:3]:
        print("-", d.metadata["arquivo"], "|", d.metadata["titulo"], "|", d.metadata["fonte_url"])
