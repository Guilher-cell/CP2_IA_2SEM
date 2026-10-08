"""
Etapa "generate" do pipeline — chama o gemma4:cloud (temperature=0, conforme
exigido pelo checkpoint para respostas fundamentadas) sobre os chunks
recuperados e monta uma RespostaRAG validada (Pydantic), com citação de
fonte (arquivo + página + trecho) para cada afirmação relevante.
"""

from __future__ import annotations

import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import MODELO_CHAT, OLLAMA_CLOUD_BASE_URL, obter_api_key
from app.schemas import Citacao, RespostaRAG

_MARCADOR_SEM_RESPOSTA = "não encontrei essa informação nos documentos disponíveis"

_PROMPT_RAG = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é o NutriAssist, um assistente de orientação nutricional geral que "
            "responde SOMENTE com base nos trechos de documentos fornecidos abaixo. "
            "Cada trecho tem um número entre colchetes, ex: [1], [2]. "
            "Ao usar uma informação de um trecho na sua resposta, cite o número "
            "correspondente entre colchetes logo após a afirmação, ex: 'o sal deve "
            "ser limitado a 5g por dia [2]'. "
            f"Procure a resposta em TODOS os trechos, mesmo que a informação apareça de forma "
            f"parcial ou com palavras diferentes das da pergunta (ex: 'açúcar livre' = 'açúcares livres'). "
            f"Somente se nenhum trecho contiver a informação, responda exatamente: "
            f"'{_MARCADOR_SEM_RESPOSTA}'. Não invente informações fora dos trechos. "
            "Você não substitui uma consulta com nutricionista ou médico.\n\n"
            "Trechos disponíveis:\n{contexto}",
        ),
        ("human", "{pergunta}"),
    ]
)


def get_llm(temperature: float = 0.0) -> ChatOllama:
    """ChatOllama apontando para o Ollama Cloud, temperature=0 por padrão
    (exigido pelo checkpoint para respostas fundamentadas e reprodutíveis)."""
    api_key = obter_api_key()
    if not api_key:
        raise EnvironmentError(
            "OLLAMA_API_KEY não encontrada no ambiente. Copie .env.example para "
            ".env e preencha sua chave do Ollama Cloud."
        )
    return ChatOllama(
        model=MODELO_CHAT,
        base_url=OLLAMA_CLOUD_BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        temperature=temperature,
    )


def _formatar_contexto(documentos: List[Document]) -> str:
    partes = []
    for i, doc in enumerate(documentos, start=1):
        pagina = doc.metadata.get("pagina")
        sufixo_pagina = f", página {pagina}" if pagina and pagina > 0 else ""
        partes.append(
            f"[{i}] (fonte: {doc.metadata.get('documento_fonte')}{sufixo_pagina})\n{doc.page_content}"
        )
    return "\n\n".join(partes)


def _extrair_citacoes(texto_resposta: str, documentos: List[Document]) -> List[Citacao]:
    """Lê os marcadores [n] presentes na resposta e monta as Citacoes
    correspondentes a partir dos documentos recuperados."""
    indices_citados = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", texto_resposta)))
    citacoes = []
    for indice in indices_citados:
        if 1 <= indice <= len(documentos):
            doc = documentos[indice - 1]
            pagina = doc.metadata.get("pagina")
            citacoes.append(
                Citacao(
                    arquivo=doc.metadata.get("documento_fonte", "desconhecido"),
                    pagina=pagina if pagina and pagina > 0 else None,
                    chunk_id=doc.metadata.get("chunk_id", "desconhecido"),
                    trecho=doc.page_content[:300],
                )
            )
    return citacoes


def responder(pergunta: str, documentos_recuperados: List[Document], llm: ChatOllama) -> RespostaRAG:
    """Gera a resposta final (RespostaRAG validada) a partir da pergunta e
    dos chunks já recuperados (ver app/retriever.py)."""
    contexto = _formatar_contexto(documentos_recuperados)
    mensagem = _PROMPT_RAG.invoke({"contexto": contexto, "pergunta": pergunta})
    saida = llm.invoke(mensagem)
    texto_resposta = saida.content.strip()

    sem_resposta = _MARCADOR_SEM_RESPOSTA in texto_resposta.lower()

    if sem_resposta:
        return RespostaRAG(
            pergunta=pergunta,
            resposta=texto_resposta,
            citacoes=[],
            confianca=0.1,
            fundamentado_em_documentos=False,
        )

    citacoes = _extrair_citacoes(texto_resposta, documentos_recuperados)

    if not citacoes:
        # O modelo respondeu de forma substantiva mas sem marcar [n] — melhor
        # esforço: citamos os documentos recuperados como proveniência e
        # reduzimos a confiança, em vez de aceitar uma resposta "fundamentada"
        # sem nenhuma citação (o schema não permite isso — ver model_validator
        # em app/schemas.py).
        citacoes = [
            Citacao(
                arquivo=doc.metadata.get("documento_fonte", "desconhecido"),
                pagina=doc.metadata.get("pagina") if doc.metadata.get("pagina", -1) > 0 else None,
                chunk_id=doc.metadata.get("chunk_id", "desconhecido"),
                trecho=doc.page_content[:300],
            )
            for doc in documentos_recuperados[:2]
        ]
        confianca = 0.5
    else:
        confianca = min(1.0, 0.6 + 0.1 * len(citacoes))

    return RespostaRAG(
        pergunta=pergunta,
        resposta=texto_resposta,
        citacoes=citacoes,
        confianca=confianca,
        fundamentado_em_documentos=True,
    )
