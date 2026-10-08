"""
DocMind RAG — entry point da aplicação (diferencial: interface Gradio).

Executar com:
    python -m app.main

Pré-requisito: rodar a ingestão antes, pelo menos uma vez:
    python -m app.ingest

A cada mensagem do usuário:
    1. app/context_manager.py reescreve perguntas de seguimento como
       perguntas autônomas (camada de contexto isolada, substitui a
       memória de chat do CKP01).
    2. app/retriever.py busca os chunks mais relevantes (com reranking,
       se o cross-encoder estiver disponível).
    3. app/generation.py gera a resposta citando a fonte de cada trecho
       usado.
"""

import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from app.config import CONFIGURACAO_PADRAO  # noqa: E402
from app.context_manager import ContextoRAG  # noqa: E402
from app.embeddings import get_embeddings  # noqa: E402
from app.generation import get_llm, responder  # noqa: E402
from app.reranker import reranking_disponivel  # noqa: E402
from app.retriever import buscar_unindo  # noqa: E402

_llm = get_llm(temperature=0.0)
_embeddings = get_embeddings()
_contexto = ContextoRAG(llm=_llm)
_usar_reranking = reranking_disponivel()


def _formatar_fontes(resposta) -> str:
    if not resposta.citacoes:
        return ""
    vistos = set()
    linhas = []
    for c in resposta.citacoes:
        chave = (c.arquivo, c.pagina)
        if chave in vistos:
            continue
        vistos.add(chave)
        sufixo_pagina = f", p. {c.pagina}" if c.pagina else ""
        linhas.append(f"- {c.arquivo}{sufixo_pagina}")
    return "\n\n**Fontes:**\n" + "\n".join(linhas)


def responder_chat(mensagem: str, historico: list) -> str:
    try:
        consulta = _contexto.preparar_consulta(mensagem)
        # A pergunta original sempre entra na busca, junto da versão reescrita.
        documentos = buscar_unindo(
            [mensagem, consulta],
            embeddings=_embeddings,
            config_chunking=CONFIGURACAO_PADRAO,
            usar_reranking=_usar_reranking,
        )
        resposta = responder(mensagem, documentos, _llm)
        # Respostas "não encontrei" não entram no histórico: poluiriam a
        # contextualização das próximas perguntas.
        if resposta.fundamentado_em_documentos:
            _contexto.registrar_turno(mensagem, resposta.resposta)
        return resposta.resposta + _formatar_fontes(resposta)
    except Exception as exc:  # erro de chamada ao modelo / embeddings
        return f"[erro ao processar sua pergunta: {exc}]"


with gr.Blocks(title="DocMind RAG — Nutrição (CKP02)") as demo:
    gr.Markdown(
        "# 📚🥗 DocMind RAG — Nutrição\n"
        "Pipeline RAG sobre uma base real de diretrizes de nutrição (Ministério da "
        "Saúde, OMS, Anvisa, EFSA). Toda resposta cita o(s) documento(s) de origem. "
        "**Não substitui uma consulta com nutricionista ou médico.**\n\n"
        f"_Reranking com cross-encoder: {'ativado' if _usar_reranking else 'indisponível neste ambiente'}._"
    )

    gr.ChatInterface(
        fn=responder_chat,
        chatbot=gr.Chatbot(label="DocMind RAG", height=460),
        textbox=gr.Textbox(
            placeholder="Ex: qual o limite recomendado de açúcar por dia?",
            label="Sua pergunta",
        ),
    )


if __name__ == "__main__":
    if not os.getenv("OLLAMA_API_KEY"):
        print(
            "Aviso: OLLAMA_API_KEY não encontrada. Copie .env.example para "
            ".env e preencha sua chave do Ollama Cloud antes de conversar."
        )
    demo.launch(server_name="127.0.0.1", server_port=7861)
