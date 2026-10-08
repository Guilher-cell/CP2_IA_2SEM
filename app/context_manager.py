"""
Camada de contexto isolada do DocMind RAG.

Correção em relação ao CKP01: no feedback daquele checkpoint ficou claro que
a "memória" (ConversationTokenBufferMemory) e a lógica de conversa estavam
misturadas na chain principal. Aqui isolamos toda a responsabilidade de
"entender a pergunta no contexto da conversa" em um único módulo, por trás
de uma interface pequena (GerenciadorDeContexto). Assim, se no futuro o
projeto trocar de estratégia de contexto (ex: voltar a usar memória bruta,
ou usar um banco de sessões), a troca é uma substituição de implementação
atrás da mesma interface — não uma reescrita do pipeline de geração.

No RAG, "contexto" não é mais o histórico de chat em si (isso o CKP01 já
cobriu); é a CONSULTA usada para buscar na base de conhecimento. Perguntas
de seguimento como "e para quem faz exercício?" só fazem sentido buscando
corretamente se forem reescritas como uma pergunta autônoma, usando o
histórico da conversa — é isso que este módulo faz.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

_PROMPT_CONDENSACAO = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você reescreve perguntas de seguimento como perguntas autônomas, "
            "usando o histórico da conversa para preencher o que falta "
            "(ex: 'e para quem pratica exercício?' vira 'quais recomendações "
            "nutricionais existem para quem pratica exercício?'). "
            "Se a pergunta de seguimento JÁ faz sentido sozinha (trata de outro assunto ou "
            "não depende do histórico), devolva-a EXATAMENTE como está, sem alterar. "
            "Responda APENAS com a pergunta, sem explicações.",
        ),
        ("human", "Histórico da conversa:\n{historico}\n\nPergunta de seguimento: {pergunta}\n\nPergunta autônoma:"),
    ]
)


class GerenciadorDeContexto(ABC):
    """Interface mínima que qualquer estratégia de contexto deve implementar."""

    @abstractmethod
    def preparar_consulta(self, pergunta: str) -> str:
        """Recebe a pergunta crua do usuário e devolve a consulta que deve
        ser usada para buscar na base de conhecimento (retriever)."""

    @abstractmethod
    def registrar_turno(self, pergunta: str, resposta: str) -> None:
        """Registra um turno de conversa concluído, para contextualizar os
        próximos turnos."""

    @abstractmethod
    def historico(self) -> List[Tuple[str, str]]:
        """Retorna o histórico de turnos (pergunta, resposta) registrados até agora."""


@dataclass
class ContextoRAG(GerenciadorDeContexto):
    """Implementação padrão: reescreve perguntas de seguimento usando o LLM.

    Mantém uma lista simples de turnos em memória (não usa as classes de
    memória do LangChain do CKP01 — aqui o objetivo é só contextualizar a
    busca, não gerar a resposta final).
    """

    llm: BaseChatModel
    max_turnos_no_historico: int = 4
    _turnos: List[Tuple[str, str]] = field(default_factory=list)

    def preparar_consulta(self, pergunta: str) -> str:
        if not self._turnos:
            return pergunta

        historico_formatado = "\n".join(
            f"Usuário: {p}\nAssistente: {r}" for p, r in self._turnos[-self.max_turnos_no_historico :]
        )
        try:
            mensagem = _PROMPT_CONDENSACAO.invoke({"historico": historico_formatado, "pergunta": pergunta})
            resposta = self.llm.invoke(mensagem)
            consulta_reescrita = resposta.content.strip()
            return consulta_reescrita if consulta_reescrita else pergunta
        except Exception:
            # Se a reescrita falhar por qualquer motivo (ex: erro de rede),
            # caímos de volta para a pergunta original — a busca ainda
            # funciona, só perde um pouco de contexto de seguimento.
            return pergunta

    def registrar_turno(self, pergunta: str, resposta: str) -> None:
        self._turnos.append((pergunta, resposta))

    def historico(self) -> List[Tuple[str, str]]:
        return list(self._turnos)

    def limpar(self) -> None:
        self._turnos.clear()
