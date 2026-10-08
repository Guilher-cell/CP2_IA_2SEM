"""
Avaliação RAGAS (faithfulness + answer_relevancy) comparando as >= 2
configurações de chunking exigidas pelo checkpoint.

Nota de compatibilidade: a API pública do RAGAS mudou algumas vezes entre
2024 e 2026. Este módulo tenta primeiro a API "moderna" (classes de métrica
+ LangchainLLMWrapper/LangchainEmbeddingsWrapper + EvaluationDataset), usada
pelas versões mais recentes (ragas >= 0.2), e cai para a API "clássica"
(métricas como objetos prontos, ex: `faithfulness`, `answer_relevancy`) se a
primeira não estiver disponível — o mesmo padrão de try/except já usado em
app/chain.py do CKP01 para o ConversationChain.

Uso:
    python -m app.evaluation
"""

from __future__ import annotations

from typing import Dict, List

import math

import pandas as pd
from dotenv import load_dotenv

from app.compat_ragas import aplicar_compat_ragas

from app.config import CONFIGURACOES_CHUNKING, LIMIAR_FAITHFULNESS_APROVACAO
from app.schemas import ResultadoRAGAS

load_dotenv()

# >= 5 perguntas de teste exigidas pelo checkpoint, cobrindo os 5 documentos
# da base (uma pergunta por documento, para garantir cobertura).
PERGUNTAS_TESTE: List[str] = [
    "O que é a classificação NOVA de alimentos segundo o Guia Alimentar Brasileiro?",
    "O que significa o selo de lupa 'ALTO EM' na rotulagem nutricional?",
    "Qual o limite recomendado pela OMS para consumo de açúcares livres?",
    "Quantos litros de água por dia a EFSA recomenda para homens e mulheres adultos?",
    "Quanta proteína por quilo de peso corporal uma pessoa fisicamente ativa precisa?",
]


def _montar_amostras(perguntas: List[str], embeddings, llm, config_chunking: str):
    """Roda o pipeline retrieve+generate para cada pergunta e monta as
    amostras no formato esperado pelo RAGAS (user_input, response,
    retrieved_contexts)."""
    from app.generation import responder
    from app.retriever import buscar

    amostras = []
    for pergunta in perguntas:
        documentos = buscar(pergunta, embeddings=embeddings, config_chunking=config_chunking)
        resposta = responder(pergunta, documentos, llm)
        amostras.append(
            {
                "user_input": pergunta,
                "response": resposta.resposta,
                "retrieved_contexts": [d.page_content for d in documentos],
            }
        )
    return amostras


def _avaliar_amostras_api_moderna(amostras, llm, embeddings) -> pd.DataFrame:
    aplicar_compat_ragas()
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy as MetricaRelevancia
    except ImportError:
        from ragas.metrics import AnswerRelevancy as MetricaRelevancia
        from ragas.metrics import Faithfulness

    juiz_llm = LangchainLLMWrapper(llm)
    juiz_embeddings = LangchainEmbeddingsWrapper(embeddings)

    dataset = EvaluationDataset.from_list(amostras)
    metricas = [Faithfulness(llm=juiz_llm), MetricaRelevancia(llm=juiz_llm, embeddings=juiz_embeddings)]
    # raise_exceptions=False: se o juiz errar o formato numa amostra, essa nota
    # vira NaN em vez de derrubar a avaliação inteira. Poucos workers e timeout
    # alto evitam estourar o limite de requisições do Ollama Cloud.
    resultado = evaluate(
        dataset=dataset,
        metrics=metricas,
        run_config=RunConfig(max_workers=2, timeout=300),
        raise_exceptions=False,
    )
    return resultado.to_pandas()


def _avaliar_amostras_api_classica(amostras, llm, embeddings) -> pd.DataFrame:
    aplicar_compat_ragas()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    faithfulness.llm = LangchainLLMWrapper(llm)
    answer_relevancy.llm = LangchainLLMWrapper(llm)
    answer_relevancy.embeddings = embeddings

    dataset = Dataset.from_list(
        [
            {
                "question": a["user_input"],
                "answer": a["response"],
                "contexts": a["retrieved_contexts"],
            }
            for a in amostras
        ]
    )
    resultado = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return resultado.to_pandas()


def avaliar_configuracao(
    config_chunking: str, perguntas: List[str], embeddings, llm
) -> List[ResultadoRAGAS]:
    """Roda o pipeline + RAGAS para uma configuração de chunking e devolve
    uma lista de ResultadoRAGAS (um por pergunta), já validada via Pydantic."""
    amostras = _montar_amostras(perguntas, embeddings, llm, config_chunking)

    try:
        df = _avaliar_amostras_api_moderna(amostras, llm, embeddings)
    except ImportError:
        df = _avaliar_amostras_api_classica(amostras, llm, embeddings)

    coluna_relevancia = "answer_relevancy" if "answer_relevancy" in df.columns else df.columns[-1]
    def _nota(valor):
        # NaN = o juiz (LLM) não devolveu o formato esperado nessa amostra.
        valor = float(valor)
        return None if math.isnan(valor) else valor

    resultados = []
    for i, pergunta in enumerate(perguntas):
        r = ResultadoRAGAS(
            configuracao_chunking=config_chunking,
            pergunta=pergunta,
            faithfulness=_nota(df.iloc[i]["faithfulness"]),
            answer_relevancy=_nota(df.iloc[i][coluna_relevancia]),
        )
        if r.faithfulness is None or r.answer_relevancy is None:
            print(f"  Aviso: nota ausente (NaN) na pergunta: {pergunta[:60]}... — rode de novo para tentar outra vez.")
        resultados.append(r)
    return resultados


def comparar_configuracoes(
    configuracoes: Dict[str, dict] = CONFIGURACOES_CHUNKING,
    perguntas: List[str] = PERGUNTAS_TESTE,
) -> pd.DataFrame:
    """Roda a avaliação para todas as configurações de chunking e devolve
    uma tabela comparativa (pandas DataFrame), pronta para colar no
    README.md ou salvar como CSV."""
    from app.embeddings import get_embeddings
    from app.generation import get_llm

    embeddings = get_embeddings()
    llm = get_llm(temperature=0.0)

    todas_linhas = []
    for nome_config in configuracoes:
        print(f"Avaliando configuração '{nome_config}'...")
        resultados = avaliar_configuracao(nome_config, perguntas, embeddings, llm)
        todas_linhas.extend(r.model_dump() for r in resultados)

    df = pd.DataFrame(todas_linhas)
    return df


def resumo_por_configuracao(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a tabela detalhada (por pergunta) em médias por configuração,
    para facilitar a decisão de qual configuração 'ganhou'."""
    resumo = df.groupby("configuracao_chunking")[["faithfulness", "answer_relevancy"]].mean()
    resumo["aprovado"] = resumo["faithfulness"] >= LIMIAR_FAITHFULNESS_APROVACAO
    return resumo.sort_values("faithfulness", ascending=False)


if __name__ == "__main__":
    df_detalhado = comparar_configuracoes()
    print("\n=== Resultado detalhado (por pergunta) ===")
    print(df_detalhado.to_string(index=False))

    df_resumo = resumo_por_configuracao(df_detalhado)
    print("\n=== Resumo por configuração de chunking ===")
    print(df_resumo.to_string())

    df_detalhado.to_csv("resultados_ragas_detalhado.csv", index=False)
    df_resumo.to_csv("resultados_ragas_resumo.csv")
    print("\nCSV salvos: resultados_ragas_detalhado.csv e resultados_ragas_resumo.csv")
