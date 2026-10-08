# CKP02 — DocMind RAG · Domínio: Nutrição

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**

## Integrantes

| Nome | RM |
|---|---|
| André Fujinaga | RM569158 |
| Arthur Machado | RM569919 |
| Conrado Gracie | RM569157 |
| Guilherme Belo | RM570079 |
| Renato Sandreschi | RM569156 |

---

## 1. Visão geral

O DocMind RAG é a continuação do chatbot **NutriAssist** (CKP01) e mantém o
**mesmo domínio: orientação nutricional geral**, como exige o arco do semestre.
O chatbot do CKP01 deixa de responder a partir do que o modelo "sabe" e passa a
responder **somente a partir de uma base de documentos reais**, citando a fonte
de cada resposta.

Pipeline completo: `load → split → embed → store → retrieve → generate`, com
comparação quantitativa de três estratégias de chunking via **RAGAS**
(faithfulness + answer_relevancy).

**Usuários-alvo:** pessoas leigas que querem orientação nutricional geral e
segura. O sistema não substitui nutricionista ou médico, e isso está explícito
no prompt e na interface.

## 2. Correções em relação ao feedback do CKP01

O CKP01 recebeu dois apontamentos. Ambos foram tratados neste projeto.

| Apontamento do CKP01 | O que foi feito |
|---|---|
| **Gestão de memória / demonstração com 5+ turnos (0,00 de 0,50)** e a orientação de *isolar a camada de contexto em um módulo só* | `app/context_manager.py` concentra toda a lógica de contexto atrás de uma interface (`GerenciadorDeContexto`). A memória de chat do CKP01 foi **substituída** pelo recuperador do RAG: a troca foi de implementação, sem reescrever a geração (`app/generation.py` não depende de como o contexto é preparado). Perguntas de seguimento (ex.: *"e para quem treina?"*) são reescritas como perguntas autônomas antes da busca. |
| **Field validators ausentes (0,00 de 0,30)** e a orientação de ampliar os schemas com proveniência | `app/schemas.py` tem 5 modelos (`DocumentoFonte`, `Chunk`, `Citacao`, `RespostaRAG`, `ResultadoRAGAS`) com `@field_validator` e `@model_validator` ativos. Exemplos: uma `RespostaRAG` marcada como fundamentada **é rejeitada se não tiver citação**; um `DocumentoFonte` sem `fonte_url` válida é rejeitado. `Citacao` carrega `arquivo`, `pagina` e `chunk_id` como proveniência por resposta. |

## 3. Base de conhecimento (5 documentos reais, com fonte citada)

| Arquivo | Fonte | URL |
|---|---|---|
| `doc1_guia_alimentar_brasileiro.md` | Ministério da Saúde (Brasil) — Guia Alimentar para a População Brasileira, 2ª ed., 2014 | https://bvsms.saude.gov.br/bvs/publicacoes/guia_alimentar_populacao_brasileira_2ed.pdf |
| `doc2_rotulagem_nutricional.md` | Anvisa — RDC nº 429/2020, IN nº 75/2020 e *Perguntas e Respostas sobre Rotulagem Nutricional* | https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/alimentos/perguntas-e-respostas-arquivos/rotulagem-nutricional_2a-edicao.pdf |
| `doc3_recomendacoes_oms_dieta_saudavel.md` | OMS — *Healthy diet* (fact sheet) | https://www.who.int/news-room/fact-sheets/detail/healthy-diet |
| `doc4_hidratacao_e_agua.md` | EFSA — parecer sobre valores de referência para água | https://www.efsa.europa.eu/en/press/news/nda100326 |
| `doc5_atividade_fisica_e_nutricao.md` | OMS — Diretrizes sobre atividade física e comportamento sedentário (2020) | https://www.who.int/europe/publications/i/item/9789240014886 |

Cada arquivo é um **resumo adaptado em português** do documento oficial citado,
não uma cópia literal. A fonte fica no cabeçalho do arquivo (`Fonte:`, `URL:`,
`Categoria:`, `Rotulo:`), e `app/loaders.py` lê esse cabeçalho e valida o
documento com `DocumentoFonte` (Pydantic) antes de aceitá-lo na base.

### Como adicionar novos documentos à base

1. Coloque o arquivo (`.md`, `.txt` ou `.pdf`) em `data/docs/`.
2. Em `.md`/`.txt`, comece o arquivo com o cabeçalho (use qualquer documento
   existente como modelo):
   ```
   Fonte: Nome do documento e órgão emissor
   URL: https://endereco-da-fonte
   Categoria: diretriz_oficial
   Rotulo: Rótulo curto da fonte (ex.: OMS — hidratação)
   ```
   Documentos sem `URL:` válida são rejeitados.
3. Em `.pdf`, as primeiras linhas do texto da primeira página devem conter ao
   menos `Fonte:` e `URL:`. Cada página vira um documento e o número da página
   aparece na citação.
4. Rode `python -m app.ingest` para reindexar todas as configurações de chunking.

## 4. Arquitetura

```
load      app/loaders.py        lê .md/.txt/.pdf e valida com DocumentoFonte
  │
split     app/chunking.py       RecursiveCharacterTextSplitter, 3 configurações
  │
embed     app/embeddings.py     nomic-embed-text (Ollama), com prefixos do modelo
  │
store     app/vectorstore.py    ChromaDB local (PersistentClient), 1 coleção por configuração
  │
retrieve  app/retriever.py      buscar(consulta) · metadata filtering · reranking
  │
generate  app/generation.py     gemma4:cloud, temperature=0, resposta com citação [n]
```

Antes do `retrieve`, `app/context_manager.py` prepara a consulta (reescrita de
perguntas de seguimento). A interface fica em `app/main.py` (Gradio).

### Decisões técnicas relevantes

- **Splitter:** `RecursiveCharacterTextSplitter` com
  `separators=["\n\n", "\n", ". ", " ", ""]` e `chunk_overlap` de 12,5% do
  `chunk_size` nas três configurações.
- **Chunk contextual:** cada chunk recebe um prefixo `[rótulo da fonte]`, por
  exemplo `[OMS — recomendações de dieta saudável]`. Sem isso, um chunk como
  *"Açúcares livres: devem representar menos de 10%..."* não diz de qual fonte
  vem, e perguntas como *"segundo a OMS"* não casavam com ele. O prefixo
  acrescenta cerca de 40 caracteres a cada chunk, além do `chunk_size` do splitter.
- **Prefixos do nomic-embed-text:** o modelo pede `search_document:` ao indexar
  e `search_query:` ao consultar. O wrapper `EmbeddingsNomic` aplica ambos, o que
  melhorou a recuperação em português.
- **Embeddings (cloud ou local):** `app/embeddings.py` usa
  `EMBEDDING_BACKEND=auto`. Tenta o Ollama Cloud e, se o endpoint de embeddings
  responder erro (por exemplo `401`), usa o Ollama local com **o mesmo modelo**
  (`nomic-embed-text`). Os vetores são equivalentes nos dois caminhos.
- **Camada de contexto:** a pergunta original **sempre** participa da busca,
  junto da versão reescrita (`buscar_unindo`). Assim uma reescrita ruim não
  deixa o chunk certo de fora. Respostas *"não encontrei"* não entram no
  histórico.
- **Top-k:** 5 chunks por pergunta (`TOP_K_PADRAO` em `app/config.py`).
- **Citação de fonte:** o prompt exige marcadores `[n]` ao lado de cada
  afirmação; `app/generation.py` converte cada marcador em uma `Citacao`
  (arquivo, página, chunk e trecho). Se a base não tiver a informação, o modelo
  responde que não encontrou, sem inventar.

## 5. Requisitos do enunciado

| Requisito | Status | Onde |
|---|---|---|
| Base real, ≥5 documentos com fonte citada | ✅ | `data/docs/` + tabela da seção 3 |
| `RecursiveCharacterTextSplitter` com os separators exigidos e overlap de 10–15% | ✅ | `app/chunking.py`, `app/config.py` |
| ≥2 configurações de `chunk_size` comparadas | ✅ | 256 / 512 / 1024 |
| `nomic-embed-text` via `OllamaEmbeddings` | ✅ | `app/embeddings.py` |
| `gemma4:cloud`, `temperature=0`, chave em `.env` | ✅ | `app/generation.py` |
| ChromaDB local, coleção com nome do domínio | ✅ | `app/vectorstore.py` (`docmind_nutricao__<config>`) |
| Pipeline end-to-end, resposta cita o chunk de origem | ✅ | `app/retriever.py` + `app/generation.py` |
| RAGAS faithfulness + answer_relevancy, ≥5 perguntas | ✅ | `app/evaluation.py` |
| Função `buscar(consulta)` modular para virar `@tool` no CKP03 | ✅ | `app/retriever.py` |
| **Diferencial:** metadata filtering | ✅ | `buscar(..., filtro_metadata={...})` |
| **Diferencial:** reranking com cross-encoder | ✅ | `app/reranker.py` (opcional, ver seção 8) |
| **Diferencial:** interface Gradio com fonte citada | ✅ | `app/main.py` |

## 6. Resultados — comparação de chunking com RAGAS

Foram usadas **5 perguntas de teste, uma por documento** (`PERGUNTAS_TESTE` em
`app/evaluation.py`), executadas em cada configuração com o mesmo modelo,
`temperature=0`:

1. O que é a classificação NOVA de alimentos segundo o Guia Alimentar Brasileiro?
2. O que significa o selo de lupa "ALTO EM" na rotulagem nutricional?
3. Qual o limite recomendado pela OMS para consumo de açúcares livres?
4. Quantos litros de água por dia a EFSA recomenda para homens e mulheres adultos?
5. Quanta proteína por quilo de peso corporal uma pessoa fisicamente ativa precisa?

### Resumo por configuração (média das 5 perguntas)

| Configuração | chunk_size | chunk_overlap | faithfulness | answer_relevancy | Aprovado (faithfulness ≥ 0,7) |
|---|---|---|---|---|---|
| `pequeno_256` | 256 | 32 | **1,000** | **0,853** | ✅ |
| `grande_1024` | 1024 | 128 | 1,000 | 0,834 | ✅ |
| `medio_512` | 512 | 64 | 1,000 | 0,775 | ✅ |

Referência do enunciado: faithfulness ≥ 0,7 aprova e ≥ 0,9 é a zona ideal. As
três configurações ficaram na zona ideal.

### Tabela detalhada por pergunta

O enunciado pede faithfulness e answer_relevancy **por pergunta**, para cada
configuração. Os valores estão em `resultados_ragas_detalhado.csv` (gerado por
`python -m app.evaluation`, junto de `resultados_ragas_resumo.csv`). Para
imprimir a tabela em Markdown e colá-la aqui:

```bash
pip install tabulate
python -c "import pandas as pd; print(pd.read_csv('resultados_ragas_detalhado.csv').to_markdown(index=False))"
```

> **[colar aqui a tabela detalhada gerada pelo comando acima]**

### Configuração escolhida e justificativa

**Escolhida: `pequeno_256` (chunk_size = 256, overlap = 32).**

- **Faithfulness empatou em 1,0** nas três configurações. Nesta base e com estas
  5 perguntas, o modelo respondeu apenas com o que estava nos trechos
  recuperados em qualquer tamanho de chunk, então essa métrica não diferencia as
  configurações.
- O **desempate foi o answer_relevancy**, em que `pequeno_256` teve a maior
  média (0,853), à frente de `grande_1024` (0,834) e `medio_512` (0,775).
- Uma explicação plausível, que não foi testada separadamente: os documentos
  são resumos curtos organizados em listas, em que cada item trata de um fato
  isolado. Chunks pequenos tendem a isolar um fato por trecho, e com 5 trechos
  por pergunta o contexto enviado ao modelo fica mais focado na pergunta.

**Limitações da comparação:** são apenas 5 perguntas por configuração, e a
faithfulness chegou ao teto (1,0), então a diferença de ~0,08 entre as
configurações em answer_relevancy é um indício, não uma prova estatística. Uma
avaliação mais forte exigiria mais perguntas, incluindo perguntas que cruzam
mais de um documento.

> Observação: o valor de `CONFIGURACAO_PADRAO` em `app/config.py` define qual
> coleção o chat (`app/main.py`) consulta. Para o chat usar a configuração
> vencedora, deixe `CONFIGURACAO_PADRAO = "pequeno_256"`.

## 7. Como executar

Não é necessário ter o Ollama instalado para o chat (`gemma4:cloud` roda no
Ollama Cloud). Os embeddings usam o Ollama Cloud e, se ele não os atender com a
sua chave, o Ollama local (ver seção 4).

```bash
# 1. Ambiente (recomendado: Python 3.11 ou 3.12)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Chave do Ollama Cloud (https://ollama.com/settings/keys)
cp .env.example .env              # edite e cole sua OLLAMA_API_KEY

# 3. Indexar os documentos (uma vez, ou após mudar data/docs/)
python -m app.ingest

# 4. Conversar (Gradio em http://localhost:7861)
python -m app.main

# 5. Reproduzir a comparação de chunking com RAGAS (leva alguns minutos)
python -m app.evaluation
```

O arquivo `.env` contém a chave e **nunca** deve ser enviado na entrega.

## 8. Diagnóstico e solução de problemas

| Sintoma | O que fazer |
|---|---|
| `401 unauthorized` em embeddings | `python -m app.diagnostico` testa a chave, o chat no cloud, os embeddings no cloud e os embeddings no local. Se só os embeddings do cloud falharem, instale o Ollama (https://ollama.com/download) e rode `ollama pull nomic-embed-text`; o modo `auto` passa a usar o local. |
| O chat diz "não encontrei" para algo que está nos documentos | `python -m app.debug_busca "sua pergunta"` mostra os chunks recuperados e a distância de cada um (os marcados com `*` vão ao modelo). Se o chunk certo não aparece, o problema está na recuperação. |
| `ModuleNotFoundError: ...vertexai` ao rodar o RAGAS | Já tratado por `app/compat_ragas.py`. O `ragas` importa classes do Vertex AI que o `langchain-community` atual removeu; o módulo registra classes vazias no lugar delas antes do import. |
| Nota `NaN` no RAGAS | O modelo juiz não devolveu o formato que o RAGAS exige naquela amostra. A nota é registrada como ausente, fica fora das médias e um aviso é impresso. Rode de novo para repetir. |
| Reranking indisponível | Opcional. `pip install -r requirements-reranking.txt` (baixa um modelo cross-encoder). Sem isso o projeto funciona normalmente, sem reordenar os resultados. |



## 9. Estrutura do projeto

```
docmind_rag/
├── app/
│   ├── __init__.py
│   ├── config.py            constantes (modelos, chunking, top-k, limiares RAGAS)
│   ├── schemas.py           modelos Pydantic v2 com validadores
│   ├── loaders.py           load
│   ├── chunking.py          split
│   ├── embeddings.py        embed (nomic-embed-text, cloud/local, prefixos)
│   ├── vectorstore.py       store (ChromaDB)
│   ├── context_manager.py   camada de contexto isolada
│   ├── retriever.py         retrieve — buscar(), filtros, reranking, busca unida
│   ├── reranker.py          cross-encoder (opcional)
│   ├── generation.py        generate com citação de fonte
│   ├── evaluation.py        RAGAS (faithfulness + answer_relevancy)
│   ├── compat_ragas.py      compatibilidade do RAGAS com langchain-community
│   ├── ingest.py            ingestão: load → split → embed → store
│   ├── diagnostico.py       teste de conexão com o Ollama
│   ├── debug_busca.py       inspeção da recuperação
│   └── main.py              interface Gradio
├── data/
│   ├── docs/                os 5 documentos da base
│   └── chroma/              índices do ChromaDB (gerados pela ingestão)
├── notebooks/
│   └── docmind_rag_colab.ipynb
├── .env.example
├── requirements.txt
├── requirements-reranking.txt
└── README.md
```

## 10. Avisos

- Modelos usados: exclusivamente `gemma4:cloud` (chat) e `nomic-embed-text`
  (embeddings).
- A `OLLAMA_API_KEY` vem do `.env` (local) ou de Secrets (Colab); nunca fica
  fixa no código.
- Este sistema oferece orientação nutricional geral e **não substitui**
  consulta com nutricionista ou médico.

---
Disciplina: Prompt Engineering and Artificial Intelligence · FIAP · CC 2026

