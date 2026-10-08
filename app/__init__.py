"""
DocMind RAG — Pipeline RAG completo sobre a base de conhecimento de Nutrição
CKP02 · Prompt Engineering & AI · FIAP · 2º Semestre 2026

Mesmo domínio do CKP01 (NutriAssist): orientação nutricional geral.

Módulos:
    - config.py            -> constantes do projeto (modelos, paths, chunk sizes)
    - schemas.py            -> modelos Pydantic v2 (documentos, chunks, respostas)
    - loaders.py            -> carregamento dos documentos reais (load)
    - chunking.py           -> estratégias de chunking (split)
    - embeddings.py         -> OllamaEmbeddings / nomic-embed-text (embed)
    - vectorstore.py        -> ChromaDB local (store)
    - context_manager.py    -> camada de contexto isolada (contextualização de perguntas)
    - retriever.py           -> busca semântica + metadata filtering (retrieve) — buscar()
    - reranker.py            -> reranking opcional com cross-encoder (diferencial)
    - generation.py          -> geração da resposta com citação de fonte (generate)
    - evaluation.py          -> avaliação RAGAS (faithfulness + answer_relevancy)
    - ingest.py               -> script de ingestão (load -> split -> embed -> store)
    - main.py                 -> interface Gradio (diferencial)
"""
