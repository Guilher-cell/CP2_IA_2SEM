"""
Constantes centrais do DocMind RAG.

Manter esses valores em um único módulo facilita ajustar o projeto (ex.: trocar
chunk_size, trocar o diretório de documentos) sem precisar caçar números mágicos
espalhados pelo código — e facilita reaproveitar esses valores no CKP03.
"""

from pathlib import Path

# --- Domínio (persistente nos 3 checkpoints do semestre) --------------------
DOMINIO = "nutricao"
NOME_COLECAO_CHROMA = f"docmind_{DOMINIO}"

# --- Caminhos -----------------------------------------------------------------
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DIRETORIO_DOCUMENTOS = RAIZ_PROJETO / "data" / "docs"
DIRETORIO_CHROMA = RAIZ_PROJETO / "data" / "chroma"

# --- Modelos (Ollama Cloud — únicos aprovados pelo checkpoint) ----------------
MODELO_CHAT = "gemma4:cloud"
MODELO_EMBEDDING = "nomic-embed-text"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"
OLLAMA_LOCAL_BASE_URL = "http://localhost:11434"

# --- Chunking -------------------------------------------------------------
# Configurações comparadas no checkpoint (>= 2 exigidas). A demonstração da
# Aula 07 usa 256/512/1024 — mantemos as 3 para dar mais pontos de comparação.
CONFIGURACOES_CHUNKING = {
    "pequeno_256": {"chunk_size": 256, "chunk_overlap": 32},   # 12.5% overlap
    "medio_512": {"chunk_size": 512, "chunk_overlap": 64},     # 12.5% overlap
    "grande_1024": {"chunk_size": 1024, "chunk_overlap": 128},  # 12.5% overlap
}
CONFIGURACAO_PADRAO = "medio_512"

SEPARADORES_SPLITTER = ["\n\n", "\n", ". ", " ", ""]

# --- Retrieval ------------------------------------------------------------
TOP_K_PADRAO = 5
TOP_K_RERANKING = 8  # recupera mais candidatos antes de rerankear e cortar para TOP_K_PADRAO

# --- RAGAS ------------------------------------------------------------------
LIMIAR_FAITHFULNESS_APROVACAO = 0.7
LIMIAR_FAITHFULNESS_IDEAL = 0.9


def obter_api_key() -> str:
    """Lê a OLLAMA_API_KEY do ambiente, removendo espaços, quebras de linha e
    aspas acidentais (causa comum de erro 401 quando a chave é colada no .env)."""
    import os

    chave = os.getenv("OLLAMA_API_KEY", "")
    return chave.strip().strip('"').strip("'").strip()
