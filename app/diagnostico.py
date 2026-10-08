"""
Diagnóstico de conexão com o Ollama (cloud e local).

Uso:
    python -m app.diagnostico

Testa separadamente: (1) a chave do .env, (2) chat no Ollama Cloud (gemma4:cloud),
(3) embeddings no Ollama Cloud, (4) embeddings no Ollama local. Assim fica claro
se o problema é a chave, o endpoint de embeddings ou a falta do servidor local.
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402

from app.config import (  # noqa: E402
    MODELO_CHAT,
    MODELO_EMBEDDING,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_LOCAL_BASE_URL,
    obter_api_key,
)


def _tentar(nome: str, funcao) -> None:
    try:
        funcao()
        print(f"[OK]    {nome}")
    except Exception as exc:
        print(f"[FALHA] {nome}\n         -> {type(exc).__name__}: {str(exc)[:200]}")


def main() -> None:
    chave = obter_api_key()
    if not chave:
        print("[FALHA] OLLAMA_API_KEY ausente. Confira se o arquivo .env existe na pasta do projeto.")
    else:
        print(f"[OK]    OLLAMA_API_KEY lida ({len(chave)} caracteres, termina em ...{chave[-4:]})")
        if "sua_chave" in chave:
            print("[FALHA] A chave ainda é o texto de exemplo do .env.example. Cole a chave real.")

    headers = {"Authorization": f"Bearer {chave}"}

    _tentar(
        f"Chat no Ollama Cloud ({MODELO_CHAT})",
        lambda: ChatOllama(model=MODELO_CHAT, base_url=OLLAMA_CLOUD_BASE_URL, headers=headers).invoke("Diga apenas: ok"),
    )
    _tentar(
        f"Embeddings no Ollama Cloud ({MODELO_EMBEDDING})",
        lambda: OllamaEmbeddings(
            model=MODELO_EMBEDDING, base_url=OLLAMA_CLOUD_BASE_URL, client_kwargs={"headers": headers}
        ).embed_query("teste"),
    )
    _tentar(
        f"Embeddings no Ollama LOCAL ({MODELO_EMBEDDING})",
        lambda: OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=OLLAMA_LOCAL_BASE_URL).embed_query("teste"),
    )

    print(
        "\nLeitura do resultado:\n"
        " - Chat OK + embeddings cloud FALHA (401): a chave vale para chat, mas o cloud não\n"
        "   libera embeddings. Use o local: instale o Ollama, `ollama pull nomic-embed-text`.\n"
        " - Chat FALHA (401): chave inválida/expirada. Gere outra em https://ollama.com/settings/keys.\n"
        " - Local FALHA (conexão recusada): o servidor Ollama local não está rodando."
    )


if __name__ == "__main__":
    main()
