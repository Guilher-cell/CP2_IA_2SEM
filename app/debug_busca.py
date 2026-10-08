"""
Mostra o que a busca vetorial recupera para uma pergunta (sem chamar o LLM).

Uso:
    python -m app.debug_busca "Qual o limite de açúcar livre segundo a OMS?"
    python -m app.debug_busca "..." grande_1024      # outra configuração de chunking

Útil quando o chatbot responde "não encontrei": se o chunk certo não aparece
aqui, o problema é a recuperação (retrieve), não a geração (generate).
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from app.config import CONFIGURACAO_PADRAO, TOP_K_PADRAO  # noqa: E402
from app.embeddings import get_embeddings  # noqa: E402
from app.vectorstore import carregar_vectorstore  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Uso: python -m app.debug_busca "sua pergunta" [configuracao_chunking]')
        return
    pergunta = sys.argv[1]
    config = sys.argv[2] if len(sys.argv) > 2 else CONFIGURACAO_PADRAO

    vs = carregar_vectorstore(get_embeddings(), config_chunking=config)
    resultados = vs.similarity_search_with_score(pergunta, k=TOP_K_PADRAO + 3)
    print(f"Pergunta: {pergunta}\nConfiguração: {config} (menor distância = mais relevante)\n")
    for i, (doc, dist) in enumerate(resultados, start=1):
        marca = "*" if i <= TOP_K_PADRAO else " "
        print(f"{marca}{i}. dist={dist:.3f} | {doc.metadata['chunk_id']}")
        print("     " + doc.page_content[:160].replace("\n", " ") + "...")
    print("\n(* = chunks que realmente vão para o LLM)")


if __name__ == "__main__":
    main()
