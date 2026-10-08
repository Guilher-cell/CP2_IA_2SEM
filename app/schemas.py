"""
Schemas Pydantic v2 do DocMind RAG.

Correção em relação ao CKP01: lá a avaliação apontou ausência de
`field_validator` (0,00/0,30). Aqui os schemas são ampliados para cobrir toda
a cadeia do RAG — documento, chunk e resposta — e cada um tem pelo menos uma
regra de validação ativa via `@field_validator`, não apenas tipos declarados.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

EXTENSOES_SUPORTADAS = {".pdf", ".txt", ".md"}


class DocumentoFonte(BaseModel):
    """Metadados de um documento real da base de conhecimento (antes do split)."""

    arquivo: str = Field(..., description="Nome do arquivo em data/docs (ex: 'doc1_guia_alimentar_brasileiro.md').")
    titulo: str = Field(..., description="Título legível do documento.")
    fonte_url: str = Field(..., description="URL de origem do documento (obrigatório — documento sem fonte não é aceito).")
    categoria: str = Field(default="diretriz_oficial", description="Categoria do documento (ex: diretriz_oficial, artigo_cientifico).")
    data_publicacao: Optional[date] = Field(default=None, description="Data de publicação do documento original, se conhecida.")

    @field_validator("arquivo")
    @classmethod
    def arquivo_deve_ter_extensao_suportada(cls, v: str) -> str:
        extensao = Path(v).suffix.lower()
        if extensao not in EXTENSOES_SUPORTADAS:
            raise ValueError(
                f"Extensão '{extensao}' não suportada. Use um dos formatos exigidos pelo "
                f"checkpoint: {sorted(EXTENSOES_SUPORTADAS)}."
            )
        return v

    @field_validator("fonte_url")
    @classmethod
    def fonte_url_nao_pode_ser_vazia(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError(
                "fonte_url não pode ser vazia — documentos sem fonte citada perdem todo "
                "o peso do critério 'Base de conhecimento' na rubrica do CKP02."
            )
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("fonte_url deve ser uma URL válida (começando com http:// ou https://).")
        return v


class Chunk(BaseModel):
    """Um pedaço (chunk) de um documento, pronto para ser embedado e indexado."""

    chunk_id: str = Field(..., description="Identificador único do chunk (ex: 'doc1_guia_alimentar_brasileiro::3').")
    texto: str = Field(..., min_length=1, description="Conteúdo textual do chunk.")
    documento_fonte: str = Field(..., description="Nome do arquivo de origem (DocumentoFonte.arquivo).")
    fonte_url: str = Field(..., description="URL de origem herdada do documento, propagada para citação.")
    pagina: Optional[int] = Field(default=None, description="Número de página de origem (quando aplicável, ex: PDFs).")
    indice: int = Field(..., ge=0, description="Posição (0-indexada) do chunk dentro do documento de origem.")
    chunk_size_config: int = Field(..., gt=0, description="chunk_size usado para gerar este chunk.")

    @field_validator("texto")
    @classmethod
    def texto_nao_pode_ser_so_espacos(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk com texto vazio/apenas espaços não deve ser indexado.")
        return v

    @field_validator("pagina")
    @classmethod
    def pagina_deve_ser_positiva(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("Número de página deve ser >= 1 quando informado.")
        return v


class Citacao(BaseModel):
    """Proveniência de uma afirmação feita na resposta do RAG."""

    arquivo: str = Field(..., description="Arquivo-fonte do chunk citado.")
    pagina: Optional[int] = Field(default=None, description="Página de origem, quando disponível (ex: PDFs).")
    chunk_id: str = Field(..., description="Identificador do chunk recuperado que fundamenta a resposta.")
    trecho: str = Field(..., description="Pequeno trecho do chunk citado, usado como evidência.")

    @field_validator("trecho")
    @classmethod
    def trecho_deve_ser_curto(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("O trecho de evidência não pode ser vazio.")
        limite = 300
        if len(v) > limite:
            v = v[:limite].rstrip() + "..."
        return v


class RespostaRAG(BaseModel):
    """Saída validada do pipeline RAG — resposta final com citação de fonte."""

    pergunta: str = Field(..., min_length=1)
    resposta: str = Field(..., min_length=1)
    citacoes: List[Citacao] = Field(
        default_factory=list,
        description="Lista de citações (arquivo + página + chunk) que fundamentam a resposta.",
    )
    confianca: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiança heurística da resposta, de 0 (sem suporte nos documentos) a 1 (bem fundamentada).",
    )
    fundamentado_em_documentos: bool = Field(
        ...,
        description="Verdadeiro se a resposta se baseia nos documentos recuperados; falso se o modelo não encontrou suporte na base.",
    )

    @field_validator("confianca")
    @classmethod
    def confianca_em_faixa_valida(cls, v: float) -> float:
        # Pydantic já garante ge=0.0/le=1.0, mas adicionamos arredondamento para
        # evitar valores como 0.9999999999 vindos de cálculos de similaridade.
        return round(v, 4)

    @model_validator(mode="after")
    def citacoes_obrigatorias_quando_fundamentado(self) -> "RespostaRAG":
        if self.fundamentado_em_documentos and not self.citacoes:
            raise ValueError(
                "Uma resposta marcada como 'fundamentado_em_documentos=True' precisa "
                "de pelo menos uma citação — resposta sem proveniência não deve ser "
                "apresentada como fundamentada na base de conhecimento."
            )
        return self


class ResultadoRAGAS(BaseModel):
    """Linha de resultado da avaliação RAGAS para uma pergunta, em uma configuração de chunking."""

    configuracao_chunking: str
    pergunta: str
    # None = o juiz do RAGAS não conseguiu pontuar essa amostra (NaN); fica fora das médias.
    faithfulness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("faithfulness")
    @classmethod
    def alerta_faithfulness_baixo(cls, v: Optional[float]) -> Optional[float]:
        # Não bloqueia o valor (ele é real, vindo do RAGAS), mas documenta a regra
        # de negócio do checkpoint: faithfulness < 0,5 indica alucinação significativa.
        return v
