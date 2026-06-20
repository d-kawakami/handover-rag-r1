"""API リクエスト・レスポンスの Pydantic モデル。"""
from __future__ import annotations

from pydantic import BaseModel

from config import TOP_K_FINAL


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K_FINAL
    recency_weight: float = 0.0
    filter_kinmu: list[str] = []
    filter_shubetsu: list[str] = []
    filter_year: int | None = None
    filter_month: int | None = None
    filter_day: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class ModelChangeRequest(BaseModel):
    model: str


class ReasoningRequest(BaseModel):
    enabled: bool


class EquipSamplesRequest(BaseModel):
    samples: int


class SudachiModeRequest(BaseModel):
    mode: str


class DictEntryCreate(BaseModel):
    surface: str
    reading: str
    pos: str = "名詞,固有名詞,一般"
    cost: int = 5000
    normalized: str = ""
    enabled: int = 1


class DictEntryUpdate(BaseModel):
    surface: str | None = None
    reading: str | None = None
    pos: str | None = None
    cost: int | None = None
    normalized: str | None = None
    enabled: int | None = None


class SuggestEntry(BaseModel):
    surface: str
    reading: str
    normalized: str = ""
    pos: str = "名詞,固有名詞,一般"
    cost: int = 5000


class SuggestRegisterRequest(BaseModel):
    entries: list[SuggestEntry]
