"""クエリルーティング: 質問文から「どの集計関数 + どのプロンプト」を使うかを決定する。

`/api/query` と `/api/query/stream` で同じ判定ロジックを共有するため
1 箇所に切り出している。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from prompts import (
    AGGREGATION_PROMPT,
    EQUIPMENT_TENDENCY_PROMPT,
    FAILURE_SEVERITY_LISTING_PROMPT,
    RAG_PROMPT,
    SEASONAL_TENDENCY_PROMPT,
    SPECIFIC_COUNT_PROMPT,
)
from search.aggregate import (
    aggregate_by_equipment,
    aggregate_by_season,
    aggregate_failure_generic,
    aggregate_for_query,
    count_specific_failure,
    list_failures_by_severity,
)
from search.classify import (
    is_aggregation_query,
    is_equipment_tendency_query,
    is_failure_severity_listing_query,
    is_generic_failure_tendency_query,
    is_seasonal_tendency_query,
    is_specific_failure_count_query,
)
from search.filters import SearchFilter


@dataclass
class Route:
    name: str
    handler: Callable[[str, SearchFilter | None], tuple[str, list[Document]]]
    prompt: ChatPromptTemplate


# クエリ判定の優先順位は元実装の /api/query に合わせる
_ROUTES: list[tuple[Callable[[str], bool], Route]] = [
    (is_failure_severity_listing_query,
     Route("severity_listing", list_failures_by_severity, FAILURE_SEVERITY_LISTING_PROMPT)),
    (is_specific_failure_count_query,
     Route("count", count_specific_failure, SPECIFIC_COUNT_PROMPT)),
    (is_aggregation_query,
     Route("aggregation", aggregate_for_query, AGGREGATION_PROMPT)),
    (is_seasonal_tendency_query,
     Route("seasonal", aggregate_by_season, SEASONAL_TENDENCY_PROMPT)),
    (is_equipment_tendency_query,
     Route("equip_tendency", aggregate_by_equipment, EQUIPMENT_TENDENCY_PROMPT)),
    (is_generic_failure_tendency_query,
     Route("failure_generic", aggregate_failure_generic, EQUIPMENT_TENDENCY_PROMPT)),
]


def classify_query(question: str) -> Route | None:
    """質問文に該当する集計ルートを返す。該当なし（通常 RAG）の場合は None。"""
    for predicate, route in _ROUTES:
        if predicate(question):
            return route
    return None


RAG_FALLBACK = Route("rag", None, RAG_PROMPT)  # type: ignore[arg-type]
