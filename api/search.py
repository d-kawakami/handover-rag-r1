"""検索専用エンドポイント: /api/search, /api/filter-options"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api._routing import classify_query
from search.dates import parse_date_str
from search.filters import sf_from_params
from search.retrieval import fulltext_scan, search_docs
from state import state

router = APIRouter()


@router.get("/api/search")
def search_only(
    q: str,
    top_k: int = 5,
    scan: bool = False,
    kinmu: list[str] = Query(default=[]),
    shubetsu: list[str] = Query(default=[]),
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="クエリを入力してください")
    sf = sf_from_params(kinmu, shubetsu, year, month, day)
    sf_or_none = sf if not sf.is_empty() else None
    try:
        if scan:
            docs = fulltext_scan(q, sf=sf_or_none)
            route_name = "fullscan"
        else:
            route = classify_query(q)
            if route is not None:
                _, docs = route.handler(q, sf_or_none)
                route_name = route.name
            else:
                docs = search_docs(q, top_k, sf=sf_or_none, enforce_and=True)
                route_name = "rag"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {e}")
    return {
        "results": [{"document": d.page_content, "metadata": d.metadata} for d in docs],
        "route": route_name,
        "filter_label": sf.label(),
    }


@router.get("/api/filter-options")
def filter_options():
    """フィルター用の選択肢（年リスト）を返す。"""
    years: set[int] = set()
    for d in state.all_docs:
        parsed = parse_date_str(d.metadata.get("日付", ""))
        if parsed:
            years.add(parsed.year)
    return {"years": sorted(years)}
