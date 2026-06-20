"""勤務・種別・年月日による検索フィルター。"""
from __future__ import annotations

import re

from langchain_core.documents import Document

from normalizer import normalize_notation
from search.dates import parse_date_str


class SearchFilter:
    """勤務・種別・年月日による追加フィルター。フィールドが空/Noneの場合は全件対象。"""
    __slots__ = ("kinmu", "shubetsu", "year", "month", "day")

    def __init__(
        self,
        kinmu: list[str] | None = None,
        shubetsu: list[str] | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> None:
        self.kinmu: frozenset[str] = frozenset(kinmu or [])
        self.shubetsu: frozenset[str] = frozenset(shubetsu or [])
        self.year = year
        self.month = month
        self.day = day

    def is_empty(self) -> bool:
        return (
            not self.kinmu and not self.shubetsu
            and self.year is None and self.month is None and self.day is None
        )

    def matches(self, doc: Document) -> bool:
        if self.is_empty():
            return True
        meta = doc.metadata
        if self.kinmu and meta.get("勤務", "") not in self.kinmu:
            return False
        if self.shubetsu and meta.get("種別", "") not in self.shubetsu:
            return False
        if self.year is not None or self.month is not None or self.day is not None:
            d = parse_date_str(meta.get("日付", ""))
            if d is None:
                return False
            if self.year is not None and d.year != self.year:
                return False
            if self.month is not None and d.month != self.month:
                return False
            if self.day is not None and d.day != self.day:
                return False
        return True

    def label(self) -> str:
        """フィルター条件を人間が読める文字列で返す（空なら空文字）。"""
        if self.is_empty():
            return ""
        parts: list[str] = []
        if self.kinmu:
            parts.append("勤務=" + "/".join(sorted(self.kinmu)))
        if self.shubetsu:
            parts.append("種別=" + "/".join(sorted(self.shubetsu)))
        if self.year is not None:
            parts.append(f"{self.year}年")
        if self.month is not None:
            parts.append(f"{self.month}月")
        if self.day is not None:
            parts.append(f"{self.day}日")
        return "、".join(parts)


def sf_from_params(
    kinmu: list[str],
    shubetsu: list[str],
    year: int | None,
    month: int | None,
    day: int | None,
) -> SearchFilter:
    return SearchFilter(kinmu=kinmu, shubetsu=shubetsu, year=year, month=month, day=day)


def split_and_keywords(query: str) -> list[str]:
    """スペース区切りの検索キーワードを返す（日付表現・1文字は除外）。"""
    clean = re.sub(r'\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:以降|から|以前|まで|頃)?の?', '', query).strip()
    terms = [normalize_notation(t.strip()) for t in clean.split()]
    return [t for t in terms if len(t) >= 2]


def doc_matches_all_terms(doc: Document, terms: list[str]) -> bool:
    """ドキュメントが全キーワードを含むか判定（AND検索）。"""
    target = normalize_notation(doc.metadata.get("故障", "") + "\n" + doc.page_content)
    return all(t in target for t in terms)
