"""日付・季節パース、recency スコア。"""
from __future__ import annotations

import calendar
import math
import re
from datetime import date, timedelta

from langchain_core.documents import Document

from search.patterns import SEASONS


def parse_date_str(s: str) -> date | None:
    """メタデータの日付文字列をパース。

    XLSX由来の「YYYY-MM-DD ...」と CSV由来の「YYYY/M/D」の両形式に対応する。
    """
    # YYYY-MM-DD (XLSXインジェスト時: "2026-05-20 00:00:00" 形式)
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    # YYYY/M/D (CSV由来)
    m = re.match(r'(\d{4})/(\d{1,2})/(\d{1,2})', s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def doc_in_range(doc: Document, from_date: date | None, to_date: date | None) -> bool:
    d = parse_date_str(doc.metadata.get("日付", ""))
    if d is None:
        return True
    if from_date and d < from_date:
        return False
    if to_date and d > to_date:
        return False
    return True


def doc_in_month_set(doc: Document, months: set[int]) -> bool:
    """文書の月が指定の月セットに含まれるか判定する（日付なしは除外）。"""
    d = parse_date_str(doc.metadata.get("日付", ""))
    if d is None:
        return False
    return d.month in months


def recency_score(doc: Document, today: date, half_life_days: int = 365) -> float:
    """文書の新しさを 0.0〜1.0 で返す。half_life_days 前の記録がスコア 0.5。"""
    doc_date = parse_date_str(doc.metadata.get("日付", ""))
    if doc_date is None:
        return 0.0
    age_days = max(0, (today - doc_date).days)
    return math.exp(-math.log(2) * age_days / half_life_days)


def _safe_date(y: str, m: str, d: str) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def _season_start(year: int, season: str) -> date:
    return date(year, SEASONS[season][0], 1)


def _season_end(year: int, season: str) -> date:
    _, em = SEASONS[season]
    y = year + 1 if season == '冬' else year
    return date(y, em, calendar.monthrange(y, em)[1])


def parse_date_filter(query: str) -> tuple[date | None, date | None]:
    """クエリから日付の絞り込み条件（from, to）を抽出する。季節表現にも対応。"""
    sp = '(春|夏|秋|冬)'
    from_date: date | None = None
    to_date: date | None = None

    # ── 季節パターン ──────────────────────────────────────────
    # 「YYYY年S1からS2にかけて/まで」「YYYY年S1〜YYYY年S2」
    hit = re.search(r'(\d{4})年' + sp + r'から(?:(\d{4})年)?' + sp + r'(?:にかけて|まで|頃)?', query)
    if hit:
        y1, s1 = int(hit.group(1)), hit.group(2)
        y2 = int(hit.group(3)) if hit.group(3) else y1
        s2 = hit.group(4)
        return _season_start(y1, s1), _season_end(y2, s2)

    # 「YYYY年S以降/から」
    hit = re.search(r'(\d{4})年' + sp + r'(?:以降|から)', query)
    if hit:
        from_date = _season_start(int(hit.group(1)), hit.group(2))

    # 「YYYY年S以前/まで」
    hit = re.search(r'(\d{4})年' + sp + r'(?:以前|まで)', query)
    if hit:
        to_date = _season_end(int(hit.group(1)), hit.group(2))

    # 「YYYY年S」単体 → その季節全体を範囲とする
    if from_date is None and to_date is None:
        hit = re.search(r'(\d{4})年' + sp, query)
        if hit:
            y, s = int(hit.group(1)), hit.group(2)
            return _season_start(y, s), _season_end(y, s)

    # ── 年のみ範囲パターン ────────────────────────────────────────
    # 「2024年から2026年」「2024年〜2026年」「2024年の間〜2026年」
    if from_date is None and to_date is None:
        hit = re.search(r'(\d{4})年(?:から|〜|~)(\d{4})年', query)
        if hit:
            return date(int(hit.group(1)), 1, 1), date(int(hit.group(2)), 12, 31)

    # 「2024年以降/から」年のみ（月なし）
    if from_date is None:
        hit = re.search(r'(\d{4})年(?:以降|から)(?!\d)', query)
        if hit:
            from_date = date(int(hit.group(1)), 1, 1)

    # 「2026年以前/まで」年のみ（月なし）
    if to_date is None:
        hit = re.search(r'(\d{4})年(?:以前|まで)(?!\d)', query)
        if hit:
            to_date = date(int(hit.group(1)), 12, 31)

    # ── 数値日付パターン（季節未検出時のフォールバック）──────────
    full = r'(\d{4})[/年](\d{1,2})[/月](\d{1,2})日?'
    month_p = r'(\d{4})[/年](\d{1,2})(?:月|(?![一-鿿]))'

    if from_date is None:
        hit = re.search(full + r'(?:以降|から)', query)
        if hit:
            from_date = _safe_date(hit.group(1), hit.group(2), hit.group(3))
        else:
            hit = re.search(month_p + r'(?:以降|から)', query)
            if hit:
                from_date = _safe_date(hit.group(1), hit.group(2), '1')

    if to_date is None:
        hit = re.search(full + r'(?:以前|まで)', query)
        if hit:
            to_date = _safe_date(hit.group(1), hit.group(2), hit.group(3))
        else:
            hit = re.search(month_p + r'(?:以前|まで)', query)
            if hit:
                y, mo = int(hit.group(1)), int(hit.group(2))
                last = calendar.monthrange(y, mo)[1]
                to_date = _safe_date(str(y), str(mo), str(last))

    # 「2026年5月20日」単独（範囲指定なし）→ その日の完全一致
    if from_date is None and to_date is None:
        hit = re.search(full, query)
        if hit:
            d = _safe_date(hit.group(1), hit.group(2), hit.group(3))
            if d:
                return d, d

    # 「2026年5月」単独（日なし）→ その月全体
    if from_date is None and to_date is None:
        hit = re.search(month_p, query)
        if hit:
            y, mo = int(hit.group(1)), int(hit.group(2))
            last = calendar.monthrange(y, mo)[1]
            fd = _safe_date(str(y), str(mo), '1')
            td = _safe_date(str(y), str(mo), str(last))
            if fd and td:
                return fd, td

    # 「2025年」単独（他のパターンで未検出の場合のみ）→ その年全体を範囲とする
    if from_date is None and to_date is None:
        hit = re.search(r'(\d{4})年', query)
        if hit:
            y = int(hit.group(1))
            return date(y, 1, 1), date(y, 12, 31)

    # 相対時制（最近・直近・今年・昨年・先月 等）— 明示日付が無いときのフォールバック
    if from_date is None and to_date is None:
        fd, td = _detect_relative_recent(query)
        if fd or td:
            return fd, td

    return from_date, to_date


# 相対時制キーワード → 過去 N 日（today 基準）
# 「直近」「最新」のような強い語ほど狭い範囲、「近年」のような広い語ほど広い範囲。
_RELATIVE_DAYS: dict[str, int] = {
    "直近": 30,
    "最新": 30,
    "ここ最近": 90,
    "最近": 180,
    "近年": 365,
}


def _is_word_match(query: str, kw: str) -> bool:
    """`kw` がクエリ中で「漢字複合語の一部ではなく」現れるかを判定する。

    直後の文字が CJK 統合漢字なら複合語の一部とみなして False。
    例:
      「最近の」「最近、」「最近 」 → True
      「最新型」「今月末」「今年度」「近年代」 → False
    末尾出現は確定で True。複数回出現する場合は単語境界に出る最初のものを採用。
    """
    start = 0
    while True:
        idx = query.find(kw, start)
        if idx < 0:
            return False
        after = idx + len(kw)
        if after >= len(query):
            return True
        next_char = query[after]
        if not ('一' <= next_char <= '鿿'):
            return True
        start = after


def _detect_relative_recent(query: str) -> tuple[date | None, date | None]:
    """「最近」「直近」「今年」等の相対時制を today 基準の date range に変換する。

    検出ルール（漢字複合語の一部は除外する。例: 「最新型」「今月末」「今年度」）:
      今月 / 先月 → 該当月の 1 日〜末日
      今年 / 昨年 → 該当年の 1/1〜12/31
      直近 / 最新 → 過去 30 日
      ここ最近    → 過去 90 日
      最近        → 過去 180 日（半年）
      近年        → 過去 365 日（1 年）
    """
    today = date.today()

    # 月単位
    if _is_word_match(query, "今月"):
        first = today.replace(day=1)
        last_day = calendar.monthrange(today.year, today.month)[1]
        return first, date(today.year, today.month, last_day)
    if _is_word_match(query, "先月"):
        first_of_this_month = today.replace(day=1)
        last_of_prev = first_of_this_month - timedelta(days=1)
        return last_of_prev.replace(day=1), last_of_prev

    # 年単位
    if _is_word_match(query, "今年"):
        return date(today.year, 1, 1), date(today.year, 12, 31)
    if _is_word_match(query, "昨年") or _is_word_match(query, "去年"):
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)

    # 期間ベース（過去 N 日）— 強い語から先にマッチさせる
    for kw, days in _RELATIVE_DAYS.items():
        if _is_word_match(query, kw):
            return today - timedelta(days=days), today

    return None, None


def normalize_query_for_bm25(query: str) -> str:
    """クエリの表記揺れ・日付形式を正規化してBM25の精度を上げる。

    施設名略称・系列番号を正規形に統一した後、日付をDB内形式に変換する。
    """
    # 循環 import を避けるため関数内 import
    from normalizer import normalize_notation

    query = normalize_notation(query)
    query = re.sub(
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',
        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
        query,
    )
    query = re.sub(
        r'(\d{4})年(\d{1,2})月',
        lambda m: f"{m.group(1)}-{int(m.group(2)):02d}",
        query,
    )
    return query
