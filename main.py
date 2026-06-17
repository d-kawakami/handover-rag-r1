"""
引継ぎノート RAG - FastAPI バックエンド
LangChain + Ollama(qwen2.5:32b) + ChromaDB
ハイブリッド検索(BM25 + ベクトル + RRF) + CrossEncoder Re-ranking
"""
import asyncio
import calendar
import csv
import io
import json
import logging
import math
import random
import re
import subprocess
import unicodedata
import threading
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import chromadb
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from dict_builder import USER_DIC_PATH, rebuild_and_reload
from dict_db import (
    add_entry, delete_entry, export_to_csv, find_by_surface, get_all, get_all_surfaces,
    import_from_csv, init_db, update_entry,
)
from ingest import CHROMA_PATH, COLLECTION_NAME, run_ingest
from tokenizer import get_tokenizer, init_tokenizer

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:14b"
#LLM_MODEL = "qwen2.5:32b"
#LLM_MODEL = "qwen3:30b-a3b"
#LLM_MODEL = "qwen3.6:35b-a3b"
LLM_REASONING = False
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
TOP_K_RETRIEVE = 40
TOP_K_FINAL = 5

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

_all_docs: list[Document] = []
_bm25_retriever: BM25Retriever | None = None
_vector_store: Chroma | None = None
_reranker: CrossEncoder | None = None
_embeddings: OllamaEmbeddings | None = None
_llm: OllamaLLM | None = None

_model_lock = threading.Lock()

_ingest_state: dict = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "待機中",
    "done": False,
    "error": None,
    "result": None,
}
_ingest_lock = threading.Lock()


def _set_ingest_state(**kwargs) -> None:
    with _ingest_lock:
        _ingest_state.update(kwargs)


def build_index() -> None:
    global _all_docs, _bm25_retriever, _vector_store
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing_names:
        log.warning("ChromaDB にコレクションがありません。先に ingest を実行してください")
        return
    col = client.get_collection(COLLECTION_NAME)
    count = col.count()
    if count == 0:
        log.warning("ChromaDB が空です")
        return

    result = col.get(limit=count, include=["documents", "metadatas"])
    _all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]
    _bm25_retriever = BM25Retriever.from_documents(
        _all_docs,
        k=TOP_K_RETRIEVE,
        preprocess_func=get_tokenizer().tokenize,
    )
    _vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=CHROMA_PATH,
    )
    log.info("インデックス構築完了: %d 件", count)


def rrf(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    seen: dict[str, int] = {}
    scores: dict[str, float] = {}
    id_to_doc: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content
            if key not in seen:
                seen[key] = len(seen)
            idx = str(seen[key])
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            id_to_doc[idx] = doc
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [id_to_doc[i] for i in sorted_ids]


def normalize_query_for_bm25(query: str) -> str:
    """クエリ日付をDB内の日付形式に変換してBM25の精度を上げる。
    XLSXインジェスト時のメタデータ日付形式は「YYYY-MM-DD HH:MM:SS」のため
    クエリ「2026年5月3日」→「2026-05-03」に変換して一致させる。"""
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


def _parse_date_str(s: str) -> date | None:
    """メタデータの日付文字列をパース。XLSX由来の「YYYY-MM-DD ...」と
    CSV由来の「YYYY/M/D」の両形式に対応する。"""
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


def _doc_in_range(doc: Document, from_date: date | None, to_date: date | None) -> bool:
    d = _parse_date_str(doc.metadata.get("日付", ""))
    if d is None:
        return True
    if from_date and d < from_date:
        return False
    if to_date and d > to_date:
        return False
    return True


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
            d = _parse_date_str(meta.get("日付", ""))
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


def _sf_from_params(
    kinmu: list[str],
    shubetsu: list[str],
    year: int | None,
    month: int | None,
    day: int | None,
) -> SearchFilter:
    return SearchFilter(kinmu=kinmu, shubetsu=shubetsu, year=year, month=month, day=day)


def _split_and_keywords(query: str) -> list[str]:
    """スペース区切りの検索キーワードを返す（日付表現・1文字は除外）。"""
    clean = re.sub(r'\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:以降|から|以前|まで|頃)?の?', '', query).strip()
    terms = [unicodedata.normalize("NFKC", t.strip()) for t in clean.split()]
    return [t for t in terms if len(t) >= 2]


def _doc_matches_all_terms(doc: Document, terms: list[str]) -> bool:
    """ドキュメントが全キーワードを含むか判定（AND検索）。"""
    target = unicodedata.normalize("NFKC", doc.metadata.get("故障", "") + "\n" + doc.page_content)
    return all(t in target for t in terms)


def _doc_in_month_set(doc: Document, months: set[int]) -> bool:
    """文書の月が指定の月セットに含まれるか判定する（日付なしは除外）。"""
    d = _parse_date_str(doc.metadata.get("日付", ""))
    if d is None:
        return False
    return d.month in months


def recency_score(doc: Document, today: date, half_life_days: int = 365) -> float:
    """文書の新しさを 0.0〜1.0 で返す。half_life_days 前の記録がスコア 0.5。"""
    doc_date = _parse_date_str(doc.metadata.get("日付", ""))
    if doc_date is None:
        return 0.0
    age_days = max(0, (today - doc_date).days)
    return math.exp(-math.log(2) * age_days / half_life_days)


def parse_date_filter(query: str) -> tuple[date | None, date | None]:
    """クエリから日付の絞り込み条件（from, to）を抽出する。季節表現にも対応。"""
    def safe_date(y: str, m: str, d: str) -> date | None:
        try:
            return date(int(y), int(m), int(d))
        except ValueError:
            return None

    # 季節 → 開始月・終了月
    SEASONS = {'春': (3, 5), '夏': (6, 8), '秋': (9, 11), '冬': (12, 2)}

    def season_start(year: int, season: str) -> date:
        return date(year, SEASONS[season][0], 1)

    def season_end(year: int, season: str) -> date:
        _, em = SEASONS[season]
        y = year + 1 if season == '冬' else year
        return date(y, em, calendar.monthrange(y, em)[1])

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
        return season_start(y1, s1), season_end(y2, s2)

    # 「YYYY年S以降/から」
    hit = re.search(r'(\d{4})年' + sp + r'(?:以降|から)', query)
    if hit:
        from_date = season_start(int(hit.group(1)), hit.group(2))

    # 「YYYY年S以前/まで」
    hit = re.search(r'(\d{4})年' + sp + r'(?:以前|まで)', query)
    if hit:
        to_date = season_end(int(hit.group(1)), hit.group(2))

    # 「YYYY年S」単体 → その季節全体を範囲とする
    if from_date is None and to_date is None:
        hit = re.search(r'(\d{4})年' + sp, query)
        if hit:
            y, s = int(hit.group(1)), hit.group(2)
            return season_start(y, s), season_end(y, s)

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
    month_p = r'(\d{4})[/年](\d{1,2})月?'

    if from_date is None:
        hit = re.search(full + r'(?:以降|から)', query)
        if hit:
            from_date = safe_date(hit.group(1), hit.group(2), hit.group(3))
        else:
            hit = re.search(month_p + r'(?:以降|から)', query)
            if hit:
                from_date = safe_date(hit.group(1), hit.group(2), '1')

    if to_date is None:
        hit = re.search(full + r'(?:以前|まで)', query)
        if hit:
            to_date = safe_date(hit.group(1), hit.group(2), hit.group(3))
        else:
            hit = re.search(month_p + r'(?:以前|まで)', query)
            if hit:
                y, mo = int(hit.group(1)), int(hit.group(2))
                last = calendar.monthrange(y, mo)[1]
                to_date = safe_date(str(y), str(mo), str(last))

    # 「2026年5月20日」単独（範囲指定なし）→ その日の完全一致
    if from_date is None and to_date is None:
        hit = re.search(full, query)
        if hit:
            d = safe_date(hit.group(1), hit.group(2), hit.group(3))
            if d:
                return d, d

    # 「2026年5月」単独（日なし）→ その月全体
    if from_date is None and to_date is None:
        hit = re.search(month_p, query)
        if hit:
            y, mo = int(hit.group(1)), int(hit.group(2))
            last = calendar.monthrange(y, mo)[1]
            fd = safe_date(str(y), str(mo), '1')
            td = safe_date(str(y), str(mo), str(last))
            if fd and td:
                return fd, td

    # 「2025年」単独（他のパターンで未検出の場合のみ）→ その年全体を範囲とする
    if from_date is None and to_date is None:
        hit = re.search(r'(\d{4})年', query)
        if hit:
            y = int(hit.group(1))
            return date(y, 1, 1), date(y, 12, 31)

    return from_date, to_date


# 〇〇計・〇〇ポンプ などの設備名を種別ごとに個別にマッチするパターン
# （ひらがな・読点・助詞などで誤って複数語が結合しないよう種別ごとに分割）
_EQUIP_PAT = re.compile(
    r'[A-Za-zＡ-Ｚａ-ｚ][A-Za-z0-9Ａ-Ｚａ-ｚ０-９]{1,}計'   # UV計, DO計, MLSS計
    r'|[ァ-ン]{2,}計'                                            # アンモニア計
    r'|[ぁ-ん]{1,3}[一-鿿]{1,}計'                              # りん酸計
    r'|[一-鿿]{2,}計'                                           # 界面計, 濃度計
    r'|[一-鿿ぁ-んァ-ン]{2,}(?:ポンプ|ブロワ)'                # 汚泥ポンプ
)
_EQUIP_BOOST = 1.5   # CrossEncoder 生スコアへの加算量


def extract_equip_keywords(query: str) -> list[str]:
    """クエリ中の設備・機器名を抽出する（例: UV計, りん酸計, 汚泥ポンプ）"""
    return list(dict.fromkeys(m.group() for m in _EQUIP_PAT.finditer(query)))


AGGREGATION_KEYWORDS = ["ランキング", "多い順", "何回", "回以上", "頻度", "件数", "集計", "何件", "何度", "上位", "順位",
                        "最多", "最も多い", "一番多い", "最も故障", "多かった", "頻繁", "よく発生", "よく故障"]


def is_aggregation_query(query: str) -> bool:
    if any(kw in query for kw in AGGREGATION_KEYWORDS):
        return True
    if re.search(r'(最も|一番|最多).{0,10}(多い|多く|多かった|発生|故障)', query):
        return True
    if re.search(r'(多い|多く|多かった).{0,8}(機器|設備|故障|装置)', query):
        return True
    if re.search(r'(繰り返し|何度も|しばしば|たびたび).{0,10}(故障|発生|停止)', query):
        return True
    return False


def parse_min_count(query: str) -> int:
    m = re.search(r'(\d+)回以上', query)
    return int(m.group(1)) if m else 1


def parse_top_n(query: str) -> int:
    for pattern in [r'(\d+)位まで', r'上位(\d+)', r'(\d+)位']:
        m = re.search(pattern, query)
        if m:
            return int(m.group(1))
    return 10


def aggregate_for_query(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """集計クエリに対して全件スキャンで集計結果テキストとサンプルドキュメントを返す。"""
    from_date, to_date = parse_date_filter(query)
    min_count = parse_min_count(query)
    top_n = parse_top_n(query)

    base = [d for d in _all_docs if (sf is None or sf.matches(d))]
    filtered = [d for d in base if _doc_in_range(d, from_date, to_date)]
    if not filtered:
        return "指定期間の記録が見つかりませんでした。", []

    date_label = f"{from_date} 〜 {to_date}" if (from_date or to_date) else "全期間"
    is_groupby = any(kw in query for kw in ["ごと", "種別別", "毎に", "全種別", "各種別"])
    is_failure_focus = (not is_groupby) and any(kw in query for kw in ["故障", "障害", "不具合", "トラブル"])

    if is_failure_focus:
        target_docs = [
            d for d in filtered
            if d.metadata.get("故障", "").strip()
            or "故障" in d.metadata.get("種別", "")
        ]
    else:
        target_docs = filtered

    if not target_docs:
        return f"期間 {date_label} に該当する故障記録が見つかりませんでした。", []

    shubetsu_counter: Counter = Counter(
        d.metadata.get("種別", "（不明）").strip() or "（不明）"
        for d in target_docs
    )

    # 故障コードごとにカウント＋代表的な内容テキストを最大3件収集
    kosho_counter: Counter = Counter()
    kosho_samples: dict[str, list[str]] = {}
    for d in target_docs:
        ko = d.metadata.get("故障", "").strip()
        if not ko:
            continue
        kosho_counter[ko] += 1
        samples = kosho_samples.setdefault(ko, [])
        if len(samples) < 3:
            m = re.search(r"^内容: (.+)$", d.page_content, re.MULTILINE)
            if m:
                content = m.group(1).strip()[:60]
                if content not in samples:
                    samples.append(content)

    lines = [
        f"集計期間: {date_label}",
        f"対象記録数: {len(target_docs)}件",
        "",
        f"■ 種別ランキング（{min_count}件以上、上位{top_n}位）:",
    ]
    shown = 0
    for rank, (name, count) in enumerate(shubetsu_counter.most_common(top_n * 2), 1):
        if count >= min_count:
            lines.append(f"  {rank}位: {name} — {count}件")
            shown += 1
            if shown >= top_n:
                break
    if shown == 0:
        lines.append(f"  （{min_count}件以上の種別はありませんでした）")

    if kosho_counter:
        lines += ["", f"■ 故障機器別ランキング（{min_count}件以上、上位{top_n}位）:",
                  "  ※ 機器コードが数字の場合は、下の「記録内容」からどの機器・設備かを読み取れます"]
        shown = 0
        for rank, (code, count) in enumerate(kosho_counter.most_common(top_n * 2), 1):
            if count >= min_count:
                lines.append(f"  {rank}位: 機器/故障識別「{code}」— {count}件発生")
                for s in kosho_samples.get(code, []):
                    lines.append(f"    └ 記録内容: {s}")
                shown += 1
                if shown >= top_n:
                    break
        if shown == 0:
            lines.append(f"  （{min_count}件以上の故障はありませんでした）")

    log.info("集計完了: 期間=%s 対象=%d件", date_label, len(target_docs))
    return "\n".join(lines), target_docs[:top_n]


_FAILURE_TYPE_SET = frozenset({"故障", "故障処置"})
_GENERIC_SUBJECTS = frozenset({"故障", "障害", "不具合", "トラブル", "発生", "故障全般"})

EQUIPMENT_TENDENCY_MAX_SAMPLES = 20  # LLMへ渡す代表記録の上限（トークン溢れ防止）

_EQUIP_TENDENCY_KWS = [
    '傾向', 'パターン', '特徴', 'しやすい', 'なりやすい',
    '起きやすい', '発生しやすい', '注意点', '多発', 'よく起きる',
]

# 水処理施設ドメイン同義語グループ。検索語がいずれかに該当する場合グループ全体を候補にする。
_SYNONYM_GROUPS: list[frozenset[str]] = [
    frozenset(["VVVF", "インバータ", "インバーター", "可変電圧可変周波数"]),
    frozenset(["PAC", "ポリ塩化アルミニウム", "凝集剤"]),
]


def _expand_synonyms(terms: list[str]) -> list[str]:
    """トークンリスト中に既知の同義語グループの要素があれば、グループ全体を追加して返す。"""
    expanded = list(terms)
    for group in _SYNONYM_GROUPS:
        if any(t in group for t in terms):
            for syn in group:
                if syn not in expanded:
                    expanded.append(syn)
    return expanded


def extract_count_subject(query: str) -> str:
    """クエリから「何件数えるか」の対象名（設備名・故障名）を抽出する。"""
    # 先頭の年月日パターンを除去（日付はparse_date_filterで別途処理）
    q = re.sub(r'^\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:から|以降|以前|まで|頃)?の?', '', query).strip()
    patterns = [
        # 「全ての〇〇が発生した回数」→ 〇〇（例: 全てのVVVF故障が発生した回数）
        r'全ての?(.+?)(?:故障|障害|不具合)?(?:が|は)?発生(?:した)?(?:回数|件数)',
        # 「〇〇が/は発生した回数/件数」→ 〇〇
        r'^(.+?)[がは]発生(?:した)?(?:回数|件数)',
        # 「〇〇の実施/点検/発生/委託等の回数」→ 〇〇
        r'^(.+?)の\S{0,4}回数',
        # 「〇〇に関する故障/処置件数・発生回数」→ 〇〇
        r'^(.+?)に関する(?:故障処置|故障|障害|不具合|発生)?(?:が|は|の)?(?:何回|何件|何度|件数|発生回数|回数)',
        # 「〇〇の故障処置/故障/障害[がは]何回/件数」→ 〇〇
        r'^(.+?)の(?:故障処置|故障|障害|不具合|発生)(?:が|は|の)?(?:何回|何件|何度|件数|発生回数|回数)',
        # 「〇〇の故障処置/故障」で終わる形（文末に件数語がある場合）
        r'^(.+?)の(?:故障処置|故障|障害|不具合)',
        # 「〇〇[がは]何回/何件」→ 〇〇
        r'^(.+?)[がは].*?(?:何回|何件|何度)',
        # 「〇〇何回/何件/件数」→ 〇〇
        r'^(.+?)(?:何回|何件|何度|件数)',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            s = m.group(1).strip()
            # 先頭の不要語を除去（「全ての」「について」等）
            s = re.sub(r'^全ての?', '', s).strip()
            s = re.sub(r'^(?:について|に関して|関連する)', '', s).strip()
            # 末尾の助詞・「に関する〇〇」を除去
            s = re.sub(r'に関する.*$', '', s).strip()
            s = re.sub(r'[のがはにをで]$', '', s).strip()
            if len(s) >= 2 and s not in _GENERIC_SUBJECTS:
                return s
    return ""


def is_specific_failure_count_query(query: str) -> bool:
    """特定の設備・故障名の発生件数を問うクエリか判定する。"""
    count_kws = ["何回", "何件", "何度", "件数", "発生回数", "回発生", "た回数", "の回数", "した回数", "故障回数", "障害回数", "回数"]
    if not any(kw in query for kw in count_kws):
        return False
    # ランキング・全体集計・グループ集計クエリは aggregate_for_query に任せる
    exclude_kws = ["ランキング", "多い順", "最多", "最も多い", "一番多い", "上位", "頻度",
                   "ごと", "種別別", "毎に"]
    if any(kw in query for kw in exclude_kws):
        return False
    return bool(extract_count_subject(query))


def count_specific_failure(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """全種別の記録から、特定の故障名・設備名の出現件数を正確にカウントする。
    故障名が明示されている場合は種別フィルタ不要 — 故障名自体がフィルタになる。"""
    from_date, to_date = parse_date_filter(query)
    subject = extract_count_subject(query)
    # NFKC 正規化: 半角カナ等を全角に統一（例: ｲﾝﾊﾞｰﾀ → インバータ）
    subject = unicodedata.normalize("NFKC", subject)
    date_label = f"{from_date} 〜 {to_date}" if (from_date or to_date) else "全期間"

    base = [d for d in _all_docs if (sf is None or sf.matches(d))]
    range_docs = [d for d in base if _doc_in_range(d, from_date, to_date)]

    if not range_docs:
        return f"期間 {date_label} に記録が見つかりませんでした。", []

    # 末尾の「故障」「障害」「不具合」を除いたバリアントも検索
    search_variants: list[str] = [subject]
    base = re.sub(r'(?:故障|障害|不具合)+$', '', subject).strip()
    if base and base != subject and len(base) >= 2:
        search_variants.append(base)

    # 同義語グループで search_variants を展開（例: VVVF → インバータ等も検索）
    for group in _SYNONYM_GROUPS:
        if any(unicodedata.normalize("NFKC", v) in group for v in search_variants):
            for syn in group:
                if syn not in search_variants:
                    search_variants.append(syn)

    # 複合語（「原水ポンプVVVF」等）向け: トークン分割して AND マッチも試みる
    # 2 トークン以上に分かれる場合のみ有効（単語検索の精度補完）
    try:
        tok_terms = [t for t in get_tokenizer().tokenize(subject) if len(t) >= 2]
    except Exception:
        tok_terms = []
    # 各トークンを同義語グループで展開: [(term1, ...), (term2, ...)] の AND
    # グループ内は OR（いずれかが target に含まれれば条件充足）
    _tok_conds: list[tuple[str, ...]] = []
    for t in tok_terms:
        group = next((g for g in _SYNONYM_GROUPS if t in g), None)
        _tok_conds.append(tuple(group) if group else (t,))

    # 「の」で分割した場合の前後候補（「りん酸濃度の上限異常」→ head/tail で AND-OR）
    # head: フレーズ全体 OR 最初のトークン（len>=3 優先、なくてもok）
    # tail: フレーズ全体 OR 最初のトークン — "異常"などの汎用語単独は除く
    _no_parts = [p.strip() for p in subject.split("の") if p.strip()]
    if len(_no_parts) >= 2:
        _head_phrase = _no_parts[0]
        _tail_phrase = "の".join(_no_parts[1:])
        try:
            _htoks = [t for t in get_tokenizer().tokenize(_head_phrase) if len(t) >= 2]
            _ttoks = [t for t in get_tokenizer().tokenize(_tail_phrase) if len(t) >= 2]
        except Exception:
            _htoks, _ttoks = [], []
        # head候補: フレーズ + 最初のトークン（より具体的な語）
        _head_cands = [_head_phrase] + ([_htoks[0]] if _htoks else [])
        # tail候補: フレーズ + 最初のトークン（汎用的すぎる1語は避けるためlen>=3）
        _tail_first = next((t for t in _ttoks if len(t) >= 3), _ttoks[0] if _ttoks else "")
        _tail_cands = [c for c in [_tail_phrase, _tail_first] if c]
    else:
        _head_cands, _tail_cands = [], []

    def _doc_matches(d: Document) -> bool:
        raw = d.metadata.get("故障", "") + "\n" + d.page_content
        target = unicodedata.normalize("NFKC", raw)
        # 正規化後の文字列マッチ（単語・短い複合語）
        if any(unicodedata.normalize("NFKC", v) in target for v in search_variants):
            return True
        # トークン AND マッチ（複合語が文書中で分散して出現する場合）
        # 各条件は同義語グループ内 OR — 全条件 AND
        if len(_tok_conds) >= 2:
            if all(any(syn in target for syn in cond) for cond in _tok_conds):
                return True
        # 「の」分割 AND-OR マッチ: head候補のいずれか AND tail候補のいずれかが存在する
        if _head_cands and _tail_cands:
            if any(c in target for c in _head_cands) and any(c in target for c in _tail_cands):
                return True
        return False

    matched = [d for d in range_docs if _doc_matches(d)]

    # 種別ごとの内訳
    shubetsu_count: Counter = Counter(
        d.metadata.get("種別", "（不明）").strip() or "（不明）"
        for d in matched
    )
    breakdown = "、".join(f"{k} {v}件" for k, v in shubetsu_count.most_common())

    # 故障コード/機器別の内訳（機器識別に使用）
    kosho_counter: Counter = Counter()
    kosho_samples: dict[str, list[str]] = {}
    for d in matched:
        ko = d.metadata.get("故障", "").strip()
        if not ko:
            ko = "（故障コードなし）"
        kosho_counter[ko] += 1
        samples = kosho_samples.setdefault(ko, [])
        if len(samples) < 2:
            m2 = re.search(r"^内容: (.+)$", d.page_content, re.MULTILINE)
            if m2:
                snip = m2.group(1).strip()[:50]
                if snip not in samples:
                    samples.append(snip)

    lines = [
        f"集計期間: {date_label}",
        f"検索語: 「{subject}」（同義語含む: {', '.join(search_variants)}）",
        "",
        f"■ 「{subject}」（同義語含む）を含む記録件数: {len(matched)}件",
    ]
    if breakdown:
        lines.append(f"  （種別内訳: {breakdown}）")

    if len(kosho_counter) > 1:
        lines += ["", "■ 故障コード/機器別内訳:"]
        for code, cnt in kosho_counter.most_common():
            lines.append(f"  「{code}」: {cnt}件")
            for s in kosho_samples.get(code, []):
                lines.append(f"    └ 記録内容例: {s}")

    if matched:
        lines += ["", f"■ 該当記録（全{len(matched)}件）:"]
        for d in sorted(matched, key=lambda d: d.metadata.get("日付", "")):
            date_val = d.metadata.get("日付", "不明")
            shubetsu = d.metadata.get("種別", "")
            m = re.search(r"^内容: (.+)$", d.page_content, re.MULTILINE)
            snippet = m.group(1)[:60] if m else ""
            lines.append(f"  {date_val} [{shubetsu}] — {snippet}")
    else:
        lines.append("（該当する記録は見つかりませんでした）")

    log.info("特定故障カウント: subject=%s variants=%s 期間=%s マッチ=%d件",
             subject, search_variants, date_label, len(matched))
    return "\n".join(lines), matched


_SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    '春': (3, 4, 5),
    '夏': (6, 7, 8),
    '秋': (9, 10, 11),
    '冬': (12, 1, 2),
}
_SEASON_LABEL: dict[str, str] = {
    '春': '春季（3〜5月）',
    '夏': '夏季（6〜8月）',
    '秋': '秋季（9〜11月）',
    '冬': '冬季（12〜2月）',
}
_SEASON_WORD_MAP: dict[str, str] = {
    '夏': '夏', '夏場': '夏', '夏季': '夏', '夏期': '夏',
    '冬': '冬', '冬場': '冬', '冬季': '冬', '冬期': '冬',
    '春': '春', '春先': '春', '春季': '春',
    '秋': '秋', '秋口': '秋', '秋季': '秋',
}
_SEASONAL_TENDENCY_KWS = [
    '傾向', 'しやすい', 'なりやすい', '多い', '多く', '起きやすい',
    '発生しやすい', '時期', 'パターン', 'よく発生', 'よく故障',
    '注意', '特徴', 'ピーク', '増える', '増加',
]


def is_seasonal_tendency_query(query: str) -> bool:
    """「夏に発生しやすい」のような年指定なし季節傾向クエリを検出する。"""
    has_season = any(w in query for w in _SEASON_WORD_MAP) or '季節' in query
    if not has_season:
        return False
    if not any(kw in query for kw in _SEASONAL_TENDENCY_KWS):
        return False
    # 年指定があれば既存の date-filtered RAG に任せる
    return not bool(re.search(r'\d{4}年', query))


def _parse_target_season(query: str) -> str | None:
    """クエリ中に最初に出現する季節語を対象季節として返す。"""
    first_pos, result = len(query), None
    for word, season in _SEASON_WORD_MAP.items():
        pos = query.find(word)
        if pos != -1 and pos < first_pos:
            first_pos, result = pos, season
    return result


def aggregate_by_season(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """季節傾向クエリに対し、月別・季節別の故障集計と対象季節の記録を返す。"""
    target_season = _parse_target_season(query)

    base_docs = [d for d in _all_docs if (sf is None or sf.matches(d))]

    # 日付あり記録のみで月別集計
    month_counter: Counter = Counter()
    month_docs: dict[int, list[Document]] = {m: [] for m in range(1, 13)}
    for d in base_docs:
        doc_date = _parse_date_str(d.metadata.get("日付", ""))
        if doc_date:
            month_counter[doc_date.month] += 1
            month_docs[doc_date.month].append(d)

    total_dated = sum(month_counter.values())

    lines = [
        f"集計対象: 全期間 日付あり記録 {total_dated}件",
        "",
        "■ 月別記録件数（全期間・全種別）:",
    ]
    for m in range(1, 13):
        cnt = month_counter[m]
        unit = max(total_dated // 100, 1)
        bar = "█" * min(cnt // unit, 20)
        lines.append(f"  {m:2d}月: {cnt:4d}件 {bar}")

    lines += ["", "■ 季節別合計件数:"]
    for s, months in _SEASON_MONTHS.items():
        cnt = sum(month_counter[m] for m in months)
        lines.append(f"  {_SEASON_LABEL[s]}: {cnt}件")

    if target_season:
        target_months = set(_SEASON_MONTHS[target_season])
        target_docs = [d for d in base_docs if _doc_in_month_set(d, target_months)]

        lines += [
            "",
            f"■ {_SEASON_LABEL[target_season]}の故障傾向:",
            f"  該当記録数: {len(target_docs)}件",
        ]

        # 種別内訳
        shubetsu_cnt: Counter = Counter(
            d.metadata.get("種別", "（不明）").strip() or "（不明）"
            for d in target_docs
        )
        if shubetsu_cnt:
            lines.append("  種別内訳: " + "、".join(
                f"{k} {v}件" for k, v in shubetsu_cnt.most_common(5)
            ))

        # 故障コード上位（機器別傾向の把握に使用）
        kosho_cnt: Counter = Counter()
        kosho_samples: dict[str, list[str]] = {}
        for d in target_docs:
            ko = d.metadata.get("故障", "").strip()
            if ko:
                kosho_cnt[ko] += 1
                slist = kosho_samples.setdefault(ko, [])
                if len(slist) < 2:
                    m2 = re.search(r"^内容: (.+)$", d.page_content, re.MULTILINE)
                    if m2:
                        snip = m2.group(1).strip()[:50]
                        if snip not in slist:
                            slist.append(snip)

        if kosho_cnt:
            lines += ["", "  頻発する故障コード（上位10件）:"]
            for code, cnt in kosho_cnt.most_common(10):
                lines.append(f"    「{code}」: {cnt}件")
                for s in kosho_samples.get(code, []):
                    lines.append(f"      └ {s}")

        # 代表的な記録（最新30件）
        sample = sorted(target_docs, key=lambda d: d.metadata.get("日付", ""), reverse=True)[:30]
        if sample:
            lines += ["", "  代表的な記録（最新30件）:"]
            for d in sample:
                date_val = d.metadata.get("日付", "不明")
                shubetsu = d.metadata.get("種別", "")
                m2 = re.search(r"^内容: (.+)$", d.page_content, re.MULTILINE)
                snippet = m2.group(1)[:60] if m2 else ""
                lines.append(f"    {date_val} [{shubetsu}] — {snippet}")

        source_docs = target_docs
    else:
        # 季節指定なし：各月から3件ずつサンプルして月別分析のみ
        source_docs = []
        for m in range(1, 13):
            source_docs.extend(month_docs[m][:3])

    log.info("季節傾向集計: target=%s 全記録=%d 対象=%d件",
             target_season, total_dated, len(source_docs))
    return "\n".join(lines), source_docs


def extract_tendency_subject(query: str) -> str:
    """「〇〇の故障傾向は？」「〇〇に関する傾向は？」から対象設備・事象名を抽出する。"""
    q = re.sub(r'^\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:から|以降|以前|まで|頃)?の?', '', query).strip()
    patterns = [
        r'^(.+?)に関する(?:故障|障害|不具合)?(?:の)?(?:傾向|パターン|特徴)',
        r'^(.+?)の(?:故障|障害|不具合)?(?:の)?(?:傾向|パターン|特徴)',
        r'^(.+?)(?:は(?:どのような|なぜ|よく)|が(?:よく|多く))(?:故障|発生|起き)',
        r'^(.+?)(?:傾向|パターン|特徴)',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            s = m.group(1).strip()
            s = re.sub(r'[のがはにをで]$', '', s).strip()
            s = re.sub(r'^(?:について|に関して)', '', s).strip()
            if len(s) >= 2 and s not in _GENERIC_SUBJECTS:
                return s
    return ""


def is_equipment_tendency_query(query: str) -> bool:
    """特定設備・事象の傾向クエリを検出する（件数・集計・季節傾向クエリは除外）。"""
    if not any(kw in query for kw in _EQUIP_TENDENCY_KWS):
        return False
    subject = extract_tendency_subject(query)
    return bool(subject) and subject not in _GENERIC_SUBJECTS


def aggregate_by_equipment(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """特定設備傾向クエリに対し、全件スキャン＋集計済みコンテキストを返す。

    LLMへ渡すコンテキストは集計統計＋代表記録上限EQUIPMENT_TENDENCY_MAX_SAMPLES件に
    限定し、マッチ件数にかかわらずトークン数を一定に保つ。
    """
    subject = extract_tendency_subject(query)
    if not subject:
        kws = extract_equip_keywords(query)
        subject = kws[0] if kws else ""

    from_date, to_date = parse_date_filter(query)
    date_label = f"{from_date}〜{to_date}" if (from_date or to_date) else "全期間"

    # 検索バリアント生成（末尾の故障/障害除去・同義語展開）
    subject_nfkc = unicodedata.normalize("NFKC", subject)
    variants: list[str] = [subject_nfkc]
    base = re.sub(r'(?:故障|障害|不具合)+$', '', subject_nfkc).strip()
    if base and base != subject_nfkc and len(base) >= 2:
        variants.append(base)
    for group in _SYNONYM_GROUPS:
        if any(unicodedata.normalize("NFKC", v) in group for v in variants):
            for syn in group:
                if syn not in variants:
                    variants.append(syn)

    base = [d for d in _all_docs if (sf is None or sf.matches(d))]
    range_docs = [d for d in base if _doc_in_range(d, from_date, to_date)]

    def _equip_target(d: Document) -> str:
        return unicodedata.normalize("NFKC", d.metadata.get("故障", "") + "\n" + d.page_content)

    # Step1: フレーズ完全一致
    matched = [d for d in range_docs if any(v in _equip_target(d) for v in variants)]

    # Step2: フレーズ不一致時はトークンAND検索（「循環ポンプのVVVF故障」のような複合語対策）
    if not matched:
        try:
            tok_terms = [t for t in get_tokenizer().tokenize(subject_nfkc) if len(t) >= 2]
        except Exception:
            tok_terms = []
        tok_conds: list[tuple[str, ...]] = []
        for t in tok_terms:
            grp = next((g for g in _SYNONYM_GROUPS if t in g), None)
            tok_conds.append(tuple(grp) if grp else (t,))
        if len(tok_conds) >= 2:
            matched = [
                d for d in range_docs
                if all(any(syn in _equip_target(d) for syn in cond) for cond in tok_conds)
            ]

    if not matched:
        return f"「{subject}」に関する記録が見つかりませんでした（期間: {date_label}）。", []

    # 種別内訳
    shubetsu_counter: Counter = Counter(
        d.metadata.get("種別", "（不明）").strip() or "（不明）"
        for d in matched
    )

    # 月別件数
    month_counter: Counter = Counter()
    for d in matched:
        doc_date = _parse_date_str(d.metadata.get("日付", ""))
        if doc_date:
            month_counter[doc_date.month] += 1

    # 故障コード別内訳（上位10件）
    kosho_counter: Counter = Counter()
    kosho_samples: dict[str, list[str]] = {}
    for d in matched:
        ko = d.metadata.get("故障", "").strip()
        if ko:
            kosho_counter[ko] += 1
            slist = kosho_samples.setdefault(ko, [])
            if len(slist) < 2:
                m2 = re.search(r"^内容: (.+)$", d.page_content, re.MULTILINE)
                if m2:
                    snip = m2.group(1).strip()[:60]
                    if snip not in slist:
                        slist.append(snip)

    lines = [
        f"対象設備: 「{subject}」（検索語: {', '.join(variants)}）",
        f"集計期間: {date_label}",
        f"該当記録数: {len(matched)}件",
        "",
        "■ 種別内訳:",
    ]
    for k, v in shubetsu_counter.most_common():
        lines.append(f"  {k}: {v}件")

    if month_counter:
        lines += ["", "■ 月別記録件数:"]
        for mo in range(1, 13):
            cnt = month_counter.get(mo, 0)
            if cnt:
                lines.append(f"  {mo:2d}月: {cnt}件")

    if kosho_counter:
        lines += ["", "■ 故障コード/機器別内訳（上位10件）:"]
        for code, cnt in kosho_counter.most_common(10):
            lines.append(f"  「{code}」: {cnt}件")
            for s in kosho_samples.get(code, []):
                lines.append(f"    └ {s}")

    # 代表記録: 最新EQUIPMENT_TENDENCY_MAX_SAMPLES件に限定（トークン上限対策）
    sample_docs = sorted(matched, key=lambda d: d.metadata.get("日付", ""), reverse=True)
    sample_docs = sample_docs[:EQUIPMENT_TENDENCY_MAX_SAMPLES]

    lines += [
        "",
        f"■ 代表的な記録（最新{len(sample_docs)}件 / 全{len(matched)}件）:",
        "  ※ LLMへは代表記録のみ渡しています。傾向分析は集計統計を根拠にしてください。",
    ]
    for d in sample_docs:
        date_val = d.metadata.get("日付", "不明")
        shubetsu = d.metadata.get("種別", "")
        m2 = re.search(r"^内容: (.+)$", d.page_content, re.MULTILINE)
        snippet = m2.group(1)[:80] if m2 else ""
        lines.append(f"  {date_val} [{shubetsu}] — {snippet}")

    log.info("設備別傾向集計: subject=%s variants=%s 期間=%s マッチ=%d件 代表=%d件",
             subject, variants, date_label, len(matched), len(sample_docs))
    return "\n".join(lines), sample_docs


def fulltext_scan(q: str, sf: SearchFilter | None = None) -> list[Document]:
    """クエリのキーワードで全件テキストスキャンし、含む記録を日付降順で全件返す。

    検索優先順位:
      1. 日付除去後のクエリ文字列でフレーズ完全一致（最精度）
      2. 未ヒット → len>=2 のトークン全てを含む AND 検索
      3. それでも未ヒット → OR 検索（シングルキーワード時のみ。複数スペース区切り時はスキップ）
    """
    from_date, to_date = parse_date_filter(q)
    base = [d for d in _all_docs if (sf is None or sf.matches(d))]
    range_docs = [d for d in base if _doc_in_range(d, from_date, to_date)]
    period_label = f"{from_date}〜{to_date}" if (from_date or to_date) else "全期間"

    # 日付表現を除去してクエリを整理
    clean_q = re.sub(
        r'\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:以降|から|以前|まで|頃)?の?', '', q
    ).strip()
    clean_q_nfkc = unicodedata.normalize("NFKC", clean_q)

    def _target(d: Document) -> str:
        return unicodedata.normalize("NFKC", d.metadata.get("故障", "") + "\n" + d.page_content)

    # Step 1: フレーズ完全一致
    matched = [d for d in range_docs if clean_q_nfkc in _target(d)]
    if matched:
        matched.sort(key=lambda d: d.metadata.get("日付", ""), reverse=True)
        log.info("全件スキャン(フレーズ): q=%s 期間=%s マッチ=%d件", clean_q, period_label, len(matched))
        return matched

    # Step 2: トークン AND 検索
    try:
        tokens = [t for t in get_tokenizer().tokenize(clean_q) if len(t) >= 2]
    except Exception:
        tokens = []
    if not tokens:
        tokens = [clean_q] if clean_q else []
    tokens = _expand_synonyms(tokens)
    if not tokens:
        return []

    nfkc_tokens = [unicodedata.normalize("NFKC", t) for t in tokens]
    matched = [d for d in range_docs if all(t in _target(d) for t in nfkc_tokens)]
    mode = "AND"

    # 形態素解析結果が2トークン以上の場合はAND結果のみ返す（OR fallback なし）
    # ※ 以前はスペース区切りで判定していたが、日本語複合語がスペースなし1語扱いになり
    #    Step2 AND が全滅した際に OR フォールバックが誤起動するため、トークン数で判定する
    base_tok_count = len([t for t in get_tokenizer().tokenize(clean_q) if len(t) >= 2])
    multi_keyword = base_tok_count >= 2 or len(_split_and_keywords(q)) >= 2
    if not matched and not multi_keyword:
        # Step 3: OR 検索（シングルキーワード時のみ）
        matched = [d for d in range_docs if any(t in _target(d) for t in nfkc_tokens)]
        mode = "OR"

    matched.sort(key=lambda d: d.metadata.get("日付", ""), reverse=True)
    log.info("全件スキャン(%s): tokens=%s 期間=%s マッチ=%d件", mode, tokens, period_label, len(matched))
    return matched


def hybrid_search(
    query: str,
    from_date: date | None = None,
    to_date: date | None = None,
    retrieve_k: int = TOP_K_RETRIEVE,
    sf: SearchFilter | None = None,
    enforce_and: bool = False,
) -> list[Document]:
    bm25_query = normalize_query_for_bm25(query)
    and_terms = _split_and_keywords(query) if enforce_and else []
    multi_and = len(and_terms) >= 2
    needs_filter = (from_date or to_date) or (sf and not sf.is_empty()) or multi_and

    if needs_filter:
        # 日付・メタデータ・ANDキーワードで絞り込んだドキュメント集合で BM25 を再構築
        filtered = [
            d for d in _all_docs
            if _doc_in_range(d, from_date, to_date)
            and (sf is None or sf.matches(d))
            and (not multi_and or _doc_matches_all_terms(d, and_terms))
        ]
        if filtered:
            temp_bm25 = BM25Retriever.from_documents(
                filtered, k=retrieve_k, preprocess_func=get_tokenizer().tokenize
            )
            bm25_docs = temp_bm25.invoke(bm25_query)
        else:
            bm25_docs = []
        # ベクトル検索は多めに取得して Python 側でフィルタ
        vec_raw = _vector_store.similarity_search(query, k=min(len(filtered), retrieve_k * 3)) if _vector_store and filtered else []
        vec_docs = [
            d for d in vec_raw
            if _doc_in_range(d, from_date, to_date)
            and (sf is None or sf.matches(d))
            and (not multi_and or _doc_matches_all_terms(d, and_terms))
        ]
    else:
        if _bm25_retriever:
            _bm25_retriever.k = retrieve_k
            bm25_docs = _bm25_retriever.invoke(bm25_query)
        else:
            bm25_docs = []
        vec_docs = _vector_store.similarity_search(query, k=retrieve_k) if _vector_store else []

    # BM25 を 2 票・vector を 1 票で RRF することでキーワード一致の recall を優先する
    fused = rrf([bm25_docs, bm25_docs, vec_docs])
    return fused[:retrieve_k]


def rerank(query: str, docs: list[Document], recency_weight: float = 0.0) -> list[Document]:
    if not _reranker or not docs:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = list(_reranker.predict(pairs))

    # クエリ中の機器名を含むドキュメントにスコアブーストをかける
    # （例: 「UV計に関する故障」→ UV計を含まない故障記録が上位に来るのを抑止）
    equip_kws = extract_equip_keywords(query)
    if equip_kws:
        for i, doc in enumerate(docs):
            if any(kw in doc.page_content for kw in equip_kws):
                scores[i] += _EQUIP_BOOST

    if recency_weight > 0:
        today = date.today()
        scores = [s + recency_weight * recency_score(doc, today) for s, doc in zip(scores, docs)]
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked]


def search_docs(
    query: str,
    top_k: int = TOP_K_FINAL,
    recency_weight: float = 0.0,
    sf: SearchFilter | None = None,
    enforce_and: bool = False,
) -> list[Document]:
    from_date, to_date = parse_date_filter(query)
    if from_date or to_date:
        log.info("日付フィルタ: %s 〜 %s", from_date, to_date)
    retrieve_k = max(top_k, TOP_K_RETRIEVE)
    candidates = hybrid_search(query, from_date, to_date, retrieve_k, sf=sf, enforce_and=enforce_and)
    reranked = rerank(query, candidates, recency_weight)
    seen_keys: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in reranked:
        date = doc.metadata.get("日付", "")
        m = re.search(r"^内容: (.+)$", doc.page_content, re.MULTILINE)
        content_text = m.group(1) if m else doc.page_content
        key = (date, content_text)
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(doc)
    return unique[:top_k]


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "あなたは水処理施設の引継ぎノート管理アシスタントです。\n"
     "【最重要】回答は必ず日本語のみで記述すること。中国語・英語の使用を固く禁ずる。\n"
     "【文字種ルール】すべての文にひらがなまたはカタカナを必ず含めること。漢字のみの文（中国語文）は1文字も出力しないこと。\n"
     "请只使用日语回答，严禁使用中文或英语。每个句子必须包含平假名或片假名。\n"
     "提供された引継ぎ記録のみを根拠に、具体的かつ正確に答えてください。\n"
     "記録に情報がない場合は「記録に該当情報はありません」と答えてください。"),
    ("human",
     "【引継ぎ記録】\n{context}\n\n【質問】\n{question}\n\n【日本語のみで回答してください】\n"),
])

SPECIFIC_COUNT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "あなたは水処理施設の引継ぎノート管理アシスタントです。\n"
     "【最重要】回答は必ず日本語のみで記述すること。中国語・英語の使用を固く禁ずる。\n"
     "【文字種ルール】すべての文にひらがなまたはカタカナを必ず含めること。漢字のみの文（中国語文）は1文字も出力しないこと。\n"
     "请只使用日语回答，严禁使用中文或英语。每个句子必须包含平假名或片假名。\n"
     "以下の集計データを元に、質問に対して正確な件数を回答してください。\n"
     "【重要ルール】\n"
     "・「含む記録件数: N件」と書いてある N をそのまま合計件数として回答すること。種別内訳の数値に置き換えてはならない\n"
     "・故障・故障処置・報告・処置など種別を問わず、検索語を含む全記録の合計件数を答えること\n"
     "・回答の冒頭に「〇〇は全体でX件発生しています。」という形式で合計件数を明示すること\n"
     "・「故障コード/機器別内訳」がある場合は、記録内容例から機器名・設備名を読み取り「内訳：機器A X件、機器B Y件…」という形式で示すこと\n"
     "・機器名が読み取れない場合はコード番号と件数をそのまま報告すること\n"
     "・該当記録がない場合は「記録に該当情報はありません」と答えること"),
    ("human",
     "【集計データ】\n{context}\n\n【質問】\n{question}\n\n"
     "【合計件数を冒頭に明示し、機器別内訳を含めて日本語のみで回答してください】\n"),
])

AGGREGATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "あなたは水処理施設の引継ぎノート管理アシスタントです。\n"
     "【最重要】回答は必ず日本語のみで記述すること。中国語・英語の使用を固く禁ずる。\n"
     "【文字種ルール】すべての文にひらがなまたはカタカナを必ず含めること。漢字のみの文（中国語文）は1文字も出力しないこと。\n"
     "请只使用日语回答，严禁使用中文或英语。每个句子必须包含平假名或片假名。\n"
     "以下の集計データを分析して質問に答えてください。\n"
     "【重要ルール】\n"
     "・数字コードや識別番号だけを回答しないこと。必ず「記録内容」から機器名・設備名・故障の種類を読み取って説明すること\n"
     "・例: 「1470」だけでなく、記録内容から「〇〇設備（コード1470）」のように機器名を明示すること\n"
     "・件数・発生傾向・主な故障内容を含めて具体的に説明すること\n"
     "・記録内容から機器名が読み取れない場合のみ、コードと件数を報告すること"),
    ("human",
     "【集計データ】\n{context}\n\n【質問】\n{question}\n\n【機器名・故障内容・件数を含めて日本語で回答してください】\n"),
])


SEASONAL_TENDENCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "あなたは水処理施設の引継ぎノート管理アシスタントです。\n"
     "【最重要】回答は必ず日本語のみで記述すること。中国語・英語の使用を固く禁ずる。\n"
     "【文字種ルール】すべての文にひらがなまたはカタカナを必ず含めること。漢字のみの文（中国語文）は1文字も出力しないこと。\n"
     "请只使用日语回答，严禁使用中文或英语。每个句子必须包含平假名或片假名。\n"
     "以下の季節別集計データをもとに、質問に答えてください。\n"
     "【重要ルール】\n"
     "・月別・季節別の件数データを根拠に、発生が多い時期・少ない時期を具体的な数字で説明すること\n"
     "・対象季節の頻発故障コード欄にある記録内容例から機器名・故障種別を読み取り、どのような故障が多いか説明すること\n"
     "・他の季節との件数比較（多い/少ない）を含めること\n"
     "・記録が存在しない季節・故障については「記録なし」と明示すること"),
    ("human",
     "【季節別集計データ】\n{context}\n\n【質問】\n{question}\n\n"
     "【月別件数と季節ごとの故障傾向を具体的に、日本語のみで回答してください】\n"),
])


EQUIPMENT_TENDENCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "あなたは水処理施設の引継ぎノート管理アシスタントです。\n"
     "【最重要】回答は必ず日本語のみで記述すること。中国語・英語の使用を固く禁ずる。\n"
     "【文字種ルール】すべての文にひらがなまたはカタカナを必ず含めること。漢字のみの文（中国語文）は1文字も出力しないこと。\n"
     "请只使用日语回答，严禁使用中文或英语。每个句子必须包含平假名或片假名。\n"
     "以下の設備別集計データをもとに、対象設備の故障・トラブルの傾向を分析してください。\n"
     "【重要ルール】\n"
     "・「該当記録数」の総件数と月別件数を根拠として傾向を分析すること（代表記録は一部サンプルであり全件ではない）\n"
     "・月別記録件数から発生が多い時期・少ない時期を具体的な数字で説明すること\n"
     "・故障コード/機器別内訳から、どの機器・設備で特に多く発生しているか説明すること\n"
     "・代表的な記録から具体的な故障内容・対処パターン・再発防止のポイントを読み取って説明すること\n"
     "・記録が少ない・存在しない場合は「記録が少ないため傾向分析が困難」と明示すること\n"
     "・回答は以下の構成で箇条書きにすること：\n"
     "  １．発生頻度（件数・時期）\n"
     "  ２．再発状況（繰り返しパターン）\n"
     "  ３．主な原因\n"
     "  ４．対応・対処方法\n"
     "・最後に全体の傾向をまとめた結論を1〜2文で述べること"),
    ("human",
     "【設備別集計データ】\n{context}\n\n【質問】\n{question}\n\n"
     "【月別傾向・故障パターン・対処方法を含めて日本語のみで回答してください】\n"),
])


def is_likely_chinese(text: str) -> bool:
    """ひらがな・カタカナが含まれず漢字のみの場合に中国語と判定する"""
    stripped = text.strip()
    if len(stripped) < 10:
        return False
    has_cjk = any('一' <= c <= '鿿' for c in stripped)
    has_hiragana = any('ぁ' <= c <= 'ゖ' for c in stripped)
    has_katakana = any('゠' <= c <= 'ヿ' for c in stripped)
    return has_cjk and not has_hiragana and not has_katakana


def _has_chinese_sentences(text: str) -> bool:
    """テキスト中に中国語文（ひらがな・カタカナを含まない5字以上の段落）が混在するか判定する。
    日中混在テキストにも対応（is_likely_chinese はテキスト全体判定のため混在を見落とす）。"""
    for seg in re.split(r'[\n。！？!?、]', text):
        if is_likely_chinese(seg):
            return True
    return False


def _filter_chinese_sentences(text: str) -> str:
    """テキストから中国語行・中国語文を除去し、日本語部分のみを返す。"""
    filtered_lines = []
    for line in text.split('\n'):
        if is_likely_chinese(line):
            log.warning("中国語行を除去: %.40s...", line)
            continue
        # 行内の「。」区切り文もチェック
        parts = line.split('。')
        kept = [p for p in parts if not is_likely_chinese(p)]
        if kept:
            filtered_lines.append('。'.join(kept))
    return '\n'.join(filtered_lines)


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    return (
        {
            "context": lambda q: format_docs(search_docs(q)),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | _llm
        | StrOutputParser()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embeddings, _llm, _reranker, LLM_MODEL, LLM_REASONING, EQUIPMENT_TENDENCY_MAX_SAMPLES
    try:
        log.info("起動中: モデル初期化...")
        config = load_config()
        LLM_MODEL = config.get("llm_model", LLM_MODEL)
        LLM_REASONING = config.get("llm_reasoning", LLM_REASONING)
        EQUIPMENT_TENDENCY_MAX_SAMPLES = int(config.get("equip_tendency_samples", EQUIPMENT_TENDENCY_MAX_SAMPLES))
        log.info("使用モデル: %s (reasoning=%s)", LLM_MODEL, LLM_REASONING)

        # DB・トークナイザ初期化
        init_db()
        sudachi_mode = config.get("sudachi_mode", "C")
        user_dic = str(USER_DIC_PATH) if USER_DIC_PATH.exists() else None
        init_tokenizer(mode=sudachi_mode, user_dic_path=user_dic)
        log.info("トークナイザ初期化完了 (mode=%s, user_dic=%s)", sudachi_mode, user_dic)

        _embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        _llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, reasoning=LLM_REASONING or None)
        log.info("Re-ranker ロード中: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        log.info("インデックス構築中...")
        build_index()
        log.info("起動完了 — bm25=%d reranker=%s", len(_all_docs), _reranker is not None)
    except Exception:
        log.exception("lifespan 初期化エラー")
        raise
    yield


app = FastAPI(title="引継ぎノート RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K_FINAL
    recency_weight: float = 0.0
    filter_kinmu: list[str] = []
    filter_shubetsu: list[str] = []
    filter_year: int | None = None
    filter_month: int | None = None
    filter_day: int | None = None


class ModelChangeRequest(BaseModel):
    model: str


class EquipSamplesRequest(BaseModel):
    samples: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問を入力してください")

    def _invoke_and_filter(chain, inputs: dict) -> str:
        answer = chain.invoke(inputs)
        if is_likely_chinese(answer) or _has_chinese_sentences(answer):
            log.warning("中国語混入を検出、フィルタ処理します")
            answer = _filter_chinese_sentences(answer).strip()
        return answer

    if is_specific_failure_count_query(req.question):
        context, docs = count_specific_failure(req.question)
        try:
            answer = _invoke_and_filter(
                SPECIFIC_COUNT_PROMPT | _llm | StrOutputParser(),
                {"context": context, "question": req.question},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLMエラー: {e}")
        sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
        return QueryResponse(answer=answer, sources=sources)

    if is_aggregation_query(req.question):
        context, docs = aggregate_for_query(req.question)
        try:
            answer = _invoke_and_filter(
                AGGREGATION_PROMPT | _llm | StrOutputParser(),
                {"context": context, "question": req.question},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLMエラー: {e}")
        sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
        return QueryResponse(answer=answer, sources=sources)

    if is_seasonal_tendency_query(req.question):
        context, docs = aggregate_by_season(req.question)
        try:
            answer = _invoke_and_filter(
                SEASONAL_TENDENCY_PROMPT | _llm | StrOutputParser(),
                {"context": context, "question": req.question},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLMエラー: {e}")
        sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
        return QueryResponse(answer=answer, sources=sources)

    if is_equipment_tendency_query(req.question):
        context, docs = aggregate_by_equipment(req.question)
        try:
            answer = _invoke_and_filter(
                EQUIPMENT_TENDENCY_PROMPT | _llm | StrOutputParser(),
                {"context": context, "question": req.question},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLMエラー: {e}")
        sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
        return QueryResponse(answer=answer, sources=sources)

    try:
        docs = search_docs(req.question, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {e}")
    try:
        raw = build_rag_chain().invoke(req.question)
        answer = _filter_chinese_sentences(raw).strip() if (
            is_likely_chinese(raw) or _has_chinese_sentences(raw)
        ) else raw
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLMエラー: {e}")
    sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
    return QueryResponse(answer=answer, sources=sources)


@app.get("/api/search")
def search_only(
    q: str,
    top_k: int = TOP_K_FINAL,
    scan: bool = False,
    kinmu: list[str] = Query(default=[]),
    shubetsu: list[str] = Query(default=[]),
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="クエリを入力してください")
    sf = _sf_from_params(kinmu, shubetsu, year, month, day)
    sf_or_none = sf if not sf.is_empty() else None
    try:
        if scan:
            docs = fulltext_scan(q, sf=sf_or_none)
            route = "fullscan"
        elif is_specific_failure_count_query(q):
            _, docs = count_specific_failure(q, sf=sf_or_none)
            route = "count"
        elif is_aggregation_query(q):
            _, docs = aggregate_for_query(q, sf=sf_or_none)
            route = "aggregation"
        elif is_seasonal_tendency_query(q):
            _, docs = aggregate_by_season(q, sf=sf_or_none)
            route = "seasonal"
        elif is_equipment_tendency_query(q):
            _, docs = aggregate_by_equipment(q, sf=sf_or_none)
            route = "equip_tendency"
        else:
            docs = search_docs(q, top_k, sf=sf_or_none, enforce_and=True)
            route = "rag"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {e}")
    return {
        "results": [{"document": d.page_content, "metadata": d.metadata} for d in docs],
        "route": route,
        "filter_label": sf.label(),
    }


@app.get("/api/filter-options")
def filter_options():
    """フィルター用の選択肢（年リスト）を返す。"""
    years: set[int] = set()
    for d in _all_docs:
        parsed = _parse_date_str(d.metadata.get("日付", ""))
        if parsed:
            years.add(parsed.year)
    return {"years": sorted(years)}


@app.get("/api/health")
def health():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        existing_names = [c.name for c in client.list_collections()]
        records = client.get_collection(COLLECTION_NAME).count() if COLLECTION_NAME in existing_names else 0
        return {
            "status": "ok",
            "records": records,
            "bm25_indexed": len(_all_docs),
            "reranker_loaded": _reranker is not None,
            "llm_model": LLM_MODEL,
            "llm_reasoning": LLM_REASONING,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/db-status")
def db_status():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        existing_names = [c.name for c in client.list_collections()]
        records = client.get_collection(COLLECTION_NAME).count() if COLLECTION_NAME in existing_names else 0
        return {"records": records, "bm25_indexed": len(_all_docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_ingest_thread(csv_path: Path, rebuild: bool) -> None:
    def progress_cb(done: int, total: int, msg: str) -> None:
        _set_ingest_state(progress=done, total=total, message=msg)
        log.info("[ingest] %s", msg)

    try:
        result = run_ingest(csv_path, progress_cb, rebuild=rebuild)
        _set_ingest_state(
            running=False,
            done=True,
            result=result,
            message=f"完了: {result['added']} 件追加 / {result['skipped']} 件スキップ / DB合計 {result['total']} 件",
        )
        log.info("[ingest] 完了: %s", result)
        try:
            build_index()
            log.info("[ingest] インデックス再構築完了: %d 件", len(_all_docs))
        except Exception as e:
            log.error("[ingest] インデックス再構築失敗: %s", e)
    except Exception as e:
        log.exception("[ingest] エラー")
        _set_ingest_state(
            running=False,
            done=True,
            error=str(e),
            message=f"エラー: {e}",
        )


@app.post("/api/ingest")
async def start_ingest(file: UploadFile = File(...), rebuild: bool = False):
    with _ingest_lock:
        if _ingest_state["running"]:
            raise HTTPException(status_code=409, detail="インジェスト実行中です")

    fname = file.filename or ""
    if not (fname.endswith(".csv") or fname.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="CSVまたはXLSXファイルを選択してください")

    suffix = ".xlsx" if fname.endswith(".xlsx") else ".csv"
    csv_path = UPLOAD_DIR / f"latest{suffix}"
    csv_path.write_bytes(await file.read())

    _set_ingest_state(
        running=True,
        done=False,
        error=None,
        progress=0,
        total=0,
        result=None,
        message="開始中...",
    )

    threading.Thread(target=_run_ingest_thread, args=(csv_path, rebuild), daemon=True).start()
    return {"status": "started", "filename": file.filename}


@app.get("/api/ingest/stream")
async def ingest_stream():
    async def generator():
        while True:
            with _ingest_lock:
                state = dict(_ingest_state)
            yield f"data: {json.dumps(state, ensure_ascii=False)}\n\n"
            if state.get("done") or not state.get("running"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ingest/status")
def ingest_status():
    with _ingest_lock:
        return dict(_ingest_state)


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問を入力してください")

    sf = _sf_from_params(req.filter_kinmu, req.filter_shubetsu, req.filter_year, req.filter_month, req.filter_day)
    sf_or_none = sf if not sf.is_empty() else None

    is_specific_count = is_specific_failure_count_query(req.question)
    is_agg = not is_specific_count and is_aggregation_query(req.question)
    is_seasonal = not is_specific_count and not is_agg and is_seasonal_tendency_query(req.question)
    is_equip = not is_specific_count and not is_agg and not is_seasonal and is_equipment_tendency_query(req.question)

    LLM_CONTEXT_K = 50  # LLM に渡す文書数上限（judge と一致させる）

    if is_specific_count:
        context, docs = count_specific_failure(req.question, sf=sf_or_none)
    elif is_agg:
        context, docs = aggregate_for_query(req.question, sf=sf_or_none)
    elif is_seasonal:
        context, docs = aggregate_by_season(req.question, sf=sf_or_none)
    elif is_equip:
        context, docs = aggregate_by_equipment(req.question, sf=sf_or_none)
    else:
        try:
            docs = search_docs(req.question, req.top_k, req.recency_weight, sf=sf_or_none)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"検索エラー: {e}")
        context = format_docs(docs[:LLM_CONTEXT_K])

    sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
    filter_suffix = f"\n\n（絞り込み条件: {sf.label()}）" if sf_or_none else ""

    async def generator():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
        if is_specific_count:
            prompt = SPECIFIC_COUNT_PROMPT
        elif is_agg:
            prompt = AGGREGATION_PROMPT
        elif is_seasonal:
            prompt = SEASONAL_TENDENCY_PROMPT
        elif is_equip:
            prompt = EQUIPMENT_TENDENCY_PROMPT
        else:
            prompt = RAG_PROMPT
        chain = prompt | _llm | StrOutputParser()

        chunks: list[str] = []
        try:
            async for chunk in chain.astream({"context": context, "question": req.question}):
                if await request.is_disconnected():
                    return
                chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
        except Exception:
            pass

        full_text = "".join(chunks)
        if is_likely_chinese(full_text) or _has_chinese_sentences(full_text):
            log.warning("中国語混入を検出、フィルタ済みテキストに置換します")
            cleaned = _filter_chinese_sentences(full_text).strip()
            if cleaned:
                yield f"data: {json.dumps({'type': 'reset'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': cleaned}, ensure_ascii=False)}\n\n"
            else:
                log.warning("フィルタ後テキストが空のためリトライします")
                yield f"data: {json.dumps({'type': 'reset'}, ensure_ascii=False)}\n\n"
                try:
                    async for chunk in chain.astream({"context": context, "question": req.question}):
                        if await request.is_disconnected():
                            return
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
                except Exception:
                    pass

        if filter_suffix:
            yield f"data: {json.dumps({'type': 'chunk', 'text': filter_suffix}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reload-index")
def reload_index():
    try:
        build_index()
        return {"status": "ok", "records": len(_all_docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ollama-models")
def get_ollama_models():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=15
        )
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return {"models": models, "current": LLM_MODEL}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ollama コマンドが見つかりません")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="ollama list がタイムアウトしました")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model")
def get_model():
    return {"model": LLM_MODEL, "reasoning": LLM_REASONING}


@app.post("/api/model")
def change_model(req: ModelChangeRequest):
    global _llm, LLM_MODEL
    if not req.model.strip():
        raise HTTPException(status_code=400, detail="モデル名を指定してください")
    with _model_lock:
        try:
            new_llm = OllamaLLM(model=req.model, base_url=OLLAMA_BASE_URL, reasoning=LLM_REASONING or None)
            _llm = new_llm
            LLM_MODEL = req.model
            config = load_config()
            config["llm_model"] = req.model
            save_config(config)
            log.info("モデルを変更しました: %s", req.model)
            return {"status": "ok", "model": req.model}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"モデル変更エラー: {e}")


class ReasoningRequest(BaseModel):
    enabled: bool


@app.get("/api/reasoning")
def get_reasoning():
    return {"enabled": LLM_REASONING}


@app.post("/api/reasoning")
def set_reasoning(req: ReasoningRequest):
    global _llm, LLM_REASONING
    with _model_lock:
        try:
            new_llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, reasoning=req.enabled or None)
            _llm = new_llm
            LLM_REASONING = req.enabled
            config = load_config()
            config["llm_reasoning"] = req.enabled
            save_config(config)
            log.info("reasoning を変更しました: %s", req.enabled)
            return {"status": "ok", "enabled": req.enabled}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"reasoning 変更エラー: {e}")


@app.get("/api/equip-samples")
def get_equip_samples():
    return {"samples": EQUIPMENT_TENDENCY_MAX_SAMPLES}


@app.post("/api/equip-samples")
def set_equip_samples(req: EquipSamplesRequest):
    global EQUIPMENT_TENDENCY_MAX_SAMPLES
    valid = {10, 20, 30, 50, 100}
    if req.samples not in valid:
        raise HTTPException(status_code=400, detail=f"サンプル数は {sorted(valid)} のいずれかを指定してください")
    EQUIPMENT_TENDENCY_MAX_SAMPLES = req.samples
    config = load_config()
    config["equip_tendency_samples"] = req.samples
    save_config(config)
    log.info("設備傾向サンプル数を変更しました: %d", req.samples)
    return {"status": "ok", "samples": req.samples}


# ─────────────────────────────────────────────────────────────────────────────
# ユーザー辞書管理 API
# ─────────────────────────────────────────────────────────────────────────────

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


class SudachiModeRequest(BaseModel):
    mode: str


@app.get("/api/dict")
def dict_list(
    search: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    return get_all(search=search, page=page, page_size=page_size)


@app.post("/api/dict", status_code=201)
def dict_add(req: DictEntryCreate):
    if not req.surface.strip():
        raise HTTPException(status_code=400, detail="surface は必須です")
    if not req.reading.strip():
        raise HTTPException(status_code=400, detail="reading は必須です")
    entry_id = add_entry(
        surface=req.surface,
        reading=req.reading,
        pos=req.pos,
        cost=req.cost,
        normalized=req.normalized,
        enabled=req.enabled,
    )
    return {"id": entry_id}


@app.get("/api/dict/check")
def dict_check(surface: str = Query(...)):
    entries = find_by_surface(surface.strip())
    return {"exists": len(entries) > 0, "entries": entries}


@app.put("/api/dict/{entry_id}")
def dict_update(entry_id: int, req: DictEntryUpdate):
    updates = req.model_dump(exclude_none=True)
    if not update_entry(entry_id, **updates):
        raise HTTPException(status_code=404, detail="エントリが見つかりません")
    return {"status": "ok"}


@app.delete("/api/dict/{entry_id}")
def dict_delete(entry_id: int):
    if not delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="エントリが見つかりません")
    return {"status": "ok"}


@app.post("/api/dict/import")
async def dict_import(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    count = import_from_csv(rows)
    return {"imported": count}


@app.get("/api/dict/export")
def dict_export():
    csv_text = export_to_csv()
    return StreamingResponse(
        io.StringIO(csv_text),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=user_dict.csv"},
    )


@app.post("/api/dict/rebuild")
def dict_rebuild():
    result = rebuild_and_reload()
    if result["status"] == "ok":
        try:
            build_index()
        except Exception as e:
            log.error("dict_rebuild: インデックス再構築失敗: %s", e)
            result["message"] += f"（BM25再構築エラー: {e}）"
    return result


@app.post("/api/dict/test")
def dict_test(req: dict):
    text = (req.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text を指定してください")
    tok = get_tokenizer()
    tokens = tok.tokenize(text)
    return {"tokens": tokens, "mode": tok.mode}


@app.get("/api/dict/mode")
def dict_get_mode():
    tok = get_tokenizer()
    return {"mode": tok.mode}


@app.post("/api/dict/mode")
def dict_set_mode(req: SudachiModeRequest):
    mode = req.mode.upper()
    if mode not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="mode は A/B/C のいずれかを指定してください")
    user_dic = str(USER_DIC_PATH) if USER_DIC_PATH.exists() else None
    init_tokenizer(mode=mode, user_dic_path=user_dic)
    config = load_config()
    config["sudachi_mode"] = mode
    save_config(config)
    try:
        build_index()
    except Exception as e:
        log.error("dict_set_mode: インデックス再構築失敗: %s", e)
    return {"status": "ok", "mode": mode}


_SUGGEST_PROMPT = """\
以下は水処理施設の引き継ぎノートの記録です。
この文書に含まれる専門用語・設備名・機器名・現場略語を抽出してください。

【抽出条件】
・一般的な国語辞典に載っていない専門用語
・設備名・機器名（例: UV計, 冷却水ポンプ, 最終沈殿池）
・現場略語（例: 最沈, CWP, ばっ気）
・アルファベット混じりの機器名（例: DO計, MLSS計）
・複合語で分割されると意味が変わる語

【出力形式】JSONのみ出力。説明文・Markdownコードブロック不要。
{{
  "candidates": [
    {{
      "surface": "文書中の表記そのまま",
      "normalized": "統一したい正規化表記（不明なら空文字）",
      "reason": "候補とした理由（短く）"
    }}
  ]
}}

【引き継ぎノート】
{sampled_text}
"""


@app.post("/api/dict/suggest")
async def dict_suggest():
    if _llm is None:
        raise HTTPException(status_code=503, detail="LLM が初期化されていません")

    config = load_config()
    sample_size: int = config.get("suggest_sample_size", 200)

    if not _all_docs:
        raise HTTPException(status_code=503, detail="ドキュメントが読み込まれていません")

    sample = random.sample(_all_docs, min(sample_size, len(_all_docs)))
    sampled_text = "\n---\n".join(d.page_content[:300] for d in sample)

    prompt = _SUGGEST_PROMPT.format(sampled_text=sampled_text)
    try:
        raw = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _llm.invoke(prompt)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 呼び出しエラー: {e}")

    m = re.search(r'\{[\s\S]*\}', raw)
    if not m:
        raise HTTPException(status_code=500, detail=f"LLM の出力から JSON を抽出できませんでした: {raw[:300]}")
    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON パースエラー: {e}\n{m.group()[:300]}")

    candidates_raw = parsed.get("candidates", [])
    existing = get_all_surfaces()
    candidates = [c for c in candidates_raw if c.get("surface") and c["surface"] not in existing]

    return {
        "candidates": candidates,
        "sampled": len(sample),
        "excluded_existing": len(candidates_raw) - len(candidates),
    }


@app.post("/api/dict/suggest/register")
def dict_suggest_register(req: SuggestRegisterRequest):
    existing = get_all_surfaces()
    registered = 0
    skipped = 0
    for e in req.entries:
        if e.surface in existing:
            skipped += 1
            continue
        add_entry(
            surface=e.surface,
            reading=e.reading,
            pos=e.pos,
            cost=e.cost,
            normalized=e.normalized,
        )
        existing.add(e.surface)
        registered += 1
    return {"registered": registered, "skipped": skipped}
