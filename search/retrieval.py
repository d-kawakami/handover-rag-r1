"""ハイブリッド検索（BM25 + ベクトル + RRF + Re-ranking）と全件スキャン。"""
from __future__ import annotations

import logging
import re
from datetime import date as date_t

import chromadb
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import TOP_K_FINAL, TOP_K_RETRIEVE
from ingest import CHROMA_PATH, COLLECTION_NAME
from normalizer import normalize_notation
from search.dates import (
    doc_in_range,
    normalize_query_for_bm25,
    parse_date_filter,
    recency_score,
)
from search.filters import SearchFilter, doc_matches_all_terms, split_and_keywords
from search.patterns import EQUIP_BOOST, expand_synonyms, extract_equip_keywords
from state import state
from tokenizer import get_tokenizer

log = logging.getLogger(__name__)


def build_index() -> None:
    """ChromaDB から全ドキュメントを読み込み、BM25 / ベクトルストアを再構築する。"""
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
    state.all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]
    state.bm25_retriever = BM25Retriever.from_documents(
        state.all_docs,
        k=TOP_K_RETRIEVE,
        preprocess_func=get_tokenizer().tokenize,
    )
    state.vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=state.embeddings,
        persist_directory=CHROMA_PATH,
    )
    log.info("インデックス構築完了: %d 件", count)


def rrf(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: 複数ランキングを 1/(k+rank) でスコア合算する。"""
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


def hybrid_search(
    query: str,
    from_date: date_t | None = None,
    to_date: date_t | None = None,
    retrieve_k: int = TOP_K_RETRIEVE,
    sf: SearchFilter | None = None,
    enforce_and: bool = False,
) -> list[Document]:
    bm25_query = normalize_query_for_bm25(query)
    vec_query = normalize_notation(query)
    and_terms = split_and_keywords(query) if enforce_and else []
    multi_and = len(and_terms) >= 2
    needs_filter = (from_date or to_date) or (sf and not sf.is_empty()) or multi_and

    if needs_filter:
        filtered = [
            d for d in state.all_docs
            if doc_in_range(d, from_date, to_date)
            and (sf is None or sf.matches(d))
            and (not multi_and or doc_matches_all_terms(d, and_terms))
        ]
        if filtered:
            temp_bm25 = BM25Retriever.from_documents(
                filtered, k=retrieve_k, preprocess_func=get_tokenizer().tokenize
            )
            bm25_docs = temp_bm25.invoke(bm25_query)
        else:
            bm25_docs = []
        vec_raw = (
            state.vector_store.similarity_search(vec_query, k=min(len(filtered), retrieve_k * 3))
            if state.vector_store and filtered else []
        )
        vec_docs = [
            d for d in vec_raw
            if doc_in_range(d, from_date, to_date)
            and (sf is None or sf.matches(d))
            and (not multi_and or doc_matches_all_terms(d, and_terms))
        ]
    else:
        if state.bm25_retriever:
            state.bm25_retriever.k = retrieve_k
            bm25_docs = state.bm25_retriever.invoke(bm25_query)
        else:
            bm25_docs = []
        vec_docs = (
            state.vector_store.similarity_search(vec_query, k=retrieve_k)
            if state.vector_store else []
        )

    # BM25 を 2 票・vector を 1 票で RRF することでキーワード一致の recall を優先する
    fused = rrf([bm25_docs, bm25_docs, vec_docs])
    return fused[:retrieve_k]


def rerank(query: str, docs: list[Document], recency_weight: float = 0.0) -> list[Document]:
    if not state.reranker or not docs:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = list(state.reranker.predict(pairs))

    # クエリ中の機器名を含むドキュメントにスコアブーストをかける
    equip_kws = extract_equip_keywords(query)
    if equip_kws:
        for i, doc in enumerate(docs):
            if any(kw in doc.page_content for kw in equip_kws):
                scores[i] += EQUIP_BOOST

    if recency_weight > 0:
        today = date_t.today()
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
        date_val = doc.metadata.get("日付", "")
        m = re.search(r"^内容: (.+)$", doc.page_content, re.MULTILINE)
        content_text = m.group(1) if m else doc.page_content
        key = (date_val, content_text)
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(doc)
    return unique[:top_k]


def fulltext_scan(q: str, sf: SearchFilter | None = None) -> list[Document]:
    """クエリのキーワードで全件テキストスキャンし、含む記録を日付降順で全件返す。

    検索優先順位:
      1. 日付除去後のクエリ文字列でフレーズ完全一致（最精度）
      2. 未ヒット → len>=2 のトークン全てを含む AND 検索
      3. それでも未ヒット → OR 検索（シングルキーワード時のみ。複数スペース区切り時はスキップ）
    """
    from_date, to_date = parse_date_filter(q)
    base = [d for d in state.all_docs if (sf is None or sf.matches(d))]
    range_docs = [d for d in base if doc_in_range(d, from_date, to_date)]
    period_label = f"{from_date}〜{to_date}" if (from_date or to_date) else "全期間"

    clean_q = re.sub(
        r'\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:以降|から|以前|まで|頃)?の?', '', q
    ).strip()
    clean_q_norm = normalize_notation(clean_q)

    def _target(d: Document) -> str:
        return normalize_notation(d.metadata.get("故障", "") + "\n" + d.page_content)

    # Step 1: フレーズ完全一致
    matched = [d for d in range_docs if clean_q_norm in _target(d)]
    if matched:
        matched.sort(key=lambda d: d.metadata.get("日付", ""), reverse=True)
        log.info("全件スキャン(フレーズ): q=%s 期間=%s マッチ=%d件",
                 clean_q_norm, period_label, len(matched))
        return matched

    # Step 2: トークン AND 検索
    try:
        tokens = [t for t in get_tokenizer().tokenize(clean_q) if len(t) >= 2]
    except Exception:
        tokens = []
    if not tokens:
        tokens = [clean_q] if clean_q else []
    tokens = expand_synonyms(tokens)
    if not tokens:
        return []

    norm_tokens = [normalize_notation(t) for t in tokens]
    matched = [d for d in range_docs if all(t in _target(d) for t in norm_tokens)]
    mode = "AND"

    # 形態素解析結果が2トークン以上の場合はAND結果のみ返す（OR fallback なし）
    base_tok_count = len([t for t in get_tokenizer().tokenize(clean_q) if len(t) >= 2])
    multi_keyword = base_tok_count >= 2 or len(split_and_keywords(q)) >= 2
    if not matched and not multi_keyword:
        # Step 3: OR 検索（シングルキーワード時のみ）
        matched = [d for d in range_docs if any(t in _target(d) for t in norm_tokens)]
        mode = "OR"

    matched.sort(key=lambda d: d.metadata.get("日付", ""), reverse=True)
    log.info("全件スキャン(%s): tokens=%s 期間=%s マッチ=%d件",
             mode, tokens, period_label, len(matched))
    return matched


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
