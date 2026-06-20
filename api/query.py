"""質問エンドポイント: /api/query, /api/query/stream"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser

from api._routing import classify_query
from api.schemas import QueryRequest, QueryResponse
from config import LLM_CONTEXT_K
from language import has_chinese_mix, filter_chinese_sentences, is_likely_chinese, sanitize_if_chinese
from prompts import RAG_PROMPT
from search.filters import sf_from_params
from search.retrieval import format_docs, search_docs
from state import state

log = logging.getLogger(__name__)

router = APIRouter()


def _log_rag_context(route_name: str, question: str, docs: list, context: str) -> None:
    """LLM に渡す context のダイジェストを INFO ログに残す。

    「該当なし」が返るとき、本当に context に関連語が無かったのか、それとも
    context にあるのに LLM が見落としたのかを後追いできる。
    """
    preview = context[:300].replace("\n", " ⏎ ")
    log.info(
        "RAG[%s] q=%r docs=%d ctx_chars=%d preview=%s",
        route_name, question, len(docs), len(context), preview,
    )


@router.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問を入力してください")

    sf = sf_from_params(
        req.filter_kinmu, req.filter_shubetsu,
        req.filter_year, req.filter_month, req.filter_day,
    )
    sf_or_none = sf if not sf.is_empty() else None

    route = classify_query(req.question)

    if route is not None:
        context, docs = route.handler(req.question, sf_or_none)
        _log_rag_context(route.name, req.question, docs, context)
        try:
            chain = route.prompt | state.llm | StrOutputParser()
            answer = sanitize_if_chinese(chain.invoke({"context": context, "question": req.question}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLMエラー: {e}")
    else:
        try:
            docs = search_docs(
                req.question, req.top_k, req.recency_weight, sf=sf_or_none,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"検索エラー: {e}")
        try:
            context = format_docs(docs)
            _log_rag_context("rag", req.question, docs, context)
            chain = RAG_PROMPT | state.llm | StrOutputParser()
            answer = sanitize_if_chinese(chain.invoke({"context": context, "question": req.question}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLMエラー: {e}")

    sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
    return QueryResponse(answer=answer, sources=sources)


@router.post("/api/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="質問を入力してください")

    sf = sf_from_params(
        req.filter_kinmu, req.filter_shubetsu,
        req.filter_year, req.filter_month, req.filter_day,
    )
    sf_or_none = sf if not sf.is_empty() else None

    route = classify_query(req.question)

    if route is not None:
        context, docs = route.handler(req.question, sf_or_none)
        prompt = route.prompt
        route_name = route.name
    else:
        try:
            docs = search_docs(
                req.question, req.top_k, req.recency_weight, sf=sf_or_none,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"検索エラー: {e}")
        context = format_docs(docs[:LLM_CONTEXT_K])
        prompt = RAG_PROMPT
        route_name = "rag"

    _log_rag_context(route_name, req.question, docs, context)

    sources = [{"document": d.page_content, "metadata": d.metadata} for d in docs]
    filter_suffix = f"\n\n（絞り込み条件: {sf.label()}）" if sf_or_none else ""

    async def generator():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
        chain = prompt | state.llm | StrOutputParser()

        chunks: list[str] = []
        try:
            async for chunk in chain.astream({"context": context, "question": req.question}):
                if await request.is_disconnected():
                    return
                chunks.append(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.exception("LLM ストリームエラー")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            return

        full_text = "".join(chunks)
        if is_likely_chinese(full_text) or has_chinese_mix(full_text):
            log.warning("中国語混入を検出、フィルタ済みテキストに置換します")
            cleaned = filter_chinese_sentences(full_text).strip()
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
                except Exception as e:
                    log.exception("LLM リトライエラー")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
                    return

        if filter_suffix:
            yield f"data: {json.dumps({'type': 'chunk', 'text': filter_suffix}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
