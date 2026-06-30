"""質問エンドポイント: /api/query, /api/query/stream"""
from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import OllamaLLM

from api._routing import classify_query
from api.schemas import QueryRequest, QueryResponse
from config import LLM_CONTEXT_K, LLM_NUM_PREDICT, OLLAMA_BASE_URL
from language import has_chinese_mix, filter_chinese_sentences, is_likely_chinese, sanitize_if_chinese
from prompts import RAG_PROMPT
from search.filters import sf_from_params
from search.retrieval import format_docs, search_docs
from state import state

log = logging.getLogger(__name__)

router = APIRouter()

# 特定の (システム指示 + context + 質問) の組み合わせで、qwen2.5 系は最初のトークンが
# ほぼ確率 1 で EOS になり、空応答（"No data received from Ollama stream" /
# "No generation chunks were returned"）になることがある。温度・top_p・言い換え等の
# サンプリング摂動では脱出できないため、二段構えで救済する:
#   1) temperature を上げた LLM で1回だけ再試行（病的でない空応答はこれで回復する）
#   2) それでも空なら、集計系ルートでは context の集計見出しから決定的に回答を組み立てる
#      （件数等の答えは既に context に算出済みのため、LLM 無しで正確に返せる）
_RETRY_TEMPERATURE = 0.7

# context 先頭の集計見出し: 例「■ 「鶴見ポンプ場」（同義語含む）の故障記録件数: 51件」
_HEADLINE_RE = re.compile(r'^■\s*(.+?件数)\s*[:：]\s*(\d+)\s*件')
_BREAKDOWN_PREFIXES = ("（種別内訳", "(種別内訳", "（内訳", "(内訳")


def _build_retry_llm() -> OllamaLLM:
    """空応答リカバリ用に temperature を上げた一時 LLM を作る（メイン設定は変えない）。"""
    return OllamaLLM(
        model=state.llm_model,
        base_url=OLLAMA_BASE_URL,
        reasoning=state.llm_reasoning,
        num_predict=LLM_NUM_PREDICT,
        temperature=_RETRY_TEMPERATURE,
    )


def _context_fallback_answer(context: str) -> str:
    """集計系 context の見出し行から、LLM を介さず決定的に回答文を組み立てる。

    集計ルート（count 等）の context は「■ …件数: N件」「（種別内訳: …）」という
    確定済みサマリを含む。LLM が空応答を返したときの最終フォールバックに使う。
    該当見出しが無ければ空文字を返す（＝フォールバック不可）。
    """
    headline = breakdown = ""
    for line in context.splitlines():
        s = line.strip()
        m = _HEADLINE_RE.match(s)
        if m and not headline:
            headline = f"{m.group(1)}は{m.group(2)}件です。"
        elif not breakdown and s.startswith(_BREAKDOWN_PREFIXES):
            breakdown = s
    if not headline:
        return ""
    return "集計データによると、" + headline + breakdown


def _invoke_with_retry(prompt, context: str, question: str) -> str:
    """LLM を invoke し、空応答なら摂動再試行→集計データ由来の決定的回答で救済する。"""
    for llm in (state.llm, _build_retry_llm()):
        try:
            out = (prompt | llm | StrOutputParser()).invoke(
                {"context": context, "question": question}
            )
            if out and out.strip():
                return out
        except Exception as e:
            log.warning("LLM invoke 失敗(%s)、再試行します: q=%r", e, question)
    fb = _context_fallback_answer(context)
    if fb:
        log.warning("LLM 空応答のため集計データから回答を生成しました: q=%r", question)
        return fb
    raise RuntimeError("LLM が回答を生成できませんでした（空応答）")


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
            answer = sanitize_if_chinese(_invoke_with_retry(route.prompt, context, req.question))
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
            answer = sanitize_if_chinese(_invoke_with_retry(RAG_PROMPT, context, req.question))
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

        # 1回目はメイン LLM、空応答なら摂動 LLM で1回だけ再試行。それでも空なら下流で
        # 集計データ由来の決定的フォールバックに回す（qwen2.5 系の即 EOS 空応答の救済）。
        chunks: list[str] = []
        disconnected = False
        for attempt, llm in enumerate((state.llm, None)):
            if attempt == 1:
                # 1回目が空応答だった場合のみ到達。UI の表示をクリアして摂動再試行。
                log.warning("Ollama 空応答、摂動して再試行します: q=%r", req.question)
                yield f"data: {json.dumps({'type': 'reset'}, ensure_ascii=False)}\n\n"
                llm = _build_retry_llm()
            chunks = []
            chain = prompt | llm | StrOutputParser()
            try:
                async for chunk in chain.astream({"context": context, "question": req.question}):
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk}, ensure_ascii=False)}\n\n"
            except Exception as e:
                log.warning("LLM ストリーム失敗(attempt=%d): %s", attempt, e)
            if disconnected or chunks:
                break
        if disconnected:
            return
        if not chunks:
            # 摂動再試行でも空 → 集計系 context なら見出しから決定的に回答を組み立てる
            fb = _context_fallback_answer(context)
            if fb:
                log.warning("再試行後も空応答、集計データから回答を生成: q=%r", req.question)
                yield f"data: {json.dumps({'type': 'reset'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': fb}, ensure_ascii=False)}\n\n"
                if filter_suffix:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': filter_suffix}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                return
            log.error("再試行後も空応答（フォールバック不可）: q=%r", req.question)
            msg = "回答を生成できませんでした。お手数ですが質問の言い回しを少し変えて再度お試しください。"
            yield f"data: {json.dumps({'type': 'error', 'message': msg}, ensure_ascii=False)}\n\n"
            return

        chain = prompt | state.llm | StrOutputParser()
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
