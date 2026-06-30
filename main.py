"""引継ぎノート RAG - FastAPI バックエンド

LangChain + Ollama + ChromaDB
ハイブリッド検索 (BM25 + ベクトル + RRF) + CrossEncoder Re-ranking

このファイルはアプリ起動・ルーター登録のみを担う。
- 設定値 → config.py
- 実行時状態 → state.py
- クエリ分類・集計・検索 → search/
- HTTP エンドポイント → api/
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sentence_transformers import CrossEncoder

from api import admin, dict as dict_router, ingest as ingest_router, query, search
from config import (
    DEFAULT_EQUIP_TENDENCY_SAMPLES,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_REASONING,
    DEFAULT_SUDACHI_MODE,
    EMBED_MODEL,
    LLM_NUM_PREDICT,
    OLLAMA_BASE_URL,
    RERANKER_MODEL,
    STATIC_DIR,
    load_config,
)
from dict_builder import USER_DIC_PATH
from dict_db import init_db
from search.retrieval import build_index
from state import state
from tokenizer import init_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log.info("起動中: モデル初期化...")
        config = load_config()
        state.llm_model = config.get("llm_model", DEFAULT_LLM_MODEL)
        state.llm_reasoning = bool(config.get("llm_reasoning", DEFAULT_LLM_REASONING))
        state.equip_tendency_samples = int(
            config.get("equip_tendency_samples", DEFAULT_EQUIP_TENDENCY_SAMPLES)
        )
        log.info("使用モデル: %s (reasoning=%s)", state.llm_model, state.llm_reasoning)

        init_db()
        sudachi_mode = config.get("sudachi_mode", DEFAULT_SUDACHI_MODE)
        user_dic = str(USER_DIC_PATH) if USER_DIC_PATH.exists() else None
        init_tokenizer(mode=sudachi_mode, user_dic_path=user_dic)
        log.info("トークナイザ初期化完了 (mode=%s, user_dic=%s)", sudachi_mode, user_dic)

        state.embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        state.llm = OllamaLLM(
            model=state.llm_model,
            base_url=OLLAMA_BASE_URL,
            # 注: `False or None` は None になり、qwen3 系の think モード既定値（ON）が
            # そのまま使われてしまう。False を明示的に渡して think=false を強制する。
            reasoning=state.llm_reasoning,
            num_predict=LLM_NUM_PREDICT,
        )

        log.info("Re-ranker ロード中: %s", RERANKER_MODEL)
        state.reranker = CrossEncoder(RERANKER_MODEL)

        log.info("インデックス構築中...")
        build_index()
        log.info(
            "起動完了 — bm25=%d reranker=%s",
            len(state.all_docs), state.reranker is not None,
        )
    except Exception:
        log.exception("lifespan 初期化エラー")
        raise
    yield


app = FastAPI(title="引継ぎノート RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_api_responses(request, call_next):
    """/api/* レスポンスをブラウザがキャッシュしないようにする。

    Safari は Cache-Control 未指定の GET レスポンスを積極的にキャッシュする
    ため、思考モード ON/OFF や モデル切替などの状態取得 API でステートが
    UI に古いまま残る現象が起きる。明示的に no-store を返してこれを防ぐ。
    """
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


app.include_router(query.router)
app.include_router(search.router)
app.include_router(ingest_router.router)
app.include_router(admin.router)
app.include_router(dict_router.router)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
