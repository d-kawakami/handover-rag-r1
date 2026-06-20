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
            reasoning=state.llm_reasoning or None,
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

app.include_router(query.router)
app.include_router(search.router)
app.include_router(ingest_router.router)
app.include_router(admin.router)
app.include_router(dict_router.router)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
