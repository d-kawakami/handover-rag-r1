"""アプリの実行時状態を保持するシングルトン。

LLM・インデックス・ロック・インジェスト進捗を 1 箇所に集約することで、
各モジュールが個別のグローバル変数を持たずに済む。
"""
from __future__ import annotations

import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    from sentence_transformers import CrossEncoder

from config import (
    DEFAULT_EQUIP_TENDENCY_SAMPLES,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_REASONING,
)


class AppState:
    def __init__(self) -> None:
        # インデックス
        self.all_docs: list["Document"] = []
        self.bm25_retriever: "BM25Retriever | None" = None
        self.vector_store: "Chroma | None" = None
        self.reranker: "CrossEncoder | None" = None

        # LLM
        self.embeddings: "OllamaEmbeddings | None" = None
        self.llm: "OllamaLLM | None" = None
        self.llm_model: str = DEFAULT_LLM_MODEL
        self.llm_reasoning: bool = DEFAULT_LLM_REASONING
        self.equip_tendency_samples: int = DEFAULT_EQUIP_TENDENCY_SAMPLES

        # 同時実行制御
        self.model_lock = threading.Lock()
        self.ingest_lock = threading.Lock()
        self.ingest_state: dict[str, Any] = self._initial_ingest_state()

    @staticmethod
    def _initial_ingest_state() -> dict[str, Any]:
        return {
            "running": False,
            "progress": 0,
            "total": 0,
            "message": "待機中",
            "done": False,
            "error": None,
            "result": None,
        }

    def update_ingest(self, **kwargs: Any) -> None:
        with self.ingest_lock:
            self.ingest_state.update(kwargs)

    def reset_ingest(self) -> None:
        with self.ingest_lock:
            self.ingest_state = self._initial_ingest_state()
            self.ingest_state["running"] = True


state = AppState()
