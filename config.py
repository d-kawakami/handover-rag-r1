"""アプリ全体の設定定数および config.json の読み書き。"""
from __future__ import annotations

import json
from pathlib import Path

# 外部サービス
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 既定値（config.json で上書き可能）
DEFAULT_LLM_MODEL = "qwen2.5:14b"
DEFAULT_LLM_REASONING = False
DEFAULT_EQUIP_TENDENCY_SAMPLES = 20
DEFAULT_SUDACHI_MODE = "C"
DEFAULT_SUGGEST_SAMPLE_SIZE = 200

# 検索パラメータ
TOP_K_RETRIEVE = 40
TOP_K_FINAL = 5
LLM_CONTEXT_K = 50  # /api/query/stream で LLM に渡す文書数上限

# 設備傾向のサンプル数選択肢
VALID_EQUIP_SAMPLES = {10, 20, 30, 50, 100}

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
CONFIG_PATH = BASE_DIR / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
