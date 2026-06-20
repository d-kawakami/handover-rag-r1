"""管理系エンドポイント: health / DB / モデル切替 / reasoning / equip-samples"""
from __future__ import annotations

import logging
import subprocess

import chromadb
from fastapi import APIRouter, HTTPException
from langchain_ollama import OllamaLLM

from api.schemas import EquipSamplesRequest, ModelChangeRequest, ReasoningRequest
from config import OLLAMA_BASE_URL, VALID_EQUIP_SAMPLES, load_config, save_config
from ingest import CHROMA_PATH, COLLECTION_NAME
from search.retrieval import build_index
from state import state

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/health")
def health():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        existing_names = [c.name for c in client.list_collections()]
        records = client.get_collection(COLLECTION_NAME).count() if COLLECTION_NAME in existing_names else 0
        return {
            "status": "ok",
            "records": records,
            "bm25_indexed": len(state.all_docs),
            "reranker_loaded": state.reranker is not None,
            "llm_model": state.llm_model,
            "llm_reasoning": state.llm_reasoning,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/api/db-status")
def db_status():
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        existing_names = [c.name for c in client.list_collections()]
        records = client.get_collection(COLLECTION_NAME).count() if COLLECTION_NAME in existing_names else 0
        return {"records": records, "bm25_indexed": len(state.all_docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/reload-index")
def reload_index():
    try:
        build_index()
        return {"status": "ok", "records": len(state.all_docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ollama-models")
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
        return {"models": models, "current": state.llm_model}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ollama コマンドが見つかりません")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="ollama list がタイムアウトしました")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/model")
def get_model():
    return {"model": state.llm_model, "reasoning": state.llm_reasoning}


@router.post("/api/model")
def change_model(req: ModelChangeRequest):
    if not req.model.strip():
        raise HTTPException(status_code=400, detail="モデル名を指定してください")
    with state.model_lock:
        try:
            new_llm = OllamaLLM(
                model=req.model,
                base_url=OLLAMA_BASE_URL,
                # False を明示的に渡すことで qwen3 系の think モード既定値（ON）を上書きする
                reasoning=state.llm_reasoning,
            )
            state.llm = new_llm
            state.llm_model = req.model
            config = load_config()
            config["llm_model"] = req.model
            save_config(config)
            log.info("モデルを変更しました: %s", req.model)
            return {"status": "ok", "model": req.model}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"モデル変更エラー: {e}")


@router.get("/api/reasoning")
def get_reasoning():
    return {"enabled": state.llm_reasoning}


@router.post("/api/reasoning")
def set_reasoning(req: ReasoningRequest):
    with state.model_lock:
        try:
            new_llm = OllamaLLM(
                model=state.llm_model,
                base_url=OLLAMA_BASE_URL,
                # OFF を明示的に Ollama へ伝えるため False をそのまま渡す
                reasoning=req.enabled,
            )
            state.llm = new_llm
            state.llm_reasoning = req.enabled
            config = load_config()
            config["llm_reasoning"] = req.enabled
            save_config(config)
            log.info("reasoning を変更しました: %s", req.enabled)
            return {"status": "ok", "enabled": req.enabled}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"reasoning 変更エラー: {e}")


@router.get("/api/equip-samples")
def get_equip_samples():
    return {"samples": state.equip_tendency_samples}


@router.post("/api/equip-samples")
def set_equip_samples(req: EquipSamplesRequest):
    if req.samples not in VALID_EQUIP_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"サンプル数は {sorted(VALID_EQUIP_SAMPLES)} のいずれかを指定してください",
        )
    state.equip_tendency_samples = req.samples
    config = load_config()
    config["equip_tendency_samples"] = req.samples
    save_config(config)
    log.info("設備傾向サンプル数を変更しました: %d", req.samples)
    return {"status": "ok", "samples": req.samples}
