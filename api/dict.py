"""ユーザー辞書管理エンドポイント。"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import random
import re

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas import (
    DictEntryCreate,
    DictEntryUpdate,
    SudachiModeRequest,
    SuggestRegisterRequest,
)
from config import DEFAULT_SUGGEST_SAMPLE_SIZE, load_config, save_config
from dict_builder import USER_DIC_PATH, rebuild_and_reload
from dict_db import (
    add_entry,
    delete_entry,
    export_to_csv,
    find_by_surface,
    get_all,
    get_all_surfaces,
    import_from_csv,
    update_entry,
)
from prompts import format_suggest_prompt
from search.retrieval import build_index
from state import state
from tokenizer import get_tokenizer, init_tokenizer

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/dict")
def dict_list(
    search: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    return get_all(search=search, page=page, page_size=page_size)


@router.post("/api/dict", status_code=201)
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


@router.get("/api/dict/check")
def dict_check(surface: str = Query(...)):
    entries = find_by_surface(surface.strip())
    return {"exists": len(entries) > 0, "entries": entries}


@router.put("/api/dict/{entry_id}")
def dict_update(entry_id: int, req: DictEntryUpdate):
    updates = req.model_dump(exclude_none=True)
    if not update_entry(entry_id, **updates):
        raise HTTPException(status_code=404, detail="エントリが見つかりません")
    return {"status": "ok"}


@router.delete("/api/dict/{entry_id}")
def dict_delete(entry_id: int):
    if not delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="エントリが見つかりません")
    return {"status": "ok"}


@router.post("/api/dict/import")
async def dict_import(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    count = import_from_csv(rows)
    return {"imported": count}


@router.get("/api/dict/export")
def dict_export():
    csv_text = export_to_csv()
    return StreamingResponse(
        io.StringIO(csv_text),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=user_dict.csv"},
    )


@router.post("/api/dict/rebuild")
def dict_rebuild():
    result = rebuild_and_reload()
    if result["status"] == "ok":
        try:
            build_index()
        except Exception as e:
            log.error("dict_rebuild: インデックス再構築失敗: %s", e)
            result["message"] += f"（BM25再構築エラー: {e}）"
    return result


@router.post("/api/dict/test")
def dict_test(req: dict):
    text = (req.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text を指定してください")
    tok = get_tokenizer()
    tokens = tok.tokenize(text)
    return {"tokens": tokens, "mode": tok.mode}


@router.get("/api/dict/mode")
def dict_get_mode():
    tok = get_tokenizer()
    return {"mode": tok.mode}


@router.post("/api/dict/mode")
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


@router.post("/api/dict/suggest")
async def dict_suggest():
    if state.llm is None:
        raise HTTPException(status_code=503, detail="LLM が初期化されていません")

    config = load_config()
    sample_size: int = config.get("suggest_sample_size", DEFAULT_SUGGEST_SAMPLE_SIZE)

    if not state.all_docs:
        raise HTTPException(status_code=503, detail="ドキュメントが読み込まれていません")

    sample = random.sample(state.all_docs, min(sample_size, len(state.all_docs)))
    sampled_text = "\n---\n".join(d.page_content[:300] for d in sample)
    prompt = format_suggest_prompt(sampled_text)

    try:
        raw = await asyncio.to_thread(state.llm.invoke, prompt)
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


@router.post("/api/dict/suggest/register")
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
