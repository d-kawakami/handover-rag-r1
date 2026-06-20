"""インジェスト関連エンドポイント。"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config import UPLOAD_DIR
from ingest import run_ingest
from search.retrieval import build_index
from state import state

log = logging.getLogger(__name__)

router = APIRouter()


def _run_ingest_thread(csv_path: Path, rebuild: bool) -> None:
    def progress_cb(done: int, total: int, msg: str) -> None:
        state.update_ingest(progress=done, total=total, message=msg)
        log.info("[ingest] %s", msg)

    try:
        result = run_ingest(csv_path, progress_cb, rebuild=rebuild)
        state.update_ingest(
            running=False,
            done=True,
            result=result,
            message=f"完了: {result['added']} 件追加 / {result['skipped']} 件スキップ / DB合計 {result['total']} 件",
        )
        log.info("[ingest] 完了: %s", result)
        try:
            build_index()
            log.info("[ingest] インデックス再構築完了: %d 件", len(state.all_docs))
        except Exception as e:
            log.error("[ingest] インデックス再構築失敗: %s", e)
    except Exception as e:
        log.exception("[ingest] エラー")
        state.update_ingest(
            running=False,
            done=True,
            error=str(e),
            message=f"エラー: {e}",
        )


@router.post("/api/ingest")
async def start_ingest(file: UploadFile = File(...), rebuild: bool = False):
    with state.ingest_lock:
        if state.ingest_state["running"]:
            raise HTTPException(status_code=409, detail="インジェスト実行中です")

    fname = file.filename or ""
    if not (fname.endswith(".csv") or fname.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="CSVまたはXLSXファイルを選択してください")

    suffix = ".xlsx" if fname.endswith(".xlsx") else ".csv"
    csv_path = UPLOAD_DIR / f"latest{suffix}"
    csv_path.write_bytes(await file.read())

    state.update_ingest(
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


@router.get("/api/ingest/stream")
async def ingest_stream():
    async def generator():
        while True:
            with state.ingest_lock:
                snapshot = dict(state.ingest_state)
            yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
            if snapshot.get("done") or not snapshot.get("running"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/ingest/status")
def ingest_status():
    with state.ingest_lock:
        return dict(state.ingest_state)
