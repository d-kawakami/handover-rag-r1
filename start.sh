#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "=== venv を作成中 ==="
  python3 -m venv venv
fi

source venv/bin/activate

echo "=== パッケージ確認・インストール ==="
pip install -q -r requirements.txt

# ChromaDB にデータがなければ取り込みを実行
CHROMA_DB="chroma_db/chroma.sqlite3"
if [ ! -f "$CHROMA_DB" ]; then
  echo "=== ChromaDB が空のため CSV を取り込み中 (初回のみ・時間がかかります) ==="
  python ingest.py
else
  echo "=== ChromaDB 既存 — ingest をスキップ ==="
fi

echo "=== サーバー起動: http://localhost:8000 ==="
python -m uvicorn main:app --host 0.0.0.0 --port 8000
