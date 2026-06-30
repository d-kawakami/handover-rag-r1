# 引継ぎノート RAG — テスト実行・回帰ゲート
#
# テストは pandas を持つ pyenv の python3 で回す（venv はアプリ用で pandas 無し）。
# 別の python を使うなら:  make test-fast PY=/path/to/python
# 別コーパスなら:          make test-fast CORPUS=handover_sample.xlsx
#
# 評価の深さ（速い→遅い）:
#   test-fast   route/purity のみ（/api/search・LLM不要・数秒・高並列）   ← push ゲート
#   test-tier1  + count/doc_ids（/api/query=LLM生成・低並列・分単位）
#   test-full   + judge（Claude 採点込み・最も遅い・夜間向け）
#
# 各深さは専用 baseline と比較し、OK→NG の回帰があれば exit 1。
# 意図した変化は make baseline / baseline-tier1 / baseline-full で基準更新。

PY       ?= python3
CORPUS   ?= uploads/latest.xlsx
BASE_URL ?= http://localhost:8000
RUNNER   := tests/runner.py

.PHONY: help test-fast test-tier1 test-full baseline baseline-tier1 baseline-full install-hooks check-server

help:
	@echo "make test-fast        route/purity 高速ゲート（数秒）— 既定の回帰チェック"
	@echo "make test-tier1       + count/doc_ids（LLM生成・分単位）"
	@echo "make test-full        + judge（Claude採点・最遅・夜間向け）"
	@echo "make baseline         test-fast の基準を現状で更新"
	@echo "make baseline-tier1   test-tier1 の基準を更新"
	@echo "make baseline-full    test-full の基準を更新"
	@echo "make install-hooks    pre-push 回帰ゲートを .git/hooks に導入"

check-server:
	@curl -fs $(BASE_URL)/api/health >/dev/null 2>&1 || { \
	  echo "✗ RAG server ($(BASE_URL)) が起動していません。 ./start.sh で起動してください。"; exit 1; }

# ---- 実行（baseline と比較・回帰で exit 1）--------------------------------
test-fast: check-server
	$(PY) $(RUNNER) $(CORPUS) --no-llm --json last_run.json --baseline baseline.json

test-tier1: check-server
	$(PY) $(RUNNER) $(CORPUS) --tier1-only --json last_run_tier1.json --baseline baseline_tier1.json

test-full: check-server
	$(PY) $(RUNNER) $(CORPUS) --json last_run_full.json --baseline baseline_full.json

# ---- 基準更新（意図した変化を取り込む）-----------------------------------
baseline: check-server
	$(PY) $(RUNNER) $(CORPUS) --no-llm --update-baseline --baseline baseline.json

baseline-tier1: check-server
	$(PY) $(RUNNER) $(CORPUS) --tier1-only --update-baseline --baseline baseline_tier1.json

baseline-full: check-server
	$(PY) $(RUNNER) $(CORPUS) --update-baseline --baseline baseline_full.json

# ---- pre-push フック導入 --------------------------------------------------
install-hooks:
	@cp tests/hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push \
	  && echo "✓ pre-push hook installed (.git/hooks/pre-push)"
