# 引継ぎノート RAG 検索システム

**日本語** | [English](README.en.md)

---

## 概要

プラントの引継ぎノート XLSX/CSV データを対象にした、ローカル完結型の RAG（Retrieval-Augmented Generation）検索システムです。  
[NotebookLM](https://notebooklm.google.com/) のように自然言語で質問すると、関連する引継ぎ記録を根拠にして AI が日本語で回答します。  
すべての処理はローカルで完結し、データが外部に送信されることはありません。

## 使用技術

| コンポーネント | 詳細 |
|---|---|
| LLM | `qwen2.5:14b`（Ollama 経由、変更可） |
| 埋め込みモデル | `nomic-embed-text`（Ollama 経由） |
| Re-ranker | `BAAI/bge-reranker-v2-m3`（HuggingFace、日本語対応多言語モデル） |
| RAG フレームワーク | LangChain（`langchain`, `langchain-ollama`, `langchain-chroma`, `langchain-community`） |
| ベクトルDB | ChromaDB（ローカル永続化） |
| 形態素解析（BM25用） | SudachiPy + sudachidict_full + neologdn |
| バックエンド | FastAPI + Uvicorn |
| フロントエンド | 素の HTML/CSS/JavaScript（依存なし） |

## クエリルーティングとシーケンス

質問の種類を自動判定し、最も適切な処理ルートで回答します。

```
ユーザーの質問
      │
      ▼
┌─────────────────────────────────────────────────┐
│ 1. 特定件数クエリ判定                           │
│    is_specific_failure_count_query()            │
│    例: 「VVVFの故障は何回？」                   │
│        「TNTP計委託の実施回数は？」           │
└────────────┬────────────────────────────────────┘
             │ Yes
             ▼
      count_specific_failure()
      ┌──────────────────────────────────────────┐
      │ 1. 対象語を extract_count_subject() で抽出│
      │ 2. 同義語グループで検索語を展開           │
      │    例: VVVF → インバータ, インバーター    │
      │ 3. 全件スキャン + テキストマッチ         │
      │ 4. 種別内訳・故障コード別内訳を集計      │
      │ 5. SPECIFIC_COUNT_PROMPT で LLM 回答生成 │
      └──────────────────────────────────────────┘
             │ No
             ▼
┌─────────────────────────────────────────────────┐
│ 2. 集計・ランキングクエリ判定                   │
│    is_aggregation_query()                       │
│    例: 「故障の多い機器ランキングを教えて」     │
│        「最も多く発生した故障は？」             │
└────────────┬────────────────────────────────────┘
             │ Yes
             ▼
      aggregate_for_query()
      ┌──────────────────────────────────────────┐
      │ 1. parse_date_filter() で期間抽出        │
      │ 2. 期間内の全件スキャン                  │
      │ 3. 種別・故障コード別カウント            │
      │ 4. ランキング形式でコンテキスト生成      │
      │ 5. AGGREGATION_PROMPT で LLM 回答生成   │
      └──────────────────────────────────────────┘
             │ No
             ▼
┌─────────────────────────────────────────────────┐
│ 3. 季節傾向クエリ判定                           │
│    is_seasonal_tendency_query()                 │
│    例: 「夏に発生しやすい故障の傾向は？」       │
│        「冬場のトラブルの特徴は？」             │
│    ※ 年指定（2024年夏など）があれば RAG へ      │
└────────────┬────────────────────────────────────┘
             │ Yes
             ▼
      aggregate_by_season()
      ┌──────────────────────────────────────────┐
      │ 1. _parse_target_season() で季節を特定   │
      │ 2. 全件の月別件数を集計                  │
      │ 3. 対象季節の月セットで _doc_in_month_set│
      │    によりフィルタ（年をまたいで集計）     │
      │ 4. 頻発故障コード・代表記録を抽出        │
      │ 5. SEASONAL_TENDENCY_PROMPT で LLM 回答 │
      └──────────────────────────────────────────┘
             │ No
             ▼
┌─────────────────────────────────────────────────┐
│ 4. 通常 RAG（ハイブリッド検索）                 │
└────────────┬────────────────────────────────────┘
             ▼
      search_docs()
      ┌──────────────────────────────────────────┐
      │ 1. parse_date_filter() で期間抽出        │
      │ 2. BM25 検索（SudachiPy トークナイズ）   │
      │    日本語変換クエリで精度向上            │
      ├──────────────────────────────────────────┤
      │ 3. ベクトル検索（nomic-embed-text）      │
      ├──────────────────────────────────────────┤
      │ 4. RRF 融合（BM25×2票 + Vector×1票）    │
      │    上位 TOP_K_RETRIEVE 件（デフォルト40）│
      ├──────────────────────────────────────────┤
      │ 5. CrossEncoder Re-ranking               │
      │    BAAI/bge-reranker-v2-m3 でスコアリング│
      │    設備名ボースト +1.5 適用              │
      ├──────────────────────────────────────────┤
      │ 6. 上位 top_k 件を返却                  │
      └──────────────────────────────────────────┘
             ▼
      RAG_PROMPT + LLM で日本語回答生成
```

## 検索精度向上の仕組み

| 手法 | 効果 |
|---|---|
| **クエリルーティング（4階層）** | 件数・集計・季節傾向クエリを自動判定し、全件スキャンで正確に回答 |
| **同義語展開** | VVVF ≡ インバータ ≡ インバーター などの同義語を自動認識して検索漏れを防ぐ |
| **BM25（キーワード検索）** | 設備名・故障番号など固有語に強い |
| **ベクトル検索** | 意味的な類似性で広く拾う |
| **RRF 融合** | BM25 と ベクトル検索を統合してランク付け（BM25 優先） |
| **CrossEncoder Re-ranking** | 質問と各レコードの関連度を精密評価。設備名ボーストで固有語を優先 |
| **SudachiPy 形態素解析** | 複合語（冷却水ポンプ等）を 1 トークンに保持し BM25 精度向上 |

## ディレクトリ構成

```
handover-rag-r1/
├── main.py             # FastAPI バックエンド（RAG パイプライン + 辞書 API）
├── ingest.py           # CSV/XLSX → ChromaDB 取り込みスクリプト（差分対応）
├── tokenizer.py        # SudachiPy トークナイザ（BM25 前処理）
├── dict_db.py          # ユーザー辞書 SQLite CRUD
├── dict_builder.py     # user.dic ビルド + トークナイザリロード
├── create_sample_xlsx.py # サンプル XLSX 生成スクリプト
├── handover_sample.xlsx  # サンプルデータ（動作確認用）
├── user_dict.db        # ユーザー辞書エントリ（自動生成）
├── user.dic            # SudachiPy ユーザー辞書バイナリ（辞書反映後に生成）
├── config.json         # LLM モデル名・SudachiPy 分割モード設定
├── static/
│   ├── index.html      # 検索 Web UI
│   ├── dict_admin.html # ユーザー辞書管理 UI
│   ├── dict_admin.js   # 辞書管理フロントエンド
│   └── dict_admin.css  # 辞書管理スタイル
├── chroma_db/          # ベクトルDB（自動生成・永続化）
├── uploads/            # Web UI 経由でアップロードされた CSV の一時保存先
├── requirements.txt    # Python 依存パッケージ
├── start.sh            # 起動スクリプト
└── README.md
```

## 検証環境

| 項目 | 詳細 |
|---|---|
| マシン | Mac mini M4 Pro |
| メモリ | 48GB |

## 前提条件

- macOS / Linux
- Python 3.11 以上
- [Ollama](https://ollama.com/) がインストールされ、以下のモデルが pull 済みであること

```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

> **Apple Silicon (arm64) での注意**  
> `sudachipy` と `neologdn` は arm64 用 wheel が配布されており、`pip install` で問題なくインストールできます。  
> `sudachidict_full`（約 120 MB）は初回インストール時にビルドが走るため、数分かかる場合があります。

> **モデルサイズの選択について**
> お使いの PC の VRAM（GPU）または RAM（Mac はユニファイドメモリ）に応じて選択してください。
>
> | モデル | 目安メモリ | 備考 |
> |---|---|---|
> | `qwen2.5:32b` | 24GB 以上推奨 | 最高精度 |
> | `qwen2.5:14b` | 12GB 以上推奨 | デフォルト（精度・速度バランス） |
> | `qwen2.5:7b` | 8GB 以上推奨 | 軽量・高速 |
>
> モデルを変更した場合は、Web UI ヘッダーのモデル選択プルダウン、または `/api/model` API を使用してください。変更は `config.json` に自動保存されます。

## セットアップと起動

### 1. リポジトリの取得

```bash
git clone https://github.com/d-kawakami/handover-rag-r1.git ~/handover-rag-r1
```

> GitHub からダウンロードした場合は ZIP を解凍して `~/handover-rag-r1/` に配置してください。

> **config.json について**  
> `config.json` はアプリが自動生成するため git 管理外です。テンプレートは `config.json.example` を参照してください。  
> ない場合は LLM モデル `qwen2.5:14b`・reasoning 無効のデフォルトで起動します。

### 2. CSV/XLSX ファイルを配置

引継ぎノートのファイル（CSV または XLSX）をプロジェクトの**カレントディレクトリ**（`~/handover-rag-r1/`）に置きます。このリポジトリにはサンプルとしてhandover_sample.xlsxが含まれています。

ファイル名が `引き継ぎノート*.csv` または `引き継ぎノート*.xlsx` の形式であれば自動検出されます。

```
~/handover-rag-r1/
├── 引き継ぎノート2026.xlsx   ← ここに置く（引き継ぎノート*.csv または *.xlsx）
├── ingest.py
└── ...
```

ファイルが複数ある場合は、最終更新日時が最新のものが自動的に選択されます。

#### ファイルが見つからない場合

`引き継ぎノート*` にマッチするファイルがない場合、以下の順序でファイルを選択します。

1. **カーソル選択**: カレントディレクトリにある `.csv` / `.xlsx` を一覧表示
   - `↑` / `↓`（または `k` / `j`）で移動、`Enter` で決定、`q` または `Esc` でキャンセル
2. **手動入力**: カレントディレクトリに候補がない場合、またはキャンセルした場合はファイルパスを直接入力

#### `--csv` オプションで直接指定

任意のファイルを直接指定することもできます。

```bash
python ingest.py --csv /path/to/your_data.xlsx
```

### 3. Ollama を起動

```bash
ollama serve
```

### 4. アプリを起動

```bash
cd ~/handover-rag-r1
bash start.sh
```

初回起動時は以下が自動で実行されます。

1. Python 仮想環境（`venv`）の作成
2. 依存パッケージのインストール（`requirements.txt`）
3. Re-ranker モデル（`BAAI/bge-reranker-v2-m3`）の自動ダウンロード（初回のみ・数百 MB）
4. CSV/XLSX の ChromaDB への取り込み（レコード数に応じて数分かかります）  
   ※ `引き継ぎノート*` にマッチするファイルがない場合はターミナルにファイル選択画面が表示されます
5. サーバー起動

2回目以降はインデックスが再利用されるため、すぐに起動します。

### 5. ブラウザでアクセス

```
http://localhost:8000
```

## 使い方

### AI 回答モード

質問を入力して **「AI 回答」** ボタンを押すと、関連する引継ぎ記録を根拠にして設定中の LLM モデルが日本語で回答します。

**質問例:**

| 質問の種類 | 例 |
|---|---|
| 特定設備の件数 | `VVVFの故障は何回発生しましたか？` |
| 作業の実施回数 | `TNTP計委託の実施回数は？` |
| 故障ランキング | `最も故障の多い機器は何ですか？` |
| 季節傾向 | `夏に発生しやすい故障の傾向は？` |
| 期間指定 | `2024年6月の夜勤での異常報告を教えてください` |
| 一般的な検索 | `高圧洗浄ポンプのトラブル事例は？` |

**ショートカット:** `Ctrl + Enter`（Mac: `Cmd + Enter`）で送信

### 検索のみモード

**「検索のみ」** ボタンを押すと、LLM を使わずに類似レコードを高速で一覧表示します。記録を直接確認したいときに便利です。

### 件数の設定

プルダウンまたは設定画面から件数を変更できます。

| 設定項目 | 変更方法 | 上限 | 説明 |
|---|---|---|---|
| **チャット件数**（参照件数） | トップ画面プルダウン または 設定画面 | 50件 | AI 回答時に LLM へ渡すレコード数 |
| **検索件数** | 設定画面のみ | 100件 | ハイブリッド検索で取得する候補数。増やすと精度向上・速度低下 |

> **件数の意味と効果**
> - **チャット件数を増やす**: LLM が参照できる記録が増えるため、関連情報の見落としが減ります。ただし LLM のコンテキストウィンドウ（32,768トークン）の上限があるため 50件 を上限としています
> - **検索件数を増やす**: BM25・ベクトル検索の候補を増やし、CrossEncoder Re-ranking の対象を広げます。精度が向上しますが検索時間（Re-ranking）が長くなります

## API エンドポイント

### 検索・RAG

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/api/query` | AI 回答 + 参照ソース返却 |
| `POST` | `/api/query/stream` | AI 回答をストリーミングで返す（SSE） |
| `GET` | `/api/search?q=クエリ&top_k=5` | 検索のみ（LLM なし） |
| `GET` | `/api/health` | サーバー状態確認 |
| `GET` | `/api/db-status` | DB 登録件数・BM25 インデックス件数 |
| `POST` | `/api/ingest` | CSV アップロード＆インジェスト開始（multipart） |
| `GET` | `/api/ingest/stream` | インジェスト進捗を SSE で取得 |
| `GET` | `/api/ingest/status` | インジェスト進捗をポーリングで取得 |
| `POST` | `/api/reload-index` | ingest 後にインデックスを再構築 |
| `GET` | `/api/ollama-models` | Ollama のインストール済みモデル一覧 |
| `GET` | `/api/model` | 現在の LLM モデル取得 |
| `POST` | `/api/model` | LLM モデルを変更・config.json に保存 |
| `GET` | `/api/reasoning` | Reasoning モード（拡張思考）の有効/無効を取得 |
| `POST` | `/api/reasoning` | Reasoning モードを変更・config.json に保存 |

### ユーザー辞書管理

| メソッド | パス | 説明 |
|---|---|---|
| `GET` | `/api/dict` | 一覧取得（`search`, `page`, `page_size` クエリパラメータ対応） |
| `POST` | `/api/dict` | 新規エントリ追加 |
| `GET` | `/api/dict/check?surface=語` | 表記が既登録かチェック |
| `PUT` | `/api/dict/{id}` | エントリ更新 |
| `DELETE` | `/api/dict/{id}` | エントリ削除 |
| `POST` | `/api/dict/import` | CSV 一括インポート（multipart） |
| `GET` | `/api/dict/export` | CSV エクスポート（ダウンロード） |
| `POST` | `/api/dict/rebuild` | 辞書ビルド＋BM25 インデックス再構築 |
| `POST` | `/api/dict/test` | テキストの形態素解析結果を返す |
| `GET` | `/api/dict/mode` | 現在の SudachiPy 分割モード取得 |
| `POST` | `/api/dict/mode` | 分割モード変更（A / B / C） |
| `POST` | `/api/dict/suggest` | AI による辞書候補抽出（LLM 必須） |
| `POST` | `/api/dict/suggest/register` | 候補の一括登録 |

### リクエスト例

```bash
# AI 回答
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ポンプの故障事例を教えてください", "top_k": 5}'

# 検索のみ
curl "http://localhost:8000/api/search?q=ポンプ故障&top_k=5"

# 形態素解析テスト
curl -X POST http://localhost:8000/api/dict/test \
  -H "Content-Type: application/json" \
  -d '{"text": "冷却水ポンプの故障について"}'
```

## ユーザー辞書管理

BM25 検索の前処理に SudachiPy による形態素解析を使っています。  
プラント専門用語（`冷却水ポンプ`、`最終沈殿池` など）を 1 語として扱うために、ユーザー辞書に登録してください。

### 辞書管理画面を開く

ブラウザで **「辞書管理」** リンク（ヘッダー右上）をクリック、または直接アクセス:

```
http://localhost:8000/static/dict_admin.html
```

### 基本的な運用フロー

1. **用語を登録** — 「新規追加」ボタンまたは CSV インポートでエントリを追加
2. **辞書を反映** — 「辞書を反映」ボタンで user.dic をビルド → BM25 インデックスを再構築
3. **テストで確認** — 「形態素解析テスト」で登録した用語が 1 トークンになっているか確認

> **注意**: 登録しただけでは BM25 には反映されません。必ず「辞書を反映」を実行してください。

### CSV インポート形式

ヘッダー行あり。`surface` と `reading` は必須、他列は省略可能（デフォルト値が使われます）。

```csv
surface,reading,pos,cost,normalized,enabled
最終沈殿池,サイシュウチンデンチ,名詞,固有名詞,一般,3000,,1
冷却水ポンプ,レイキャクスイポンプ,名詞,固有名詞,一般,3000,,1
最沈,サイチン,名詞,固有名詞,一般,5000,最終沈殿池,1
```

| 列 | 説明 | デフォルト |
|---|---|---|
| `surface` | 表記（文書中の表現） | 必須 |
| `reading` | 読み（カタカナ） | 必須 |
| `pos` | 品詞（カンマ区切り最大3要素） | `名詞,固有名詞,一般` |
| `cost` | 優先度コスト（低いほど優先） | `5000` |
| `normalized` | 正規化表記（略語→正式名など） | 空（表記と同じ） |
| `enabled` | 有効(1) / 無効(0) | `1` |

### 分割モード

`config.json` の `sudachi_mode` キーで変更できます（A / B / C、デフォルトは C）。

| モード | 動作 |
|---|---|
| **C**（推奨） | 複合語を最大限 1 語に保つ（`冷却水ポンプ` → 1 トークン） |
| B | 中程度の分割 |
| A | 最小単位に細かく分割 |

辞書管理画面の「分割モード」セレクトボックスでも変更でき、config.json に自動保存されます。

### AI による辞書候補抽出

辞書管理画面の「AI 辞書候補抽出」エリアから、引き継ぎノートをサンプリングして LLM に専門用語の候補を抽出させることができます。

1. 「候補を自動抽出」ボタンをクリック（LLM の推論に数秒〜十数秒かかります）
2. 抽出された候補を確認し、読みを入力
3. 登録したい候補にチェックを入れて「選択を登録」

サンプリング件数は `config.json` の `suggest_sample_size`（デフォルト 200）で変更できます。

## CSV/XLSX の再取り込み

### Web UI から操作する（推奨）

サーバー起動後、ブラウザの **「DB管理」** パネルを開いて CSV/XLSX ファイルを選択し、ボタンで操作します。

| ボタン | 動作 |
|---|---|
| **差分更新** | 既存 DB を保持したまま新規行のみ追加 |
| **全件再構築** | 既存 DB を削除して全レコードを再インジェスト |

進捗はリアルタイムのプログレスバーで確認できます。

### コマンドラインから操作する

**全件再構築**（データを丸ごと差し替えたい場合）:

```bash
rm -rf ~/handover-rag-r1/chroma_db
bash start.sh
```

**差分追加のみ**（既存データの末尾に行が追加された場合）:

```bash
cd ~/handover-rag-r1
source venv/bin/activate
python ingest.py                        # 自動検出またはファイル選択
python ingest.py --csv path/to/file.xlsx  # ファイルを直接指定する場合
python ingest.py --rebuild              # 全件再構築する場合
curl -X POST http://localhost:8000/api/reload-index
```

## CSV フォーマット

CSVはタイトル行・ヘッダー行・データ行の構成です（データ行は0始まりのインデックスで参照）。

| 列インデックス | フィールド | 説明 |
|---|---|---|
| 0 | （管理用） | 連番など（取り込み時は読み飛ばし） |
| 1 | 故障 | 故障番号（任意） |
| 2 | 記入年月日 | `YYYY/M/DD` 形式 |
| 3 | 曜日 | 月〜日 |
| 4 | 勤務 | 日勤 / 夜勤 など |
| 5 | 種別 | 報告 / 故障 / 処置 など |
| 6 | 時刻 | 午前 / 午後 / HH:MM など |
| 7 | 内容 | 本文（検索・AI 回答の主対象） |
| 8 | 記入者 | 担当者名 |

## 主要パラメータ

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `TOP_K_RETRIEVE` | 40 | BM25・ベクトル検索の候補数（UI の「検索件数」で変更） |
| `TOP_K_FINAL` | 5 | Re-ranking 後の最終件数（UI の「チャット件数」で変更） |
| `LLM_CONTEXT_K` | 50 | LLM へ渡す最大件数の上限（コンテキストウィンドウ保護） |
| `EMBED_MODEL` | `nomic-embed-text` | ベクトル埋め込みモデル |
| `LLM_MODEL` | `qwen2.5:14b` | 回答生成 LLM（config.json または UI で変更可） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Re-ranker モデル |
| `_EQUIP_BOOST` | 1.5 | 設備名を含む文書への Re-ranker スコアブースト量 |

## ライセンス

[MIT License](LICENSE)
