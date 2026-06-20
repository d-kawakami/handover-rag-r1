"""クエリの種別判定（ルーティング）と対象主語抽出。"""
from __future__ import annotations

import re

from search.patterns import (
    AGGREGATION_KEYWORDS,
    EQUIP_TENDENCY_KWS,
    FAILURE_SEVERITY_LEVELS,
    GENERIC_SUBJECTS,
    SEASON_WORD_MAP,
    SEASONAL_TENDENCY_KWS,
)


def is_aggregation_query(query: str) -> bool:
    if any(kw in query for kw in AGGREGATION_KEYWORDS):
        return True
    if re.search(r'(最も|一番|最多).{0,10}(多い|多く|多かった|発生|故障)', query):
        return True
    if re.search(r'(多い|多く|多かった).{0,8}(機器|設備|故障|装置)', query):
        return True
    if re.search(r'(繰り返し|何度も|しばしば|たびたび).{0,10}(故障|発生|停止)', query):
        return True
    return False


def parse_min_count(query: str) -> int:
    m = re.search(r'(\d+)回以上', query)
    return int(m.group(1)) if m else 1


def parse_top_n(query: str) -> int:
    for pattern in [r'(\d+)位まで', r'上位(\d+)', r'(\d+)位']:
        m = re.search(pattern, query)
        if m:
            return int(m.group(1))
    return 10


def extract_count_subject(query: str) -> str:
    """クエリから「何件数えるか」の対象名（設備名・故障名）を抽出する。"""
    # 先頭の年月日パターンを除去（日付は parse_date_filter で別途処理）
    q = re.sub(r'^\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:から|以降|以前|まで|頃)?の?', '', query).strip()
    patterns = [
        # 「全ての〇〇が発生した回数」→ 〇〇（例: 全てのVVVF故障が発生した回数）
        r'全ての?(.+?)(?:故障|障害|不具合)?(?:が|は)?発生(?:した)?(?:回数|件数)',
        # 「〇〇が/は発生した回数/件数」→ 〇〇
        r'^(.+?)[がは]発生(?:した)?(?:回数|件数)',
        # 「〇〇の実施/点検/発生/委託等の回数」→ 〇〇
        r'^(.+?)の\S{0,4}回数',
        # 「〇〇に関する故障/処置件数・発生回数」→ 〇〇
        r'^(.+?)に関する(?:故障処置|故障|障害|不具合|発生)?(?:が|は|の)?(?:何回|何件|何度|件数|発生回数|回数)',
        # 「〇〇の故障処置/故障/障害[がは]何回/件数」→ 〇〇
        r'^(.+?)の(?:故障処置|故障|障害|不具合|発生)(?:が|は|の)?(?:何回|何件|何度|件数|発生回数|回数)',
        # 「〇〇の故障処置/故障」で終わる形（文末に件数語がある場合）
        r'^(.+?)の(?:故障処置|故障|障害|不具合)',
        # 「〇〇[がは]何回/何件」→ 〇〇
        r'^(.+?)[がは].*?(?:何回|何件|何度)',
        # 「〇〇何回/何件/件数」→ 〇〇
        r'^(.+?)(?:何回|何件|何度|件数)',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            s = m.group(1).strip()
            # 先頭の不要語を除去（「全ての」「について」等）
            s = re.sub(r'^全ての?', '', s).strip()
            s = re.sub(r'^(?:について|に関して|関連する)', '', s).strip()
            # 末尾の助詞・「に関する〇〇」を除去
            s = re.sub(r'に関する.*$', '', s).strip()
            s = re.sub(r'[のがはにをで]$', '', s).strip()
            if len(s) >= 2 and s not in GENERIC_SUBJECTS:
                return s
    return ""


def is_specific_failure_count_query(query: str) -> bool:
    """特定の設備・故障名の発生件数を問うクエリか判定する。"""
    count_kws = [
        "何回", "何件", "何度", "件数", "発生回数", "回発生",
        "た回数", "の回数", "した回数", "故障回数", "障害回数", "回数",
    ]
    if not any(kw in query for kw in count_kws):
        return False
    # ランキング・全体集計・グループ集計クエリは aggregate_for_query に任せる
    exclude_kws = [
        "ランキング", "多い順", "最多", "最も多い", "一番多い", "上位", "頻度",
        "ごと", "種別別", "毎に",
    ]
    if any(kw in query for kw in exclude_kws):
        return False
    return bool(extract_count_subject(query))


def is_seasonal_tendency_query(query: str) -> bool:
    """「夏に発生しやすい」のような年指定なし季節傾向クエリを検出する。"""
    has_season = any(w in query for w in SEASON_WORD_MAP) or '季節' in query
    if not has_season:
        return False
    if not any(kw in query for kw in SEASONAL_TENDENCY_KWS):
        return False
    # 年指定があれば既存の date-filtered RAG に任せる
    return not bool(re.search(r'\d{4}年', query))


def parse_target_season(query: str) -> str | None:
    """クエリ中に最初に出現する季節語を対象季節として返す。"""
    first_pos, result = len(query), None
    for word, season in SEASON_WORD_MAP.items():
        pos = query.find(word)
        if pos != -1 and pos < first_pos:
            first_pos, result = pos, season
    return result


def extract_tendency_subject(query: str) -> str:
    """「〇〇の故障傾向は？」「〇〇に関する傾向は？」から対象設備・事象名を抽出する。"""
    q = re.sub(r'^\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?(?:から|以降|以前|まで|頃)?の?', '', query).strip()
    patterns = [
        r'^(.+?)に関する(?:故障|障害|不具合)?(?:の)?(?:傾向|パターン|特徴)',
        r'^(.+?)の(?:故障|障害|不具合)?(?:の)?(?:傾向|パターン|特徴)',
        r'^(.+?)(?:は(?:どのような|なぜ|よく)|が(?:よく|多く))(?:故障|発生|起き)',
        r'^(.+?)(?:傾向|パターン|特徴)',
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            s = m.group(1).strip()
            s = re.sub(r'[のがはにをで]$', '', s).strip()
            s = re.sub(r'^(?:について|に関して)', '', s).strip()
            if len(s) >= 2 and s not in GENERIC_SUBJECTS:
                return s
    return ""


def is_equipment_tendency_query(query: str) -> bool:
    """特定設備・事象の傾向クエリを検出する（件数・集計・季節傾向クエリは除外）。"""
    if not any(kw in query for kw in EQUIP_TENDENCY_KWS):
        return False
    subject = extract_tendency_subject(query)
    if not subject or subject in GENERIC_SUBJECTS:
        return False
    # 「これまでの故障」「最近の障害」など、末尾が汎用故障語で
    # それを除いた部分が空またはテンポラル語なら設備名ではない。
    # 「VVVF故障」など設備名＋故障はOK（VVVF が残るため除外されない）。
    core = re.sub(r'(?:故障|障害|不具合|トラブル)$', '', subject).strip()
    core = re.sub(r'[のがはにをで]$', '', core).strip()
    if not core or core in GENERIC_SUBJECTS:
        return False
    return True


def is_generic_failure_tendency_query(query: str) -> bool:
    """「故障の特徴は？」「故障の傾向は？」のような汎用故障傾向クエリを検出する。

    特定の設備名がなく「故障」「障害」等の汎用語が主語の傾向・特徴クエリに対応する。
    """
    if not any(kw in query for kw in EQUIP_TENDENCY_KWS):
        return False
    if is_equipment_tendency_query(query):
        return False
    if is_seasonal_tendency_query(query) or is_aggregation_query(query):
        return False
    return any(kw in query for kw in ["故障", "障害", "不具合", "トラブル"])


def is_failure_severity_listing_query(query: str) -> bool:
    """「〇〇の重故障は？」「重故障の一覧は？」のような故障レベル一覧取得クエリを検出する。

    中央監視システムが発報する「重故障」「軽故障」は設備名ではなく警報レベルであり、
    標準RAG(top-5)では全件を網羅できない。全件スキャンで対応する。
    件数集計・傾向クエリは別パスで処理するためここでは除外する。
    """
    if not any(kw in query for kw in FAILURE_SEVERITY_LEVELS):
        return False
    if is_aggregation_query(query) or is_specific_failure_count_query(query):
        return False
    if any(kw in query for kw in EQUIP_TENDENCY_KWS):
        return False
    return True
