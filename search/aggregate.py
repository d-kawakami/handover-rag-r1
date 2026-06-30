"""集計関数群（カウント・ランキング・季節別・設備別）。

LLM へは集計済みテキストと代表レコードのみを渡し、トークン数を一定に保つ。
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from langchain_core.documents import Document

from normalizer import normalize_notation
from search.classify import (
    extract_count_subject,
    extract_tendency_subject,
    parse_min_count,
    parse_target_season,
    parse_top_n,
)
from search.dates import doc_in_range, parse_date_filter, parse_date_str
from search.filters import SearchFilter
from search.patterns import (
    FAILURE_SEVERITY_LEVELS,
    FAILURE_TYPE_SET,
    SEASON_LABEL,
    SEASON_MONTHS,
    SYNONYM_GROUPS,
    extract_equip_keywords,
)
from state import state
from tokenizer import get_tokenizer

log = logging.getLogger(__name__)


# ─── 共通ヘルパー ─────────────────────────────────────────────────────────
def _filter_base(sf: SearchFilter | None) -> list[Document]:
    return [d for d in state.all_docs if (sf is None or sf.matches(d))]


def _date_label(from_date, to_date) -> str:
    return f"{from_date}〜{to_date}" if (from_date or to_date) else "全期間"


def _content_snippet(doc: Document, max_len: int = 60) -> str:
    m = re.search(r"^内容: (.+)$", doc.page_content, re.MULTILINE)
    return m.group(1)[:max_len] if m else ""


def _normalized_target(doc: Document) -> str:
    return normalize_notation(doc.metadata.get("故障", "") + "\n" + doc.page_content)


# ─── 1. ランキング・集計クエリ ────────────────────────────────────────────
def aggregate_for_query(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """集計クエリに対して全件スキャンで集計結果テキストとサンプルドキュメントを返す。"""
    from_date, to_date = parse_date_filter(query)
    min_count = parse_min_count(query)
    top_n = parse_top_n(query)

    filtered = [d for d in _filter_base(sf) if doc_in_range(d, from_date, to_date)]
    if not filtered:
        return "指定期間の記録が見つかりませんでした。", []

    date_label = f"{from_date} 〜 {to_date}" if (from_date or to_date) else "全期間"
    is_groupby = any(kw in query for kw in ["ごと", "種別別", "毎に", "全種別", "各種別"])
    is_failure_focus = (not is_groupby) and any(kw in query for kw in ["故障", "障害", "不具合", "トラブル"])

    if is_failure_focus:
        target_docs = [
            d for d in filtered
            if d.metadata.get("故障", "").strip()
            or "故障" in d.metadata.get("種別", "")
        ]
    else:
        target_docs = filtered

    if not target_docs:
        return f"期間 {date_label} に該当する故障記録が見つかりませんでした。", []

    shubetsu_counter: Counter = Counter(
        d.metadata.get("種別", "（不明）").strip() or "（不明）"
        for d in target_docs
    )

    # 故障コードごとにカウント＋代表的な内容テキストを最大3件収集
    kosho_counter: Counter = Counter()
    kosho_samples: dict[str, list[str]] = {}
    for d in target_docs:
        ko = d.metadata.get("故障", "").strip()
        if not ko:
            continue
        kosho_counter[ko] += 1
        samples = kosho_samples.setdefault(ko, [])
        if len(samples) < 3:
            content = _content_snippet(d, max_len=60).strip()
            if content and content not in samples:
                samples.append(content)

    lines = [
        f"集計期間: {date_label}",
        f"対象記録数: {len(target_docs)}件",
        "",
        f"■ 種別ランキング（{min_count}件以上、上位{top_n}位）:",
    ]
    shown = 0
    for rank, (name, count) in enumerate(shubetsu_counter.most_common(top_n * 2), 1):
        if count >= min_count:
            lines.append(f"  {rank}位: {name} — {count}件")
            shown += 1
            if shown >= top_n:
                break
    if shown == 0:
        lines.append(f"  （{min_count}件以上の種別はありませんでした）")

    if kosho_counter:
        lines += ["", f"■ 故障機器別ランキング（{min_count}件以上、上位{top_n}位）:",
                  "  ※ 機器コードが数字の場合は、下の「記録内容」からどの機器・設備かを読み取れます"]
        shown = 0
        for rank, (code, count) in enumerate(kosho_counter.most_common(top_n * 2), 1):
            if count >= min_count:
                lines.append(f"  {rank}位: 機器/故障識別「{code}」— {count}件発生")
                for s in kosho_samples.get(code, []):
                    lines.append(f"    └ 記録内容: {s}")
                shown += 1
                if shown >= top_n:
                    break
        if shown == 0:
            lines.append(f"  （{min_count}件以上の故障はありませんでした）")

    log.info("集計完了: 期間=%s 対象=%d件", date_label, len(target_docs))
    return "\n".join(lines), target_docs[:top_n]


# ─── 2. 特定設備・故障の発生件数 ────────────────────────────────────────
def count_specific_failure(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """全種別の記録から、特定の故障名・設備名の出現件数を正確にカウントする。

    故障名が明示されている場合は種別フィルタ不要 — 故障名自体がフィルタになる。
    """
    from_date, to_date = parse_date_filter(query)
    subject = normalize_notation(extract_count_subject(query))
    date_label = _date_label(from_date, to_date)

    range_docs = [d for d in _filter_base(sf) if doc_in_range(d, from_date, to_date)]
    if not range_docs:
        return f"期間 {date_label} に記録が見つかりませんでした。", []

    # 末尾の「故障」「障害」「不具合」を除いたバリアントも検索
    search_variants: list[str] = [subject]
    stripped = re.sub(r'(?:故障|障害|不具合)+$', '', subject).strip()
    if stripped and stripped != subject and len(stripped) >= 2:
        search_variants.append(stripped)

    # 同義語グループで search_variants を展開
    for group in SYNONYM_GROUPS:
        if any(normalize_notation(v) in group for v in search_variants):
            for syn in group:
                if syn not in search_variants:
                    search_variants.append(syn)

    # 複合語（「原水ポンプVVVF」等）向け: トークン分割して AND マッチも試みる
    try:
        tok_terms = [t for t in get_tokenizer().tokenize(subject) if len(t) >= 2]
    except Exception:
        tok_terms = []
    tok_conds: list[tuple[str, ...]] = []
    for t in tok_terms:
        group = next((g for g in SYNONYM_GROUPS if t in g), None)
        tok_conds.append(tuple(group) if group else (t,))

    # 「の」で分割した場合の前後候補
    no_parts = [p.strip() for p in subject.split("の") if p.strip()]
    if len(no_parts) >= 2:
        head_phrase = no_parts[0]
        tail_phrase = "の".join(no_parts[1:])
        try:
            htoks = [t for t in get_tokenizer().tokenize(head_phrase) if len(t) >= 2]
            ttoks = [t for t in get_tokenizer().tokenize(tail_phrase) if len(t) >= 2]
        except Exception:
            htoks, ttoks = [], []
        head_cands = [head_phrase] + ([htoks[0]] if htoks else [])
        tail_first = next((t for t in ttoks if len(t) >= 3), ttoks[0] if ttoks else "")
        tail_cands = [c for c in [tail_phrase, tail_first] if c]
    else:
        head_cands, tail_cands = [], []

    def doc_matches(d: Document) -> bool:
        target = _normalized_target(d)
        if any(v in target for v in search_variants):
            return True
        if len(tok_conds) >= 2:
            if all(any(syn in target for syn in cond) for cond in tok_conds):
                return True
        if head_cands and tail_cands:
            if any(c in target for c in head_cands) and any(c in target for c in tail_cands):
                return True
        return False

    matched = [d for d in range_docs if doc_matches(d)]

    # 「〇〇の故障は何件」のように故障を明示する質問では、報告・処置などを含む
    # 全記録ではなく故障系の記録（種別=故障/故障処置）だけを数える。これにより
    # 「故障件数」を一意に確定でき、LLM が総数と内訳のどちらを答えるか揺れる問題を防ぐ。
    # 「全体で」等の総数を明示する質問、および「異常」のみ（例: 上限異常）の質問は対象外。
    FAILURE_TYPES = {"故障", "故障処置"}
    is_failure_scoped = (
        any(w in query for w in ("故障", "障害", "不具合"))
        and not any(w in query for w in ("全体", "すべて", "全て", "総数", "全部"))
    )
    if is_failure_scoped:
        matched = [
            d for d in matched
            if d.metadata.get("種別", "").strip() in FAILURE_TYPES
        ]
    count_label = "故障記録件数" if is_failure_scoped else "記録件数"

    shubetsu_count: Counter = Counter(
        d.metadata.get("種別", "（不明）").strip() or "（不明）"
        for d in matched
    )
    breakdown = "、".join(f"{k} {v}件" for k, v in shubetsu_count.most_common())

    kosho_counter: Counter = Counter()
    kosho_samples: dict[str, list[str]] = {}
    for d in matched:
        ko = d.metadata.get("故障", "").strip() or "（故障コードなし）"
        kosho_counter[ko] += 1
        samples = kosho_samples.setdefault(ko, [])
        if len(samples) < 2:
            snip = _content_snippet(d, max_len=50).strip()
            if snip and snip not in samples:
                samples.append(snip)

    lines = [
        f"集計期間: {date_label}",
        f"検索語: 「{subject}」（同義語含む: {', '.join(search_variants)}）",
        "",
        f"■ 「{subject}」（同義語含む）の{count_label}: {len(matched)}件",
    ]
    if breakdown:
        lines.append(f"  （種別内訳: {breakdown}）")

    if len(kosho_counter) > 1:
        lines += ["", "■ 故障コード/機器別内訳:"]
        for code, cnt in kosho_counter.most_common():
            lines.append(f"  「{code}」: {cnt}件")
            for s in kosho_samples.get(code, []):
                lines.append(f"    └ 記録内容例: {s}")

    if matched:
        sample_docs = sorted(matched, key=lambda d: d.metadata.get("日付", ""))
        sample_docs = sample_docs[-state.equip_tendency_samples:]
        lines += ["", f"■ 該当記録（全{len(matched)}件・最新{len(sample_docs)}件を表示）:"]
        for d in sample_docs:
            date_val = d.metadata.get("日付", "不明")
            shubetsu = d.metadata.get("種別", "")
            snippet = _content_snippet(d, max_len=60)
            lines.append(f"  {date_val} [{shubetsu}] — {snippet}")
    else:
        lines.append("（該当する記録は見つかりませんでした）")

    log.info("特定故障カウント: subject=%s variants=%s 期間=%s マッチ=%d件",
             subject, search_variants, date_label, len(matched))
    return "\n".join(lines), matched


# ─── 3. 季節傾向 ─────────────────────────────────────────────────────────
def aggregate_by_season(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """季節傾向クエリに対し、月別・季節別の故障集計と対象季節の記録を返す。"""
    from search.dates import doc_in_month_set

    target_season = parse_target_season(query)
    base_docs = _filter_base(sf)

    month_counter: Counter = Counter()
    month_docs: dict[int, list[Document]] = {m: [] for m in range(1, 13)}
    for d in base_docs:
        doc_date = parse_date_str(d.metadata.get("日付", ""))
        if doc_date:
            month_counter[doc_date.month] += 1
            month_docs[doc_date.month].append(d)

    total_dated = sum(month_counter.values())

    lines = [
        f"集計対象: 全期間 日付あり記録 {total_dated}件",
        "",
        "■ 月別記録件数（全期間・全種別）:",
    ]
    for m in range(1, 13):
        cnt = month_counter[m]
        unit = max(total_dated // 100, 1)
        bar = "█" * min(cnt // unit, 20)
        lines.append(f"  {m:2d}月: {cnt:4d}件 {bar}")

    lines += ["", "■ 季節別合計件数:"]
    for s, months in SEASON_MONTHS.items():
        cnt = sum(month_counter[m] for m in months)
        lines.append(f"  {SEASON_LABEL[s]}: {cnt}件")

    if target_season:
        target_months = set(SEASON_MONTHS[target_season])
        target_docs = [d for d in base_docs if doc_in_month_set(d, target_months)]

        lines += [
            "",
            f"■ {SEASON_LABEL[target_season]}の故障傾向:",
            f"  該当記録数: {len(target_docs)}件",
        ]

        shubetsu_cnt: Counter = Counter(
            d.metadata.get("種別", "（不明）").strip() or "（不明）"
            for d in target_docs
        )
        if shubetsu_cnt:
            lines.append("  種別内訳: " + "、".join(
                f"{k} {v}件" for k, v in shubetsu_cnt.most_common(5)
            ))

        kosho_cnt: Counter = Counter()
        kosho_samples: dict[str, list[str]] = {}
        for d in target_docs:
            ko = d.metadata.get("故障", "").strip()
            if ko:
                kosho_cnt[ko] += 1
                slist = kosho_samples.setdefault(ko, [])
                if len(slist) < 2:
                    snip = _content_snippet(d, max_len=50).strip()
                    if snip and snip not in slist:
                        slist.append(snip)

        if kosho_cnt:
            lines += ["", "  頻発する故障コード（上位10件）:"]
            for code, cnt in kosho_cnt.most_common(10):
                lines.append(f"    「{code}」: {cnt}件")
                for s in kosho_samples.get(code, []):
                    lines.append(f"      └ {s}")

        sample = sorted(target_docs, key=lambda d: d.metadata.get("日付", ""), reverse=True)[:30]
        if sample:
            lines += ["", "  代表的な記録（最新30件）:"]
            for d in sample:
                date_val = d.metadata.get("日付", "不明")
                shubetsu = d.metadata.get("種別", "")
                snippet = _content_snippet(d, max_len=60)
                lines.append(f"    {date_val} [{shubetsu}] — {snippet}")

        source_docs = target_docs
    else:
        # 季節指定なし：各月から3件ずつサンプルして月別分析のみ
        source_docs = []
        for m in range(1, 13):
            source_docs.extend(month_docs[m][:3])

    log.info("季節傾向集計: target=%s 全記録=%d 対象=%d件",
             target_season, total_dated, len(source_docs))
    return "\n".join(lines), source_docs


# ─── 4. 設備別傾向 ──────────────────────────────────────────────────────
def aggregate_by_equipment(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """特定設備傾向クエリに対し、全件スキャン＋集計済みコンテキストを返す。

    LLMへ渡すコンテキストは集計統計＋代表記録 state.equip_tendency_samples 件に
    限定し、マッチ件数にかかわらずトークン数を一定に保つ。
    """
    subject = extract_tendency_subject(query)
    if not subject:
        kws = extract_equip_keywords(query)
        subject = kws[0] if kws else ""

    from_date, to_date = parse_date_filter(query)
    date_label = _date_label(from_date, to_date)

    subject_norm = normalize_notation(subject)
    variants: list[str] = [subject_norm]
    stripped = re.sub(r'(?:故障|障害|不具合)+$', '', subject_norm).strip()
    if stripped and stripped != subject_norm and len(stripped) >= 2:
        variants.append(stripped)
    for group in SYNONYM_GROUPS:
        if any(normalize_notation(v) in group for v in variants):
            for syn in group:
                if syn not in variants:
                    variants.append(syn)

    range_docs = [d for d in _filter_base(sf) if doc_in_range(d, from_date, to_date)]

    # Step1: フレーズ完全一致
    matched = [d for d in range_docs if any(v in _normalized_target(d) for v in variants)]

    # Step2: フレーズ不一致時はトークン AND 検索（「循環ポンプのVVVF故障」等）
    if not matched:
        try:
            tok_terms = [t for t in get_tokenizer().tokenize(subject_norm) if len(t) >= 2]
        except Exception:
            tok_terms = []
        tok_conds: list[tuple[str, ...]] = []
        for t in tok_terms:
            grp = next((g for g in SYNONYM_GROUPS if t in g), None)
            tok_conds.append(tuple(grp) if grp else (t,))
        if len(tok_conds) >= 2:
            matched = [
                d for d in range_docs
                if all(any(syn in _normalized_target(d) for syn in cond) for cond in tok_conds)
            ]

    if not matched:
        return f"「{subject}」に関する記録が見つかりませんでした（期間: {date_label}）。", []

    return _format_tendency_lines(
        matched=matched,
        date_label=date_label,
        sample_limit=state.equip_tendency_samples,
        header=[
            f"対象設備: 「{subject}」（検索語: {', '.join(variants)}）",
            f"集計期間: {date_label}",
            f"該当記録数: {len(matched)}件",
        ],
        log_msg=("設備別傾向集計: subject=%s variants=%s 期間=%s マッチ=%d件",
                 subject, variants, date_label),
    )


# ─── 5. 汎用故障傾向 ──────────────────────────────────────────────────
def aggregate_failure_generic(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """汎用故障傾向クエリに対し、種別=故障/故障処置に絞った集計と代表記録を返す。"""
    from_date, to_date = parse_date_filter(query)
    date_label = _date_label(from_date, to_date)

    range_docs = [d for d in _filter_base(sf) if doc_in_range(d, from_date, to_date)]
    matched = [
        d for d in range_docs
        if d.metadata.get("種別", "").strip() in FAILURE_TYPE_SET
        or d.metadata.get("故障", "").strip()
    ]
    if not matched:
        return f"期間 {date_label} に故障・故障処置の記録が見つかりませんでした。", []

    return _format_tendency_lines(
        matched=matched,
        date_label=date_label,
        sample_limit=state.equip_tendency_samples,
        header=[
            f"集計期間: {date_label}",
            "対象: 故障・故障処置記録のみ（点検・報告等は除外）",
            f"該当記録数: {len(matched)}件",
        ],
        log_msg=("汎用故障傾向集計: 期間=%s マッチ=%d件", date_label),
    )


# ─── 6. 重大度別故障一覧 ─────────────────────────────────────────────────
def list_failures_by_severity(query: str, sf: SearchFilter | None = None) -> tuple[str, list[Document]]:
    """重故障・軽故障の全件一覧をスキャンして返す。

    「中央監視の重故障は？」のようなクエリでは「重故障」キーワードを直接
    全文スキャンし、日付フィルタ内のすべての該当レコードを返す。
    LLMへは state.equip_tendency_samples 件で上限を設けて渡す。
    """
    severity = next((kw for kw in FAILURE_SEVERITY_LEVELS if kw in query), "重故障")
    from_date, to_date = parse_date_filter(query)
    date_label = _date_label(from_date, to_date)

    range_docs = [d for d in _filter_base(sf) if doc_in_range(d, from_date, to_date)]
    matched = [d for d in range_docs if severity in _normalized_target(d)]

    if not matched:
        return f"期間 {date_label} に「{severity}」を含む記録が見つかりませんでした。", []

    shubetsu_counter: Counter = Counter(
        d.metadata.get("種別", "（不明）").strip() or "（不明）"
        for d in matched
    )
    month_counter: Counter = Counter()
    for d in matched:
        doc_date = parse_date_str(d.metadata.get("日付", ""))
        if doc_date:
            month_counter[doc_date.month] += 1

    lines = [
        f"集計期間: {date_label}",
        f"「{severity}」を含む記録: {len(matched)}件",
        "",
        "■ 種別内訳:",
    ]
    for k, v in shubetsu_counter.most_common():
        lines.append(f"  {k}: {v}件")

    if month_counter:
        lines += ["", "■ 月別件数:"]
        for mo in range(1, 13):
            cnt = month_counter.get(mo, 0)
            if cnt:
                lines.append(f"  {mo:2d}月: {cnt}件")

    sample_docs = sorted(matched, key=lambda d: d.metadata.get("日付", ""), reverse=True)
    sample_docs = sample_docs[:state.equip_tendency_samples]

    lines += [
        "",
        f"■ 全{len(matched)}件の記録（最新{len(sample_docs)}件を表示）:",
        "  ※ 記録が多い場合は日付範囲を指定して絞り込めます",
    ]
    for d in sorted(sample_docs, key=lambda d: d.metadata.get("日付", "")):
        date_val = d.metadata.get("日付", "不明")[:10]
        shubetsu = d.metadata.get("種別", "")
        snippet = _content_snippet(d, max_len=80)
        lines.append(f"  {date_val} [{shubetsu}] {snippet}")

    log.info("重大度別故障一覧: severity=%s 期間=%s マッチ=%d件",
             severity, date_label, len(matched))
    return "\n".join(lines), matched


# ─── 共通: 傾向集計レポート整形 ───────────────────────────────────────
def _format_tendency_lines(
    matched: list[Document],
    date_label: str,
    sample_limit: int,
    header: list[str],
    log_msg: tuple,
) -> tuple[str, list[Document]]:
    """設備別/汎用故障傾向で共通する集計レポートを組み立てる。"""
    shubetsu_counter: Counter = Counter(
        d.metadata.get("種別", "（不明）").strip() or "（不明）"
        for d in matched
    )

    month_counter: Counter = Counter()
    for d in matched:
        doc_date = parse_date_str(d.metadata.get("日付", ""))
        if doc_date:
            month_counter[doc_date.month] += 1

    kosho_counter: Counter = Counter()
    kosho_samples: dict[str, list[str]] = {}
    for d in matched:
        ko = d.metadata.get("故障", "").strip()
        if ko:
            kosho_counter[ko] += 1
            slist = kosho_samples.setdefault(ko, [])
            if len(slist) < 2:
                snip = _content_snippet(d, max_len=60).strip()
                if snip and snip not in slist:
                    slist.append(snip)

    lines = list(header)
    lines += ["", "■ 種別内訳:"]
    for k, v in shubetsu_counter.most_common():
        lines.append(f"  {k}: {v}件")

    if month_counter:
        lines += ["", "■ 月別記録件数:"]
        for mo in range(1, 13):
            cnt = month_counter.get(mo, 0)
            if cnt:
                lines.append(f"  {mo:2d}月: {cnt}件")

    if kosho_counter:
        lines += ["", "■ 故障コード/機器別内訳（上位10件）:"]
        for code, cnt in kosho_counter.most_common(10):
            lines.append(f"  「{code}」: {cnt}件")
            for s in kosho_samples.get(code, []):
                lines.append(f"    └ {s}")

    sample_docs = sorted(matched, key=lambda d: d.metadata.get("日付", ""), reverse=True)[:sample_limit]

    lines += [
        "",
        f"■ 代表的な記録（最新{len(sample_docs)}件 / 全{len(matched)}件）:",
        "  ※ LLMへは代表記録のみ渡しています。傾向分析は集計統計を根拠にしてください。",
    ]
    for d in sample_docs:
        date_val = d.metadata.get("日付", "不明")
        shubetsu = d.metadata.get("種別", "")
        snippet = _content_snippet(d, max_len=80)
        lines.append(f"  {date_val} [{shubetsu}] — {snippet}")

    msg_args = log_msg[1:] + (len(matched),)
    log.info(log_msg[0], *msg_args)
    return "\n".join(lines), sample_docs
