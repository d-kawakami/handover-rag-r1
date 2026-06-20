"""LLM 出力に混入する中国語を検出・除去するユーティリティ。"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# 中国語助詞「的」が修飾語として使われるパターン（例: 「なかしょうがいのねむ的内容」）
# 日本語では「的」の直前後に漢字/かな、直後に漢字が来る「〇〇的〇〇」形式は中国語的表現
_CHINESE_DE_PAT = re.compile(r'[぀-ヿ一-鿿]{2,}的[一-鿿]')


def is_likely_chinese(text: str) -> bool:
    """ひらがな・カタカナが含まれず漢字のみの場合に中国語と判定する。"""
    stripped = text.strip()
    if len(stripped) < 10:
        return False
    has_cjk = any('一' <= c <= '鿿' for c in stripped)
    has_hiragana = any('ぁ' <= c <= 'ゖ' for c in stripped)
    has_katakana = any('゠' <= c <= 'ヿ' for c in stripped)
    return has_cjk and not has_hiragana and not has_katakana


def has_chinese_mix(text: str) -> bool:
    """中国語文または中国語助詞「的」混入パターンを検出する。"""
    if _CHINESE_DE_PAT.search(text):
        return True
    for seg in re.split(r'[\n。！？!?、]', text):
        if is_likely_chinese(seg):
            return True
    return False


def filter_chinese_sentences(text: str) -> str:
    """テキストから中国語行・中国語「的」混入文を除去し、日本語部分のみを返す。"""
    filtered_lines = []
    for line in text.split('\n'):
        if is_likely_chinese(line) or _CHINESE_DE_PAT.search(line):
            log.warning("中国語混入行を除去: %.60s...", line)
            continue
        parts = line.split('。')
        kept = [p for p in parts if not is_likely_chinese(p) and not _CHINESE_DE_PAT.search(p)]
        if kept:
            filtered_lines.append('。'.join(kept))
    return '\n'.join(filtered_lines)


def sanitize_if_chinese(text: str) -> str:
    """中国語混入を検出した場合のみフィルタを適用する。検出されなければ無加工。"""
    if is_likely_chinese(text) or has_chinese_mix(text):
        log.warning("中国語混入を検出、フィルタ処理します")
        return filter_chinese_sentences(text).strip()
    return text
