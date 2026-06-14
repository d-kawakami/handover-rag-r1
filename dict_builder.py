"""
SudachiPy ユーザー辞書のビルドとリロード
"""
import logging
import pathlib
import shutil
import site
import subprocess

from dict_db import get_enabled_entries
from tokenizer import get_tokenizer

log = logging.getLogger(__name__)

USER_DIC_PATH = pathlib.Path(__file__).parent / "user.dic"
_NOUN_CONNECT_ID = "4786"


def _find_system_dic() -> pathlib.Path:
    for sp in site.getsitepackages():
        p = pathlib.Path(sp) / 'sudachidict_full' / 'resources' / 'system.dic'
        if p.exists():
            return p
    p = pathlib.Path(site.getusersitepackages()) / 'sudachidict_full' / 'resources' / 'system.dic'
    if p.exists():
        return p
    raise FileNotFoundError("sudachidict_full の system.dic が見つかりません")


def _entry_to_csv_row(entry: dict) -> str:
    """dict entry → Sudachi ユーザー辞書 CSV 行 (18列)"""
    surface = entry['surface']
    reading = entry['reading']
    cost = str(entry.get('cost', 5000))
    normalized = entry.get('normalized') or ''
    pos = entry.get('pos', '名詞,固有名詞,一般')

    parts = [p.strip() for p in pos.split(',')]
    while len(parts) < 3:
        parts.append('*')
    pos1, pos2, pos3 = parts[0], parts[1], parts[2]

    # 18列: surface, left_id, right_id, cost, pos1-4, conj_type, conj_form,
    #        reading, written_form, normalized, splitting, split_a, split_b, ?, word_structure
    fields = [
        surface, _NOUN_CONNECT_ID, _NOUN_CONNECT_ID, cost,
        pos1, pos2, pos3, '*', '*', '*',
        reading,
        surface,         # written form
        normalized,      # normalized form (empty = same as surface)
        '*', '*', '*', '*', '*',
    ]
    # CSV escape: フィールドにカンマ・改行・ダブルクォートが含まれる場合に対応
    escaped = []
    for f in fields:
        if ',' in f or '"' in f or '\n' in f:
            escaped.append('"' + f.replace('"', '""') + '"')
        else:
            escaped.append(f)
    return ','.join(escaped)


def build_user_dic(entries: list[dict]) -> pathlib.Path:
    """
    enabled エントリから user.dic をビルドして返す。
    失敗時は例外を raise し、既存の user.dic を維持する。
    """
    if not entries:
        if USER_DIC_PATH.exists():
            USER_DIC_PATH.unlink()
        return USER_DIC_PATH

    sys_dic = _find_system_dic()
    tmp_csv = USER_DIC_PATH.with_suffix('.csv.tmp')
    tmp_dic = USER_DIC_PATH.with_suffix('.dic.tmp')

    try:
        csv_lines = [_entry_to_csv_row(e) for e in entries]
        tmp_csv.write_text('\n'.join(csv_lines) + '\n', encoding='utf-8')

        result = subprocess.run(
            ['sudachipy', 'ubuild', '-s', str(sys_dic), '-o', str(tmp_dic), str(tmp_csv)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sudachipy ubuild 失敗 (rc={result.returncode}):\n{result.stderr}"
            )

        # ビルド成功 → 本番パスに移動
        if USER_DIC_PATH.exists():
            backup = USER_DIC_PATH.with_suffix('.dic.bak')
            shutil.copy2(USER_DIC_PATH, backup)
        shutil.move(str(tmp_dic), str(USER_DIC_PATH))
        log.info("ユーザー辞書ビルド完了: %d エントリ → %s", len(entries), USER_DIC_PATH)
        return USER_DIC_PATH

    finally:
        if tmp_csv.exists():
            tmp_csv.unlink()
        if tmp_dic.exists():
            tmp_dic.unlink()


def rebuild_and_reload() -> dict:
    """DB の enabled エントリ → user.dic ビルド → tokenizer リロード"""
    entries = get_enabled_entries()
    try:
        build_user_dic(entries)
        user_dic = str(USER_DIC_PATH) if (entries and USER_DIC_PATH.exists()) else None
        get_tokenizer().reload(user_dic_path=user_dic)
        return {
            'status': 'ok',
            'built': len(entries),
            'message': f'{len(entries)} 件のエントリでユーザー辞書を更新しました',
        }
    except Exception as exc:
        log.exception("rebuild_and_reload 失敗")
        return {'status': 'error', 'built': 0, 'message': str(exc)}
