"""
ユーザー辞書エントリの SQLite CRUD
"""
import csv
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "user_dict.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS user_dict_entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                surface     TEXT NOT NULL,
                reading     TEXT NOT NULL,
                pos         TEXT NOT NULL DEFAULT '名詞,固有名詞,一般',
                cost        INTEGER NOT NULL DEFAULT 5000,
                normalized  TEXT NOT NULL DEFAULT '',
                enabled     INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT,
                updated_at  TEXT
            )
        """)
        con.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_all(search: str = '', page: int = 1, page_size: int = 20) -> dict:
    offset = (page - 1) * page_size
    with _conn() as con:
        if search:
            pattern = f'%{search}%'
            rows = con.execute(
                "SELECT id,surface,reading,pos,cost,normalized,enabled,created_at,updated_at "
                "FROM user_dict_entries "
                "WHERE surface LIKE ? OR reading LIKE ? OR normalized LIKE ? "
                "ORDER BY surface ASC LIMIT ? OFFSET ?",
                (pattern, pattern, pattern, page_size, offset),
            ).fetchall()
            total = con.execute(
                "SELECT COUNT(*) FROM user_dict_entries "
                "WHERE surface LIKE ? OR reading LIKE ? OR normalized LIKE ?",
                (pattern, pattern, pattern),
            ).fetchone()[0]
        else:
            rows = con.execute(
                "SELECT id,surface,reading,pos,cost,normalized,enabled,created_at,updated_at "
                "FROM user_dict_entries ORDER BY surface ASC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
            total = con.execute("SELECT COUNT(*) FROM user_dict_entries").fetchone()[0]

    return {
        'items': [dict(r) for r in rows],
        'total': total,
        'page': page,
        'page_size': page_size,
    }


def add_entry(
    surface: str,
    reading: str,
    pos: str = '名詞,固有名詞,一般',
    cost: int = 5000,
    normalized: str = '',
    enabled: int = 1,
) -> int:
    now = _now()
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO user_dict_entries "
            "(surface,reading,pos,cost,normalized,enabled,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (surface, reading, pos, cost, normalized, enabled, now, now),
        )
        con.commit()
        return cur.lastrowid


def update_entry(entry_id: int, **kwargs) -> bool:
    allowed = {'surface', 'reading', 'pos', 'cost', 'normalized', 'enabled'}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    updates['updated_at'] = _now()
    sets = ', '.join(f'{k}=?' for k in updates)
    vals = list(updates.values()) + [entry_id]
    with _conn() as con:
        cur = con.execute(f"UPDATE user_dict_entries SET {sets} WHERE id=?", vals)
        con.commit()
        return cur.rowcount > 0


def delete_entry(entry_id: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM user_dict_entries WHERE id=?", (entry_id,))
        con.commit()
        return cur.rowcount > 0


def get_enabled_entries() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id,surface,reading,pos,cost,normalized "
            "FROM user_dict_entries WHERE enabled=1",
        ).fetchall()
    return [dict(r) for r in rows]


def import_from_csv(rows: list[dict]) -> int:
    now = _now()
    count = 0
    with _conn() as con:
        for r in rows:
            surface = str(r.get('surface', '')).strip()
            reading = str(r.get('reading', '')).strip()
            if not surface or not reading:
                continue
            try:
                cost = int(r.get('cost', 5000))
            except (ValueError, TypeError):
                cost = 5000
            try:
                enabled = int(r.get('enabled', 1))
            except (ValueError, TypeError):
                enabled = 1
            con.execute(
                "INSERT INTO user_dict_entries "
                "(surface,reading,pos,cost,normalized,enabled,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    surface, reading,
                    str(r.get('pos', '名詞,固有名詞,一般')),
                    cost,
                    str(r.get('normalized', '')),
                    enabled,
                    now, now,
                ),
            )
            count += 1
        con.commit()
    return count


def export_to_csv() -> str:
    with _conn() as con:
        rows = con.execute(
            "SELECT surface,reading,pos,cost,normalized,enabled "
            "FROM user_dict_entries ORDER BY id",
        ).fetchall()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=['surface', 'reading', 'pos', 'cost', 'normalized', 'enabled'])
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
    return buf.getvalue()


def get_all_surfaces() -> set[str]:
    with _conn() as con:
        rows = con.execute("SELECT surface FROM user_dict_entries").fetchall()
    return {r[0] for r in rows}


def find_by_surface(surface: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id,surface,reading,pos,cost,normalized,enabled "
            "FROM user_dict_entries WHERE surface=?",
            (surface,),
        ).fetchall()
    return [dict(r) for r in rows]
