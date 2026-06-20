"""
日本語トークナイザ (SudachiPy + neologdn)
BM25Retriever の preprocess_func として使う。
"""
import json
import logging
import pathlib
import site

import neologdn
import sudachipy

from normalizer import normalize_notation

log = logging.getLogger(__name__)

_FILTER_POS = frozenset({'助詞', '助動詞', '記号', '空白', '補助記号'})

_MODE_MAP = {
    'A': sudachipy.SplitMode.A,
    'B': sudachipy.SplitMode.B,
    'C': sudachipy.SplitMode.C,
}


def _find_system_dic() -> pathlib.Path:
    for sp in site.getsitepackages():
        p = pathlib.Path(sp) / 'sudachidict_full' / 'resources' / 'system.dic'
        if p.exists():
            return p
    # fallback: user site-packages
    p = pathlib.Path(site.getusersitepackages()) / 'sudachidict_full' / 'resources' / 'system.dic'
    if p.exists():
        return p
    raise FileNotFoundError(
        "sudachidict_full の system.dic が見つかりません。"
        "`pip install sudachidict_full` を実行してください。"
    )


class JapaneseTokenizer:
    def __init__(self, mode: str = 'C', user_dic_path: str | None = None):
        self._mode_str = mode.upper()
        self._mode = _MODE_MAP.get(self._mode_str, sudachipy.SplitMode.C)
        self._user_dic_path = user_dic_path
        self._sys_dic = _find_system_dic()
        self._dict_obj: sudachipy.Dictionary | None = None
        self._tok_obj = None
        self._build()

    def _build(self) -> None:
        config: dict = {'systemDict': str(self._sys_dic)}
        if self._user_dic_path and pathlib.Path(self._user_dic_path).exists():
            config['userDict'] = [self._user_dic_path]
        self._dict_obj = sudachipy.Dictionary(config=json.dumps(config))
        self._tok_obj = self._dict_obj.create()

    def tokenize(self, text: str) -> list[str]:
        text = neologdn.normalize(text)
        text = normalize_notation(text)
        try:
            morphemes = self._tok_obj.tokenize(text, self._mode)
        except Exception as exc:
            log.warning("SudachiPy tokenize error: %s", exc)
            return list(text)
        return [
            m.surface()
            for m in morphemes
            if m.part_of_speech()[0] not in _FILTER_POS and m.surface().strip()
        ]

    def reload(self, user_dic_path: str | None = None) -> None:
        self._user_dic_path = user_dic_path
        self._build()
        log.info("トークナイザ再ロード完了 (mode=%s, user_dic=%s)", self._mode_str, self._user_dic_path)

    @property
    def mode(self) -> str:
        return self._mode_str


_tokenizer: JapaneseTokenizer | None = None


def init_tokenizer(mode: str = 'C', user_dic_path: str | None = None) -> None:
    global _tokenizer
    _tokenizer = JapaneseTokenizer(mode=mode, user_dic_path=user_dic_path)
    log.info("トークナイザ初期化完了 (mode=%s, user_dic=%s)", mode, user_dic_path)


def get_tokenizer() -> JapaneseTokenizer:
    if _tokenizer is None:
        init_tokenizer()
    return _tokenizer
