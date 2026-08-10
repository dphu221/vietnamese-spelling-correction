"""Transparent heuristic labels for displaying correction differences."""

from __future__ import annotations

import re
import unicodedata


CORE = re.compile(r"^([^A-Za-zÀ-ỹĐđ0-9]*)([A-Za-zÀ-ỹĐđ0-9]+)([^A-Za-zÀ-ỹĐđ0-9]*)$")
TEENCODE = {"k", "ko", "kh", "hok", "hong", "mik", "mk", "m", "tui", "dc", "đc", "chs", "nx"}
INITIAL_PAIRS = {
    ("tr", "ch"), ("ch", "tr"), ("s", "x"), ("x", "s"),
    ("d", "gi"), ("gi", "d"), ("r", "gi"), ("gi", "r"),
    ("n", "l"), ("l", "n"),
}
LABELS = {
    "teencode": "Teencode / viết tắt",
    "telex_vni": "Kiểu gõ Telex / VNI",
    "missing_diacritic": "Thiếu hoặc sai dấu",
    "initial_consonant": "Nhầm phụ âm đầu",
    "other": "Lỗi chính tả khác",
}


def token_core(token: str) -> str:
    match = CORE.match(token)
    return match.group(2) if match else token


def _plain(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )


def infer_error_type(original: str, replacement: str) -> tuple[str, str]:
    """Infer a human-readable category; this is deliberately not model output."""
    source = token_core(original).casefold()
    target = token_core(replacement).casefold()
    if source in TEENCODE:
        code = "teencode"
    elif any(char.isdigit() for char in source) or re.search(r"(?:aa|aw|dd|ee|oo|ow|uw|[sfrxj])$", source):
        code = "telex_vni"
    elif _plain(source) == _plain(target) and source != target:
        code = "missing_diacritic"
    elif any(source.startswith(left) and target.startswith(right) for left, right in INITIAL_PAIRS):
        code = "initial_consonant"
    else:
        code = "other"
    return code, LABELS[code]
