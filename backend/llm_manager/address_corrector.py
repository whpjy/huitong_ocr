"""Conservative administrative-division correction for extracted addresses.

Only province, prefecture, and county names are considered.  Historical names
in the bundled dataset are accepted as-is, while an OCR-shaped one-character
substitution is repaired only when it has a unique candidate in the applicable
administrative hierarchy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "admin_divisions"
    / "areacodes-1981-2025.json"
)
_WHITESPACE = re.compile(r"\s+")
_BRACKETED_HOUSE_NUMBER = re.compile(
    r"[\[［【〔(（]([0-9OoIl|丨ZzSsBbGg]+)[\]］】〕)）](?=号)"
)
_SEPARATED_THREE_DIGIT_NUMBER = re.compile(
    r"(?<![0-9OoIl|丨ZzSsBbGg])"
    r"([0-9OoIl|丨ZzSsBbGg])[-－—/／·•]"
    r"([0-9OoIl|丨ZzSsBbGg])[-－—/／·•]"
    r"([0-9OoIl|丨ZzSsBbGg])"
    r"(?=(?:号|栋|幢|单元|楼|室|组|户))"
)
_ADDRESS_NUMBER_TOKEN = re.compile(
    r"[0-9OoIl|丨ZzSsBbGg]+(?=(?:号|栋|幢|单元|楼|室|组|户))"
)
_NUMBER_OCR_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "丨": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "B": "8",
        "b": "8",
        "G": "6",
        "g": "6",
    }
)
_MASK_CHARACTERS = frozenset("*＊×Xx")
_GENERIC_NAMES = frozenset({"市辖区", "县", "省直辖县级行政区划"})
_PROVINCE_ALIASES = {
    "内蒙古自治区": ("内蒙古",),
    "广西壮族自治区": ("广西",),
    "西藏自治区": ("西藏",),
    "宁夏回族自治区": ("宁夏",),
    "新疆维吾尔自治区": ("新疆",),
    "香港特别行政区": ("香港",),
    "澳门特别行政区": ("澳门",),
}


@dataclass(frozen=True)
class _Division:
    name: str
    children: tuple["_Division", ...] = ()


@dataclass(frozen=True)
class _Match:
    consumed: int
    replacement: str
    divisions: tuple[_Division, ...]
    corrected: bool


def _parse_division(value: Any) -> _Division | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    children = tuple(
        child
        for item in value.get("children", [])
        if (child := _parse_division(item)) is not None
    )
    return _Division(name=name.strip(), children=children)


@lru_cache(maxsize=1)
def _provinces() -> tuple[_Division, ...]:
    """Load the bundled snapshot once; unavailable data disables correction."""

    try:
        payload = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return tuple(
        division
        for item in items
        if (division := _parse_division(item)) is not None
    )


def _unique_divisions(divisions: Iterable[_Division]) -> tuple[_Division, ...]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[_Division] = []
    for division in divisions:
        identity = (division.name, tuple(child.name for child in division.children))
        if identity not in seen:
            seen.add(identity)
            unique.append(division)
    return tuple(unique)


def _forms(
    division: _Division,
    *,
    province_level: bool,
) -> tuple[str, ...]:
    aliases = _PROVINCE_ALIASES.get(division.name, ()) if province_level else ()
    return (division.name, *aliases)


def _difference_count(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(a != b for a, b in zip(left, right))


def _match_at(
    address: str,
    offset: int,
    divisions: Iterable[_Division],
    *,
    province_level: bool = False,
    allow_correction: bool = True,
) -> _Match | None:
    candidates = [
        (division, form)
        for division in divisions
        if division.name not in _GENERIC_NAMES
        for form in _forms(division, province_level=province_level)
    ]

    exact = [item for item in candidates if address.startswith(item[1], offset)]
    if exact:
        longest = max(len(form) for _, form in exact)
        exact = [(division, form) for division, form in exact if len(form) == longest]
        matched_text = address[offset : offset + longest]
        return _Match(
            consumed=longest,
            replacement=matched_text,
            divisions=_unique_divisions(division for division, _ in exact),
            corrected=False,
        )

    if not allow_correction:
        return None

    fuzzy: list[tuple[_Division, str]] = []
    for division, form in candidates:
        observed = address[offset : offset + len(form)]
        if (
            len(observed) == len(form)
            and not any(character in _MASK_CHARACTERS for character in observed)
            and _difference_count(observed, form) == 1
        ):
            fuzzy.append((division, form))

    replacement_names = {form for _, form in fuzzy}
    if len(replacement_names) != 1:
        return None
    replacement = replacement_names.pop()
    return _Match(
        consumed=len(replacement),
        replacement=replacement,
        divisions=_unique_divisions(
            division for division, form in fuzzy if form == replacement
        ),
        corrected=True,
    )


def _children(divisions: Iterable[_Division]) -> tuple[_Division, ...]:
    return _unique_divisions(
        child for division in divisions for child in division.children
    )


def _grandchildren(divisions: Iterable[_Division]) -> tuple[_Division, ...]:
    return _children(_children(divisions))


def _apply_match(address: str, offset: int, match: _Match) -> tuple[str, int]:
    end = offset + match.consumed
    if match.corrected:
        address = address[:offset] + match.replacement + address[end:]
    return address, offset + len(match.replacement)


def clean_address_number_ocr(value: str) -> str:
    """Repair OCR-shaped characters in explicit address number positions.

    Replacements are deliberately limited to tokens immediately followed by an
    address unit.  Letters elsewhere in an address are therefore untouched.
    Brackets around a number are treated as two misread ``1`` characters, and
    separators are removed only from the common three-single-digit OCR shape.
    """

    address = _WHITESPACE.sub("", value or "")
    if not address:
        return address

    address = _BRACKETED_HOUSE_NUMBER.sub(r"1\g<1>1", address)
    address = _SEPARATED_THREE_DIGIT_NUMBER.sub(r"\1\2\3", address)

    def normalize_token(match: re.Match[str]) -> str:
        token = match.group(0)
        # A digit anchor prevents ordinary Latin words from being interpreted
        # as a house number (for example, an English label next to ``号``).
        if not any(character.isdigit() for character in token):
            return token
        return token.translate(_NUMBER_OCR_TRANSLATION)

    return _ADDRESS_NUMBER_TOKEN.sub(normalize_token, address)


def correct_address_divisions(value: str) -> str:
    """Correct unique one-character OCR errors in address division names.

    The detailed address after the county level is never inspected or changed.
    If the dataset is missing or a match is ambiguous, the original value is
    returned for that portion. Masking in the detailed address is preserved;
    masking inside a division name prevents that division from being guessed.
    """

    address = clean_address_number_ocr(value)
    if not address:
        return address

    provinces = _provinces()
    if not provinces:
        return address

    offset = 0
    province = _match_at(address, offset, provinces, province_level=True)
    if province:
        address, offset = _apply_match(address, offset, province)
        prefectures = _children(province.divisions)
        prefecture = _match_at(address, offset, prefectures)
        if prefecture:
            address, offset = _apply_match(address, offset, prefecture)
            county = _match_at(address, offset, _children(prefecture.divisions))
            if county:
                address, _ = _apply_match(address, offset, county)
            return address

        # Addresses often omit the prefecture, for example 山东省滕州市.
        county = _match_at(address, offset, _grandchildren(province.divisions))
        if county:
            address, _ = _apply_match(address, offset, county)
        return address

    prefectures = _children(provinces)
    prefecture = _match_at(address, offset, prefectures)
    if prefecture:
        address, offset = _apply_match(address, offset, prefecture)
        county = _match_at(address, offset, _children(prefecture.divisions))
        if county:
            address, _ = _apply_match(address, offset, county)
        return address

    # County-only addresses are corrected only when the nationwide candidate
    # is unique. Duplicate county names therefore remain untouched.
    county = _match_at(address, offset, _grandchildren(provinces))
    if county:
        address, _ = _apply_match(address, offset, county)
    return address
