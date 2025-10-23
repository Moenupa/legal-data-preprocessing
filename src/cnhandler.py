import re
import sys

from pycnnum import cn2num


def chinese_to_int(s: str) -> int:
    if s is None:
        return 0

    assert isinstance(s, str)
    return int(s) if s.isdigit() else cn2num(s)


def get_months(year: str | None, month: str | None) -> int:
    return chinese_to_int(year) * 12 + chinese_to_int(month)


# Common regex fragments
YEAR = r"([\d]{1,2}|[一二三四五六七八九十]{1,3})年"
MONTH = r"零?([\d]{1,2}|[一二三四五六七八九十]{1,3})个?月"


FULL_PATTERNS = [
    rf"(?:有期徒刑){YEAR}{MONTH}",
    rf"{YEAR}{MONTH}(?:有期徒刑)",
]
YEAR_ONLY_PATTERNS = [
    rf"(?:有期徒刑){YEAR}",
    rf"{YEAR}(?:有期徒刑)",
]
MONTH_ONLY_PATTERNS = [
    rf"(?:有期徒刑|拘役){MONTH}",
    rf"{MONTH}(?:有期徒刑|拘役)",
]


def extract_months(text: str) -> int | None:
    if text.isdigit():
        return int(text)

    # last match for full pattern matches
    for pattern in FULL_PATTERNS:
        match = re.findall(pattern, text)
        if match:
            return get_months(*match[-1])
    for pattern in MONTH_ONLY_PATTERNS:
        match = re.findall(pattern, text)
        if match:
            return get_months(year=None, month=match[-1])
    for pattern in YEAR_ONLY_PATTERNS:
        match = re.findall(pattern, text)
        if match:
            return get_months(year=match[-1], month=None)

    # warn if no match, treated as 0
    print("WARN: no match, treated as 0.", text, file=sys.stderr)
    return 0
