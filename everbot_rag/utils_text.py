# everbot_rag/utils_text.py
from __future__ import annotations
import re
from typing import List, Optional

ARTICLE_RE = re.compile(r"第\s*(\d+)\s*條")
ARTICLE_RE_ALT = re.compile(r"^\s*(\d+)\s*條\s*$")

CHAPTER_SECTION_RE = re.compile(
    r"^\s*第\s*[一二三四五六七八九十百千零〇兩]+\s*(章|節|編|款|目|項)\s*.*$"
)

CH_NUM_MAP = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9
}

def has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))

def cjk_only(s: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", s or ""))

def clean_text(s: str) -> str:
    s = (s or "").replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_article_lines(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        ln2 = ln.strip()
        if not ln2:
            continue
        if CHAPTER_SECTION_RE.match(ln2):
            continue
        out.append(ln.rstrip())
    return out

def zh_num_to_int(s: str) -> Optional[int]:
    if not s:
        return None
    total = 0
    num = 0
    has_unit = False
    for ch in s:
        if ch in CH_NUM_MAP:
            num = CH_NUM_MAP[ch]
        elif ch == "十":
            has_unit = True
            if num == 0:
                num = 1
            total += num * 10
            num = 0
        elif ch == "百":
            has_unit = True
            if num == 0:
                num = 1
            total += num * 100
            num = 0
        elif ch == "千":
            has_unit = True
            if num == 0:
                num = 1
            total += num * 1000
            num = 0
        else:
            return None
    total += num
    if total == 0 and has_unit:
        return 10
    return total if total > 0 else None
