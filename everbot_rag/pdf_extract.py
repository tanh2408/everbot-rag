# everbot_rag/pdf_extract.py
from __future__ import annotations
import json
from typing import Dict, List, Tuple
import pdfplumber

from .utils_text import clean_text, clean_article_lines, ARTICLE_RE, ARTICLE_RE_ALT

def pdf_to_text(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)

def split_into_article_blocks(text: str) -> List[Tuple[int, List[str]]]:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    starts = []
    for i, ln in enumerate(lines):
        m = ARTICLE_RE.search(ln)
        if m:
            starts.append((i, int(m.group(1))))
            continue
        m2 = ARTICLE_RE_ALT.match(ln)
        if m2:
            starts.append((i, int(m2.group(1))))
    if not starts:
        return []
    blocks: List[Tuple[int, List[str]]] = []
    for idx, (start_i, art_no) in enumerate(starts):
        end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        block_lines = [l for l in lines[start_i:end_i] if l.strip() != ""]
        block_lines = clean_article_lines(block_lines)
        if block_lines:
            blocks.append((art_no, block_lines))
    return blocks

def export_articles_jsonl_from_pdf(pdf_path: str, out_jsonl: str) -> List[Dict]:
    raw = pdf_to_text(pdf_path)
    cleaned = clean_text(raw)
    blocks = split_into_article_blocks(cleaned)
    if not blocks:
        raise RuntimeError("Cannot split articles from PDF (no '第X條' markers).")
    articles = []
    for art_no, lines in blocks:
        title = f"第 {art_no} 條"
        content_lines = lines[:]
        if content_lines:
            first = content_lines[0].strip()
            if ARTICLE_RE.search(first) or ARTICLE_RE_ALT.match(first):
                content_lines = content_lines[1:]
        content = "\n".join(content_lines).strip()
        articles.append({"num": int(art_no), "title": title, "content": content})
    articles.sort(key=lambda x: x["num"])
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return articles

def load_articles_jsonl(path: str) -> List[Dict]:
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            num = int(obj["num"])
            title = str(obj.get("title", f"第 {num} 條")).strip()
            content = str(obj.get("content", "")).strip()
            text = f"{title}\n{content}".strip()
            metas.append({"id": f"law::art{num}", "article_no": num, "text": text})
    metas.sort(key=lambda x: x["article_no"])
    return metas
