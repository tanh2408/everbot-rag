# everbot_rag/reranker.py
from __future__ import annotations
import json
import re
from typing import List, Dict, Tuple, Optional
import ollama

from .config import (
    ENABLE_LLM_RERANK, RERANK_CAND_N, RERANK_KEEP_TOP, RERANK_MODEL_TEMPERATURE,
    LLM_MODEL_DEFAULT
)

def _safe_json_array(text: str) -> Optional[List]:
    text = (text or "").strip()
    if not text:
        return None
    # try direct
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # try extract first [...] block
    m = re.search(r"(\[[\s\S]*\])", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list):
                return obj
        except Exception:
            return None
    return None

def llm_rerank_articles(
    query_zh: str,
    supports: List[Dict],
    llm_model: str = LLM_MODEL_DEFAULT,
    cand_n: int = RERANK_CAND_N,
    keep_top: int = RERANK_KEEP_TOP,
    enabled: bool = ENABLE_LLM_RERANK,
) -> List[Dict]:
    if not enabled or not supports:
        return supports

    cands = supports[:max(1, min(cand_n, len(supports)))]
    items = []
    for i, s in enumerate(cands, 1):
        # keep the first ~180 chars for rerank
        t = (s.get("text","") or "").strip().replace("\n"," ")
        t = re.sub(r"\s+", " ", t)
        t_short = t[:180] + ("…" if len(t) > 180 else "")
        items.append(f"{i}. {s.get('article_zh')}｜{t_short}")

    prompt = (
        "You are reranking constitutional articles for a user question.\n"
        "Return the best order of items by relevance.\n"
        "Rules: return JSON array of item numbers only, e.g. [3,1,2]. No explanations.\n\n"
        f"Question: {query_zh}\n\n"
        "Candidates:\n" + "\n".join(items)
    )

    r = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": "Return STRICT JSON only."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": float(RERANK_MODEL_TEMPERATURE)},
    )
    out = (r.get("message", {}) or {}).get("content", "")
    arr = _safe_json_array(out)
    if not arr:
        return supports

    order = []
    used = set()
    for x in arr:
        try:
            k = int(x)
        except Exception:
            continue
        if 1 <= k <= len(cands) and k not in used:
            used.add(k)
            order.append(cands[k-1])

    # append anything not mentioned
    for i, s in enumerate(cands, 1):
        if i not in used:
            order.append(s)

    reranked = order + supports[len(cands):]
    return reranked[:max(1, keep_top)] + reranked[max(1, keep_top):]
