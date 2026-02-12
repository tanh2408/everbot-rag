# everbot_rag/qa.py
from __future__ import annotations
import re
from typing import List, Dict, Optional
import ollama

from .config import FAIL_MSG, MAX_EVIDENCE_CHARS, LLM_MODEL_DEFAULT

def translate_to_english(text_zh: str, llm_model: str = LLM_MODEL_DEFAULT) -> str:
    text_zh = (text_zh or "").strip()
    if not text_zh:
        return "Translation error."
    prompt = (
        "Translate the following Traditional Chinese legal text into English.\n"
        "Output ENGLISH ONLY. No Chinese. No explanations.\n\n"
        f"{text_zh}"
    )
    r = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a precise translator. Return English only."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )
    out = (r.get("message", {}) or {}).get("content", "").strip()
    if re.search(r"[\u4e00-\u9fff]", out) or out.strip() == "":
        return "Translation error."
    return out

def build_evidence_text(supports: List[Dict]) -> str:
    blocks = []
    for i, s in enumerate(supports, 1):
        blocks.append(f"【證據{i}｜{s['article_zh']}】\n{s['text']}")
    ev = "\n\n".join(blocks).strip()
    if len(ev) > MAX_EVIDENCE_CHARS:
        ev = ev[:MAX_EVIDENCE_CHARS].rstrip() + "…"
    return ev

def answer_with_evidence(query_zh: str, supports: List[Dict], llm_model: str = LLM_MODEL_DEFAULT) -> Optional[Dict[str,str]]:
    evidence_text = build_evidence_text(supports)

    system_zh = (
        "你是法律文件問答助手。\n"
        "你只能根據【證據】回答，不得使用外部資料。\n"
        "若【證據】不足以回答問題，請只輸出一句：證據不足，無法回答。\n"
        "若可回答：請先概括總結（避免逐字照抄），再回答問題。\n"
        "輸出格式：1~3 句 或 2~5 點條列。\n"
        "不得編造證據中不存在的數字、機關、程序或細節。\n"
        "回答必須是繁體中文。\n"
    )

    user_zh = f"""【證據】
{evidence_text}

問題：{query_zh}
"""

    r = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_zh},
            {"role": "user", "content": user_zh},
        ],
        options={"temperature": 0.0},
    )
    ans_zh = (r.get("message", {}) or {}).get("content", "").strip()
    if not ans_zh:
        return None
    if "證據不足" in ans_zh or "無法回答" in ans_zh:
        return None

    ans_en = translate_to_english(ans_zh, llm_model=llm_model)
    if ans_en == "Translation error.":
        return None

    return {"answer_zh": ans_zh, "answer_en": ans_en}
