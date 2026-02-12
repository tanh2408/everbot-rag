# everbot_rag/ontology.py
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional
import ollama

from .utils_text import cjk_only

GENERIC_WORDS = {
    "人民","國民","公民","國家","政府","本憲法","憲法","法律","法",
    "權利","義務","自由","保障","限制","規定","依","依法","依法律",
    "除","但","及","或","與","並","為","得","不得","應","不得以",
    "以上","前項","後項","本條","各條","左列","右列","之一","之二",
    "其","所","者","時","內","外","因","由","於","對於","關於",
    "一","二","三","四","五","六","七","八","九","十"
}

STOP_BIGRAMS = {
    "本憲","憲法","法律","人民","國民","公民","條文","規定",
    "之規","之定","之義","之權","之務","之內","之國","之人","之所","之者",
    "前項","後項","本條","各條","左列","右列","以上","之一","之二","之三",
    "不得","應予","依照","依本","依其","其所","其為"
}

SEED_SYNONYMS: Dict[str, List[str]] = {
    "公民": ["人民", "國民"],
    "國民": ["人民"],
    "人民": ["國民"],
    "權利": ["自由", "權益", "受保障"],
    "義務": ["責任", "應盡義務"],
    "自由": ["權利", "不受限制"],

    "居住": ["住居", "住所", "居留"],
    "遷徙": ["移動", "旅行", "通行"],
    "居住及遷徙自由": ["居住自由", "遷徙自由", "選擇居住地", "自由移動"],

    "言論": ["表達", "發言"],
    "出版": ["發表", "著作"],
    "宗教": ["信仰", "信教"],
    "集會": ["會議", "聚會"],
    "結社": ["組織", "社團"],

    "納稅": ["繳稅", "稅捐"],
    "服兵役": ["兵役", "軍役", "從軍"],
    "國民教育": ["義務教育", "受教育"],

    "開宗明義": ["三民主義", "民主共和國", "民有民治民享"],
    "治理理念": ["三民主義", "民主共和國", "民有民治民享"],
    "建國理念": ["三民主義", "民主共和國", "民有民治民享"],
    "立國精神": ["三民主義", "民主共和國", "民有民治民享"],

    "主權": ["主權屬於國民全體", "國民主權", "主權在民", "政權"],
    "國家主權": ["主權屬於國民全體", "國民主權", "主權", "政權"],
    "國民主權": ["主權屬於國民全體", "主權在民", "主權", "政權"],
    "主權在民": ["主權屬於國民全體", "國民主權", "主權"],
    "參與": ["參政", "行使", "選舉", "罷免", "創制", "複決"],
    "參與國家主權": ["國民主權", "主權在民", "主權屬於國民全體", "政權", "參政權"],
    "參政": ["選舉", "罷免", "創制", "複決", "政權"],
    "參政權": ["選舉", "罷免", "創制", "複決", "政權"],

    "权利": ["權利"],
    "选择": ["選擇"],
    "移动": ["移動", "遷徙"],
}

def extract_article_keywords(article_text: str, top_k: int = 60) -> List[str]:
    body = cjk_only(article_text)
    if not body:
        return []
    freq: Dict[str, int] = {}

    def add_ngram(n: int):
        for i in range(0, len(body) - n + 1):
            g = body[i:i+n]
            if n == 2 and g in STOP_BIGRAMS:
                continue
            if g in GENERIC_WORDS:
                continue
            freq[g] = freq.get(g, 0) + 1

    add_ngram(2)
    add_ngram(3)

    items = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    out: List[str] = []
    for g, c in items:
        if c <= 1 and len(out) >= 20:
            break
        out.append(g)
        if len(out) >= top_k:
            break
    return out

def _llm_make_article_concepts(llm_model: str, article_no: int, article_text: str) -> List[str]:
    prompt = (
        "You are building question-like concept aliases for a constitutional article (Traditional Chinese).\n"
        "Given the article text, output 6~12 short Traditional Chinese phrases (2~10 chars), representing\n"
        "the core concept and how users might ask about it.\n"
        "Rules:\n"
        "1) Output JSON array only, e.g. [\"A\",\"B\",...]\n"
        "2) No explanation\n"
        "3) Avoid overly generic words like 「權利」「自由」 alone\n"
        f"Article: 第{article_no}條\n"
        f"Text:\n{article_text}\n"
    )
    r = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": "Return STRICT JSON only."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )
    txt = (r.get("message", {}) or {}).get("content", "").strip()
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            out = []
            seen = set()
            for x in arr:
                if not isinstance(x, str):
                    continue
                x = x.strip()
                if not x or len(x) == 1:
                    continue
                if x in GENERIC_WORDS:
                    continue
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out[:24]
    except Exception:
        return []
    return []

def build_ontology(
    metas: List[Dict],
    out_path: str,
    llm_model: str,
    per_article_term_cap: int,
    use_llm_concepts: bool = False,
    use_llm_synonyms: bool = False,
) -> Dict:
    per_article: Dict[str, List[str]] = {}
    per_aliases: Dict[str, List[str]] = {}

    for m in metas:
        art_no = int(m["article_no"])
        text = m["text"]
        per_article[str(art_no)] = extract_article_keywords(text, top_k=per_article_term_cap)

        if use_llm_concepts:
            per_aliases[str(art_no)] = _llm_make_article_concepts(llm_model, art_no, text)

    synonym_map: Dict[str, List[str]] = dict(SEED_SYNONYMS)

    # Optional global synonym expansion (kept minimal; OFF by default)
    if use_llm_synonyms:
        global_terms: List[str] = []
        for kws in per_article.values():
            global_terms.extend(kws[:8])
        seen = set()
        global_terms_u = []
        for t in global_terms:
            if t not in seen and len(t) >= 2:
                seen.add(t)
                global_terms_u.append(t)
        global_terms_u = global_terms_u[:60]

        prompt = (
            "Build a synonym/near-synonym table for constitutional terms (Traditional Chinese).\n"
            "For each term, output 3~6 related expressions. Output JSON only.\n"
            "Format: {\"term\":[\"syn1\",\"syn2\",...], ...}\n"
            f"Terms: {', '.join(global_terms_u)}"
        )

        r = ollama.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": "Return STRICT JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0},
        )
        txt = (r.get("message", {}) or {}).get("content", "").strip()
        try:
            llm_map = json.loads(txt)
            if isinstance(llm_map, dict):
                for k, vs in llm_map.items():
                    if not isinstance(vs, list):
                        continue
                    synonym_map.setdefault(k, [])
                    for v in vs:
                        if isinstance(v, str) and v and v != k and v not in synonym_map[k]:
                            synonym_map[k].append(v)
        except Exception:
            pass

    onto = {
        "version": 1,
        "per_article_keywords": per_article,
        "per_article_aliases": per_aliases,
        "synonym_map": synonym_map,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(onto, f, ensure_ascii=False, indent=2)
    return onto

def load_ontology(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = {"per_article_keywords": {}, "per_article_aliases": {}, "synonym_map": dict(SEED_SYNONYMS)}

    obj.setdefault("synonym_map", {})
    for k, vs in SEED_SYNONYMS.items():
        obj["synonym_map"].setdefault(k, [])
        for v in vs:
            if v not in obj["synonym_map"][k]:
                obj["synonym_map"][k].append(v)

    obj.setdefault("per_article_keywords", {})
    obj.setdefault("per_article_aliases", {})
    return obj
