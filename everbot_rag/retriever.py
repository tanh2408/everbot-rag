# everbot_rag/retriever.py
from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
import ollama

from .config import (
    INDEX_PATH, META_PATH, ONTOLOGY_PATH,
    EMB_MODEL_DEFAULT, LLM_MODEL_DEFAULT,
    USE_VECTOR_ASSIST, VECTOR_WEIGHT, LEX_WEIGHT, KEYIDX_WEIGHT, ALIAS_WEIGHT,
    MIN_LEX_SCORE, MIN_TOTAL_HITS, VEC_KEEP_TH,
    EXPLICIT_ARTICLE_BOOST, EXPLICIT_ARTICLE_HARD_FIRST,
)
from .utils_text import has_cjk, cjk_only, ARTICLE_RE, ARTICLE_RE_ALT
from .embeddings import embed_query
from .ontology import load_ontology

_G_INDEX = None
_G_METAS = None
_G_ONTO = None

def load_index_and_meta(base_dir: str = "."):
    global _G_INDEX, _G_METAS
    if _G_INDEX is not None and _G_METAS is not None:
        return _G_INDEX, _G_METAS

    index_path = os.path.join(base_dir, INDEX_PATH)
    meta_path = os.path.join(base_dir, META_PATH)
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index not found. Run build first.")

    _G_INDEX = faiss.read_index(index_path)
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    _G_METAS = metas
    return _G_INDEX, _G_METAS

def load_onto(base_dir: str = "."):
    global _G_ONTO
    if _G_ONTO is not None:
        return _G_ONTO
    onto_path = os.path.join(base_dir, ONTOLOGY_PATH)
    _G_ONTO = load_ontology(onto_path)
    return _G_ONTO

def normalize_to_zh_tw(query_raw: str, llm_model: str = LLM_MODEL_DEFAULT) -> str:
    q = (query_raw or "").strip()
    if not q:
        return q
    prompt = (
        "請將以下文字改寫/轉換為「繁體中文」，保留原意。\n"
        "規則：\n"
        "1) 只輸出繁體中文\n"
        "2) 不要解釋\n"
        "3) 盡量使用法律常見用語\n\n"
        f"文字：{q}"
    )
    r = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": "你是精準的法律語言轉換助手。"},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )
    out = (r.get("message", {}) or {}).get("content", "").strip()
    return out if has_cjk(out) else q

def extract_article_numbers(query_zh: str) -> List[int]:
    nums = set()
    for m in re.finditer(r"第\s*(\d+)\s*條", query_zh):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"Article\s*(\d+)", query_zh, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"Điều\s*(\d+)", query_zh, flags=re.IGNORECASE):
        nums.add(int(m.group(1)))
    return sorted(nums)

def extract_keywords(query_zh: str, n_min: int = 2, n_max: int = 4) -> List[str]:
    q_cjk = cjk_only(query_zh)
    if not q_cjk:
        return []
    kws: List[str] = []

    # n-grams
    L = len(q_cjk)
    for n in range(n_min, n_max + 1):
        for i in range(0, L - n + 1):
            g = q_cjk[i:i+n]
            kws.append(g)

    seen = set()
    out = []
    for k in kws:
        k = k.strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

def expand_query_with_ontology(query_zh: str, base_dir: str = ".") -> str:
    onto = load_onto(base_dir)
    syn = onto.get("synonym_map", {}) or {}
    extra_terms = []

    for k, vs in syn.items():
        if k and k in query_zh:
            extra_terms.extend(vs)

    # per-article aliases are used for scoring, but some are helpful as query expansion too
    # (only add if query already hints the concept via substring match)
    if re.search(r"(主權|政權|國民全體|三民主義|民有民治民享|民主共和國)", query_zh):
        extra_terms.extend(["主權屬於國民全體", "國民主權", "主權在民"])
    if re.search(r"(居住|住居|住所|移動|旅行|遷徙)", query_zh):
        extra_terms.extend(["居住及遷徙自由", "遷徙自由"])

    extra_terms_u = []
    seen = set()
    for t in extra_terms:
        if not t or t in query_zh:
            continue
        if t not in seen:
            seen.add(t)
            extra_terms_u.append(t)

    return (query_zh + " " + " ".join(extra_terms_u)).strip() if extra_terms_u else query_zh

def vector_scores(query_zh: str, emb_model: str, base_dir: str, top_k: int) -> Dict[int, float]:
    index, metas = load_index_and_meta(base_dir)
    q_emb = embed_query(emb_model, "query: " + query_zh)
    scores, ids = index.search(np.asarray(q_emb, dtype="float32"), top_k)
    out: Dict[int, float] = {}
    for sc, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        art_no = metas[int(i)].get("article_no")
        if art_no is not None:
            out[int(art_no)] = float(sc)
    return out

def lexical_score(text: str, terms: List[str]) -> Tuple[int, int]:
    hit = 0
    score = 0
    for t in terms:
        if t and t in text:
            hit += 1
            score += 3 if len(t) >= 3 else 1
    return hit, score

def keyword_index_score(article_no: int, terms: List[str], base_dir: str) -> Tuple[int, int]:
    onto = load_onto(base_dir)
    per = onto.get("per_article_keywords", {}) or {}
    kws = per.get(str(article_no), []) or []
    s = 0
    hit = 0
    kws_set = set(kws)
    for t in terms:
        if t in kws_set:
            hit += 1
            s += 4 if len(t) >= 3 else 1
    return hit, s

def alias_score(article_no: int, query_zh: str, base_dir: str) -> Tuple[int, int]:
    onto = load_onto(base_dir)
    per = onto.get("per_article_aliases", {}) or {}
    als = per.get(str(article_no), []) or []
    hit = 0
    score = 0
    for a in als:
        if a and a in query_zh:
            hit += 1
            score += 6 if len(a) >= 3 else 2
    return hit, score

def retrieve_related_articles(
    query_zh: str,
    top_n: int,
    base_dir: str = ".",
    emb_model: str = EMB_MODEL_DEFAULT,
    llm_model: str = LLM_MODEL_DEFAULT,
) -> Tuple[List[Dict], str]:
    _, metas = load_index_and_meta(base_dir)

    explicit = extract_article_numbers(query_zh)

    expanded_query = expand_query_with_ontology(query_zh, base_dir=base_dir)
    terms = extract_keywords(expanded_query)

    vmap = vector_scores(expanded_query, emb_model, base_dir, top_k=max(60, top_n*4)) if USE_VECTOR_ASSIST else {}

    scored = []
    for m in metas:
        art_no = int(m["article_no"])
        text = m["text"]

        hit1, lex1 = lexical_score(text, terms)
        hit2, lex2 = keyword_index_score(art_no, terms, base_dir)
        hit3, lex3 = alias_score(art_no, expanded_query, base_dir)

        lex_total = (lex1 * 1.0) + (KEYIDX_WEIGHT * lex2) + (ALIAS_WEIGHT * lex3)
        hit_total = hit1 + hit2 + hit3

        final = LEX_WEIGHT * float(lex_total)
        if art_no in vmap:
            final += VECTOR_WEIGHT * float(vmap[art_no])

        if explicit and art_no in explicit:
            final += EXPLICIT_ARTICLE_BOOST

        scored.append({
            "article_no": art_no,
            "article_zh": f"第{art_no}條",
            "id": m.get("id",""),
            "text": text,
            "hit_count": int(hit_total),
            "lex_score": float(lex_total),
            "vec_score": float(vmap.get(art_no, 0.0)),
            "final_score": float(final),
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    supports: List[Dict] = []
    seen = set()

    if explicit:
        for a in explicit:
            for it in scored:
                if it["article_no"] == a and it["article_zh"] not in seen:
                    supports.append(it)
                    seen.add(it["article_zh"])
                    break

    for it in scored:
        if len(supports) >= top_n:
            break
        if it["article_zh"] in seen:
            continue
        if it["lex_score"] < MIN_LEX_SCORE and it["vec_score"] < VEC_KEEP_TH and (it["article_no"] not in explicit):
            continue
        supports.append(it)
        seen.add(it["article_zh"])

    if EXPLICIT_ARTICLE_HARD_FIRST and explicit:
        supports.sort(key=lambda x: (0 if x["article_no"] in explicit else 1, -x["final_score"]))
    else:
        supports.sort(key=lambda x: -x["final_score"])

    return supports, expanded_query

def enough_evidence(supports: List[Dict], query_zh: str) -> bool:
    explicit = extract_article_numbers(query_zh)
    if explicit:
        found = set(int(s["article_no"]) for s in supports)
        return all(a in found for a in explicit)
    total_hits = sum(int(s.get("hit_count", 0)) for s in supports)
    return total_hits >= MIN_TOTAL_HITS
