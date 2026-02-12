# everbot_rag/pipeline.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from .config import FAIL_MSG, DEFAULT_TOPK, LLM_MODEL_DEFAULT, EMB_MODEL_DEFAULT
from .retriever import normalize_to_zh_tw, retrieve_related_articles, enough_evidence
from .reranker import llm_rerank_articles
from .qa import answer_with_evidence, translate_to_english

def ask(
    query_raw: str,
    topk: int = DEFAULT_TOPK,
    base_dir: str = ".",
    emb_model: str = EMB_MODEL_DEFAULT,
    llm_model: str = LLM_MODEL_DEFAULT,
    enable_rerank: bool = True,
) -> Optional[Dict]:
    query_raw = (query_raw or "").strip()
    if not query_raw:
        return None

    query_zh = normalize_to_zh_tw(query_raw, llm_model=llm_model)
    supports, expanded_query = retrieve_related_articles(
        query_zh, top_n=topk, base_dir=base_dir, emb_model=emb_model, llm_model=llm_model
    )

    if not supports or not enough_evidence(supports, expanded_query):
        return None

    reranked = llm_rerank_articles(expanded_query, supports, llm_model=llm_model, enabled=enable_rerank)
    result = answer_with_evidence(expanded_query, reranked, llm_model=llm_model)
    if not result:
        return None

    # attach evidence translations for display
    evidences = []
    for s in reranked:
        evidences.append({
            "article_no": int(s["article_no"]),
            "article_zh": s["article_zh"],
            "hit_count": s.get("hit_count", 0),
            "lex_score": s.get("lex_score", 0.0),
            "vec_score": s.get("vec_score", 0.0),
            "text_zh": s["text"],
        })

    return {
        "query_zh": query_zh,
        "expanded_query_zh": expanded_query,
        "answer_zh": result["answer_zh"],
        "answer_en": result["answer_en"],
        "supports": evidences,
    }
