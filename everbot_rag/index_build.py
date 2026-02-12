# everbot_rag/index_build.py
from __future__ import annotations
import json
import os
from typing import List, Dict, Optional

import faiss
import numpy as np

from .config import (
    ARTICLES_JSONL, INDEX_PATH, META_PATH, ONTOLOGY_PATH,
    EMB_MODEL_DEFAULT, LLM_MODEL_DEFAULT, PER_ARTICLE_TERM_CAP
)
from .pdf_extract import export_articles_jsonl_from_pdf, load_articles_jsonl
from .embeddings import embed_passages
from .ontology import build_ontology

def build_all(
    pdf_path: str,
    out_dir: str = ".",
    emb_model: str = EMB_MODEL_DEFAULT,
    llm_model: str = LLM_MODEL_DEFAULT,
    use_llm_concepts: bool = False,
    use_llm_synonyms: bool = False,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    articles_jsonl = os.path.join(out_dir, ARTICLES_JSONL)
    index_path = os.path.join(out_dir, INDEX_PATH)
    meta_path = os.path.join(out_dir, META_PATH)
    ontology_path = os.path.join(out_dir, ONTOLOGY_PATH)

    # 1) PDF -> text -> clean -> article blocks -> JSONL (audit artifact)
    export_articles_jsonl_from_pdf(pdf_path, articles_jsonl)

    # 2) JSONL -> metas for indexing
    metas = load_articles_jsonl(articles_jsonl)

    passages = ["passage: " + m["text"] for m in metas]
    embs = embed_passages(emb_model, passages)
    dim = int(embs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embs, dtype="float32"))

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 3) Build ontology (keywords for ALL articles + optional LLM aliases/synonyms)
    build_ontology(
        metas=metas,
        out_path=ontology_path,
        llm_model=llm_model,
        per_article_term_cap=PER_ARTICLE_TERM_CAP,
        use_llm_concepts=use_llm_concepts,
        use_llm_synonyms=use_llm_synonyms,
    )

    return {
        "articles_jsonl": articles_jsonl,
        "index_path": index_path,
        "meta_path": meta_path,
        "ontology_path": ontology_path,
    }
