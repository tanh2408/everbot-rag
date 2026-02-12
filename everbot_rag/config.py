# everbot_rag/config.py
from __future__ import annotations

# ===== Default models =====
LAW_PDF_DEFAULT = "中華民國憲法.pdf"
EMB_MODEL_DEFAULT = "intfloat/multilingual-e5-base"
LLM_MODEL_DEFAULT = "qwen2.5:7b-instruct"

# ===== Artifacts =====
ARTICLES_JSONL = "articles_extracted.jsonl"   # audit/debug artifact derived from PDF
INDEX_PATH = "law.index"
META_PATH = "law_chunks.jsonl"
ONTOLOGY_PATH = "law_ontology.json"

# ===== Output policy =====
FAIL_MSG = "Cannot answer the question based on the extracted constitutional articles."

# ===== Retrieval params =====
DEFAULT_TOPK = 8
USE_VECTOR_ASSIST = True
VECTOR_WEIGHT = 0.35

LEX_WEIGHT = 1.00
KEYIDX_WEIGHT = 2.0
ALIAS_WEIGHT = 2.0

MIN_LEX_SCORE = 1
MIN_TOTAL_HITS = 1
VEC_KEEP_TH = 0.72

# Explicit article routing
EXPLICIT_ARTICLE_BOOST = 1000.0
EXPLICIT_ARTICLE_HARD_FIRST = True

# Evidence truncation
MAX_EVIDENCE_CHARS = 9000

# ===== LLM cheap rerank =====
ENABLE_LLM_RERANK = True
RERANK_CAND_N = 14
RERANK_KEEP_TOP = 5
RERANK_MODEL_TEMPERATURE = 0.0

# ===== Ontology build =====
PER_ARTICLE_TERM_CAP = 30
