# Everbot Offline RAG (Law PDF) — Refactored Package

This package implements an **offline RAG pipeline** for a law PDF (example: `中華民國憲法.pdf`).
It is designed to be **audit-friendly** for reviews:

- **Still uses the PDF as the only source of truth**
- Extracts **Article blocks** (`第X條`) from the PDF
- Writes an intermediate artifact **`articles_extracted.jsonl`** (derived from PDF) for debugging/auditing
- Builds FAISS index **from that JSONL** (not from any external dataset)

---

## 1) Folder Structure

```
everbot_rag_refactor/
├─ rag_ollama_min.py                 # thin CLI wrapper
└─ everbot_rag/
   ├─ cli.py                         # CLI entry (build / ask / batch)
   ├─ config.py                      # parameters
   ├─ pdf_extract.py                 # PDF -> text -> clean -> article blocks -> articles_extracted.jsonl
   ├─ index_build.py                 # JSONL -> FAISS index + meta + ontology
   ├─ ontology.py                    # auto keywords for all articles + synonym map + optional LLM concepts
   ├─ retriever.py                   # keyword-first retrieval + vector assist
   ├─ reranker.py                    # cheap LLM rerank for top-N candidates
   ├─ qa.py                          # answer (Chinese) + Engsub translation
   ├─ pipeline.py                    # glue: normalize -> retrieve -> rerank -> answer
   └─ utils_text.py                  # cleaning, regex, helpers
```

---

## 2) Install / Environment

Python 3.10+ recommended.

Required packages:
- `pdfplumber`
- `faiss-cpu`
- `sentence-transformers`
- `ollama`
- `numpy`

**Offline notes**
- Pull Ollama model once before going offline (example):
  - `ollama pull qwen2.5:7b-instruct`
- Download the embedding model once (HuggingFace). For offline runs, set:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

---

## 3) Build (PDF → JSONL → Index)

Put your PDF next to the script (or provide `--pdf`).

```bash
python rag_ollama_min.py build --pdf "中華民國憲法.pdf"
```

Outputs in the current folder (or `--out-dir`):
- `articles_extracted.jsonl`  ✅ audit artifact generated from the PDF
- `law.index`                 (FAISS)
- `law_chunks.jsonl`          (meta: id, article_no, text)
- `law_ontology.json`         (auto keywords for ALL articles + synonym map)

Optional (build-time, slow):
- `--llm-concepts` : generate per-article “how users may ask” aliases
- `--llm-synonyms` : expand synonym table

---

## 4) Ask

```bash
python rag_ollama_min.py ask "根據憲法第20條的規定，國民有什麼樣的義務？" --topk 5
```

Output includes:
- Answer in **Traditional Chinese**
- `Engsub` (English translation)
- Extracted articles + evidence in both languages

If evidence is insufficient, it prints **only**:
```
Cannot answer the question based on the extracted constitutional articles.
```

---

## 5) How retrieval works (high-level)

1. Normalize the question into **Traditional Chinese** using Ollama.
2. Expand query terms using ontology:
   - seed synonym map (handcrafted)
   - optional per-article aliases (if you enabled `--llm-concepts`)
3. Retrieve candidate articles:
   - **keyword-first** scoring against:
     - raw article text
     - per-article keyword index (auto-built for all articles)
     - per-article aliases (optional)
   - + **vector assist** (FAISS cosine / IndexFlatIP)
4. **Cheap LLM re-rank** the top candidates (optional)
5. Answer using only evidence (strict grounding), then translate to English.

---

## 6) Troubleshooting

- If no articles are extracted: your PDF may be scanned or lacks `第X條` markers.
- If answers are often "Cannot answer...":
  - increase `--topk` (e.g., 8 or 12)
  - enable build-time aliases: `--llm-concepts`
  - extend `SEED_SYNONYMS` in `ontology.py`

---
