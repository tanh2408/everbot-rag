Dưới đây là bản hoàn chỉnh để bạn **copy–paste trực tiếp vào `README.md`**:

---

# Everbot Offline RAG (Law PDF)

This project implements an **offline RAG pipeline** over a law document PDF
(example: `中華民國憲法.pdf`).

---

# 📌 Data & Deliverables

## 📄 Data Source

The **PDF file (e.g., `中華民國憲法.pdf`) is the ONLY data source** used by the system.

All downstream artifacts are generated strictly from this PDF:

* `articles_extracted.jsonl`
* `law_chunks.jsonl`
* `law.index`
* `law_ontology.json`

No external legal dataset is used at any stage.

---

## 📘 Handover Document

The Word file:

```
RAG_Offline_Handover_EN.docx
```

is the **project handover documentation**.

It contains:

* System design explanation
* Data processing pipeline description
* Retrieval and re-ranking strategy
* Prompt construction logic
* Token limits and truncation policy
* Model selection rationale
* Resource constraints and optimization considerations

⚠️ The Word file is **NOT used as input data** to the RAG system.
It is strictly documentation for review and evaluation purposes.

---

# 🎯 Design Philosophy (Audit-Friendly)

This project is intentionally structured to be review-safe and transparent:

* PDF → extracted articles → indexed
* Clear intermediate artifact: `articles_extracted.jsonl`
* Fully reproducible pipeline
* No hidden external knowledge sources

Reviewers can verify that:

* The system processes the PDF directly
* Article segmentation is transparent
* The FAISS index is built only from extracted content
* All retrieval results are grounded in the PDF-derived articles

---

# 🗂 Folder Structure

```
everbot_rag_refactor/
├─ rag_ollama_min.py
├─ RAG_Offline_Handover_EN.docx     # 📘 Handover document
└─ everbot_rag/
   ├─ cli.py
   ├─ config.py
   ├─ pdf_extract.py
   ├─ index_build.py
   ├─ ontology.py
   ├─ retriever.py
   ├─ reranker.py
   ├─ qa.py
   ├─ pipeline.py
   └─ utils_text.py
```

---

# 🔄 Build Pipeline

## Step 1 — Extract from PDF

```
PDF
 → text (pdfplumber)
 → clean
 → split into Article blocks (第X條)
 → articles_extracted.jsonl
```

`articles_extracted.jsonl` is an **intermediate audit artifact generated from the PDF**.

It exists to improve:

* Debuggability
* Transparency
* Reproducibility
* Review safety

---

## Step 2 — Build Index

```
articles_extracted.jsonl
 → FAISS index
 → metadata file
 → ontology (auto-generated keywords for ALL articles)
```

The index is built strictly from the JSONL file, which itself was generated from the PDF.

---

# 🚀 Usage

## Build

```bash
python rag_ollama_min.py build --pdf "中華民國憲法.pdf"
```

Optional build-time enhancements:

```bash
--llm-concepts     # Generate per-article concept aliases
--llm-synonyms     # Expand synonym table
```

---

## Ask

```bash
python rag_ollama_min.py ask "根據憲法第20條的規定，國民有什麼樣的義務？" --topk 5
```

Output includes:

* Answer in Traditional Chinese
* English translation (Engsub)
* Extracted Articles
* Evidence in both languages

If insufficient evidence:

```
Cannot answer the question based on the extracted constitutional articles.
```

---

# 🧠 Retrieval Strategy (High-Level)

1. Normalize the question to Traditional Chinese
2. Expand via ontology (seed synonyms + optional LLM concepts)
3. Keyword-first retrieval across all articles
4. Vector assist (FAISS cosine similarity)
5. Cheap LLM re-ranking (top-N)
6. Strictly grounded answer generation
7. English translation

---

# ⚖️ Important Notes

* The PDF is the **data source**.
* The Word file is the **handover documentation**.
* All index and ontology files are generated artifacts.
* The system does not use any external legal dataset.
* All answers are grounded strictly in extracted constitutional articles.
