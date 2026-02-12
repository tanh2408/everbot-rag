# rag_ollama_min.py
# RAG (Offline) for 中華民國憲法.pdf — REVIEW-SAFE PIPELINE + LLM CHEAP RERANK
#
# ✅ Build pipeline:
#   PDF -> text (pdfplumber) -> clean
#       -> split Article blocks (第X條)
#       -> export articles_extracted.jsonl  (audit/debug artifact)
#       -> build FAISS index + meta + ontology from articles_extracted.jsonl
#
# ✅ Ask:
#   - Normalize query to Traditional Chinese (繁體中文)
#   - Parse explicit article references: 第20條 / 第二十條 / 第一百二十條...
#   - Expand query via:
#       (1) seed synonyms + rule-based expansion
#       (2) build-time LLM concept aliases for ALL articles (when --llm_concepts)
#   - Retrieve top candidates via (keyword-index + alias overlap + text lex + vector assist)
#   - ✅ Cheap LLM rerank topN: pick most relevant articles & judge sufficiency
#   - Answer in Chinese (Traditional) + Engsub
#   - If insufficient evidence => print FAIL_MSG only
#
# Commands:
#   Build:
#     python rag_ollama_min.py build [--llm_concepts] [--llm_syn]
#   Ask:
#     python rag_ollama_min.py ask "國民是否有參與國家主權的權利" --topk 3

import os
import re
import json
import sys
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
import ollama

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# -------------------------
# Paths / Models
# -------------------------
LAW_PDF = "中華民國憲法.pdf"
ARTICLES_JSONL = "articles_extracted.jsonl"  # artifact generated from PDF

EMB_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = "qwen2.5:7b-instruct"

INDEX_PATH = "law.index"
META_PATH = "law_chunks.jsonl"       # per-article meta lines
ONTOLOGY_PATH = "law_ontology.json"  # per-article keywords + synonym map + concept aliases

# If insufficient evidence, output ONLY this string (nothing else)
FAIL_MSG = "Cannot answer the question based on the extracted constitutional articles."

# -------------------------
# Retrieval / Ranking
# -------------------------
DEFAULT_TOPK = 8

# Evidence thresholds (relaxed for conceptual questions)
MIN_LEX_SCORE = 1
MIN_TOTAL_HITS = 1

# Hybrid scoring weights
USE_VECTOR_ASSIST = True
VECTOR_WEIGHT = 0.35
LEX_WEIGHT = 1.00
ALIAS_WEIGHT = 2.0
KEYIDX_WEIGHT = 2.0

# Keep candidates even if lexical weak but vector strong
VEC_KEEP_TH = 0.72

# Explicit article must rank #1
EXPLICIT_ARTICLE_BOOST = 1000.0
EXPLICIT_ARTICLE_HARD_FIRST = True

# Truncate evidence passed to LLM
MAX_EVIDENCE_CHARS = 9000

# -------------------------
# Cheap LLM Re-rank
# -------------------------
ENABLE_LLM_RERANK = True
RERANK_CAND_N = 14
RERANK_KEEP_TOP = 5
RERANK_MODEL_TEMPERATURE = 0.0

# -------------------------
# Regex / stopwords / synonyms
# -------------------------
ARTICLE_RE = re.compile(r"第\s*(\d+)\s*條")
ARTICLE_RE_ALT = re.compile(r"^\s*(\d+)\s*條\s*$")

# Chapter/Section markers sometimes leak into content
CHAPTER_SECTION_RE = re.compile(
    r"^\s*第\s*[一二三四五六七八九十百千零〇兩]+\s*(章|節|編|款|目|項)\s*.*$"
)

GENERIC_WORDS = {
    "人民", "國民", "公民", "國家", "政府", "本憲法", "憲法", "法律", "法",
    "權利", "義務", "自由", "保障", "限制", "規定", "依", "依法", "依法律",
    "除", "但", "及", "或", "與", "並", "為", "得", "不得", "應", "不得以",
    "以上", "前項", "後項", "本條", "各條", "左列", "右列", "之一", "之二",
    "其", "所", "者", "時", "內", "外", "因", "由", "於", "對於", "關於",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"
}

STOP_BIGRAMS = {
    "本憲", "憲法", "法律", "人民", "國民", "公民", "條文", "規定",
    "之規", "之定", "之義", "之權", "之務", "之內", "之國", "之人", "之所", "之者",
    "前項", "後項", "本條", "各條", "左列", "右列", "以上", "之一", "之二", "之三",
    "不得", "應予", "依照", "依本", "依其", "其所", "其為"
}

# Seed synonyms: keep small but systematic + add sovereignty/political power
SEED_SYNONYMS = {
    # People naming
    "公民": ["人民", "國民"],
    "國民": ["人民"],
    "人民": ["國民"],

    # Rights / duties framing
    "權利": ["自由", "權益", "受保障"],
    "義務": ["責任", "應盡義務"],
    "自由": ["權利", "不受限制"],

    # Residence / movement (Article 10)
    "居住": ["住居", "住所", "居留"],
    "遷徙": ["移動", "旅行", "通行"],
    "居住及遷徙自由": ["居住自由", "遷徙自由", "選擇居住地", "自由移動"],

    # Speech / religion / assembly (11~14)
    "言論": ["表達", "發言"],
    "出版": ["發表", "著作"],
    "宗教": ["信仰", "信教"],
    "集會": ["會議", "聚會"],
    "結社": ["組織", "社團"],

    # Tax / military / education (19~21)
    "納稅": ["繳稅", "稅捐"],
    "服兵役": ["兵役", "軍役", "從軍"],
    "國民教育": ["義務教育", "受教育"],

    # Preamble-ish (Article 1)
    "開宗明義": ["三民主義", "民主共和國", "民有民治民享"],
    "治理理念": ["三民主義", "民主共和國", "民有民治民享"],
    "建國理念": ["三民主義", "民主共和國", "民有民治民享"],
    "立國精神": ["三民主義", "民主共和國", "民有民治民享"],

    # Sovereignty / political participation (Article 2 / 17)
    "主權": ["主權屬於國民全體", "國民主權", "主權在民", "政權"],
    "國家主權": ["主權屬於國民全體", "國民主權", "主權", "政權"],
    "國民主權": ["主權屬於國民全體", "主權在民", "主權", "政權"],
    "主權在民": ["主權屬於國民全體", "國民主權", "主權"],
    "參與": ["參政", "行使", "選舉", "罷免", "創制", "複決"],
    "參與主權": ["國民主權", "主權在民", "主權屬於國民全體", "政權", "參政權"],
    "參與國家主權": ["國民主權", "主權在民", "主權屬於國民全體", "政權", "參政權"],
    "參政": ["選舉", "罷免", "創制", "複決", "政權"],
    "參政權": ["選舉", "罷免", "創制", "複決", "政權"],

    # simplified backups
    "权利": ["權利"],
    "选择": ["選擇"],
    "移动": ["移動", "遷徙"],
}

# =========================
# Globals (cache)
# =========================
_G_EMB = None
_G_INDEX = None
_G_METAS = None
_G_ONTOLOGY = None


# =========================
# Base utilities
# =========================
def get_emb_model():
    global _G_EMB
    if _G_EMB is None:
        _G_EMB = SentenceTransformer(EMB_MODEL)
    return _G_EMB


def load_index_and_meta():
    global _G_INDEX, _G_METAS
    if _G_INDEX is not None and _G_METAS is not None:
        return _G_INDEX, _G_METAS

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Index not found. Run: python rag_ollama_min.py build")

    _G_INDEX = faiss.read_index(INDEX_PATH)
    metas = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))

    if metas and ("article_no" not in metas[0]):
        raise RuntimeError(
            "Meta schema is old (missing article_no). Delete and rebuild:\n"
            "  del law.index law_chunks.jsonl law_ontology.json articles_extracted.jsonl\n"
            "  python rag_ollama_min.py build --llm_concepts"
        )

    _G_METAS = metas
    return _G_INDEX, _G_METAS


def has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def cjk_only(s: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", s or ""))


def pdf_to_text(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts)


def clean_text(s: str) -> str:
    s = (s or "").replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_article_lines(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        ln2 = ln.strip()
        if not ln2:
            continue
        if CHAPTER_SECTION_RE.match(ln2):
            continue
        out.append(ln.rstrip())
    return out


# =========================
# Chinese numeral -> int
# =========================
CH_NUM_MAP = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9
}


def zh_num_to_int(s: str) -> Optional[int]:
    if not s:
        return None

    total = 0
    num = 0
    has_unit = False

    for ch in s:
        if ch in CH_NUM_MAP:
            num = CH_NUM_MAP[ch]
        elif ch == "十":
            has_unit = True
            if num == 0:
                num = 1
            total += num * 10
            num = 0
        elif ch == "百":
            has_unit = True
            if num == 0:
                num = 1
            total += num * 100
            num = 0
        elif ch == "千":
            has_unit = True
            if num == 0:
                num = 1
            total += num * 1000
            num = 0
        else:
            return None

    total += num
    if total == 0 and has_unit:
        return 10
    return total if total > 0 else None


# =========================
# Normalize query to Traditional Chinese (ALWAYS)
# =========================
def normalize_to_zh_tw(query_raw: str) -> str:
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
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是精準的法律語言轉換助手。"},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )
    out = (r.get("message", {}) or {}).get("content", "").strip()
    return out if has_cjk(out) else q


# =========================
# PDF -> Article blocks -> articles_extracted.jsonl
# =========================
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")

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


# =========================
# Auto keywords + LLM concepts (ALL articles)
# =========================
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

    out = []
    for g, c in items:
        if c <= 1 and len(out) >= 20:
            break
        out.append(g)
        if len(out) >= top_k:
            break
    return out


def llm_make_article_concepts(article_no: int, article_text: str) -> List[str]:
    prompt = (
        "你在為《中華民國憲法》建立「使用者可能會怎麼問」的概念別名。\n"
        "給定某一條的條文內容，請產出 6~12 個「繁體中文」短語（2~10 字為主），"
        "用來代表這條的核心概念/常見問法（可包含同義/近義說法）。\n"
        "規則：\n"
        "1) 只輸出 JSON 陣列，例如 [\"A\",\"B\",...]\n"
        "2) 不要解釋、不要加多餘文字\n"
        "3) 避免過度泛化（不要只寫「權利」「自由」這種太泛）\n"
        "4) 允許用「誰/是否/如何/為何/何謂」等問法，但要簡短\n\n"
        f"條號：第{article_no}條\n"
        f"條文：\n{article_text}\n"
    )
    r = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是法律文件語意標註與術語整理專家。"},
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
                if not x:
                    continue
                if x in GENERIC_WORDS or len(x) == 1:
                    continue
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out[:24]
    except Exception:
        pass
    return []


def build_ontology(
    metas: List[Dict],
    use_llm_synonyms: bool = False,
    use_llm_concepts: bool = False,
    per_article_term_cap: int = 30
):
    per_article = {}
    per_aliases = {}

    for m in metas:
        art_no = m.get("article_no")
        if art_no is None:
            continue
        art_no = int(art_no)
        text = m.get("text", "")

        kws = extract_article_keywords(text, top_k=per_article_term_cap)
        per_article[str(art_no)] = kws

        if use_llm_concepts:
            aliases = llm_make_article_concepts(art_no, text)
            per_aliases[str(art_no)] = aliases

    synonym_map = dict(SEED_SYNONYMS)

    # Optional one-time synonym expansion (global)
    if use_llm_synonyms:
        global_terms = []
        for _, kws in per_article.items():
            global_terms.extend(kws[:8])

        seen = set()
        global_terms_u = []
        for t in global_terms:
            if t not in seen and len(t) >= 2:
                seen.add(t)
                global_terms_u.append(t)
        global_terms_u = global_terms_u[:60]

        prompt = (
            "你要幫我建立憲法用語的同義詞/近義詞表（繁體中文）。\n"
            "請針對每個詞，給出 3~6 個常見近義/相關表達（也用繁體中文），用 JSON 輸出。\n"
            "格式：{ \"詞\": [\"近義1\", \"近義2\", ...], ... }\n"
            "規則：只輸出 JSON，不能解釋。\n\n"
            f"詞列表：{', '.join(global_terms_u)}"
        )
        r = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是法律語言術語專家。"},
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
                        if isinstance(v, str) and v != k and v not in synonym_map[k]:
                            synonym_map[k].append(v)
        except Exception:
            pass

    onto = {
        "version": 4,
        "per_article_keywords": per_article,
        "per_article_aliases": per_aliases,
        "synonym_map": synonym_map
    }
    with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
        json.dump(onto, f, ensure_ascii=False, indent=2)
    return onto


def load_ontology():
    global _G_ONTOLOGY
    if _G_ONTOLOGY is not None:
        return _G_ONTOLOGY

    if not os.path.exists(ONTOLOGY_PATH):
        _G_ONTOLOGY = {
            "version": 4,
            "per_article_keywords": {},
            "per_article_aliases": {},
            "synonym_map": dict(SEED_SYNONYMS)
        }
        return _G_ONTOLOGY

    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        _G_ONTOLOGY = json.load(f)

    _G_ONTOLOGY.setdefault("synonym_map", {})
    for k, vs in SEED_SYNONYMS.items():
        _G_ONTOLOGY["synonym_map"].setdefault(k, [])
        for v in vs:
            if v not in _G_ONTOLOGY["synonym_map"][k]:
                _G_ONTOLOGY["synonym_map"][k].append(v)

    _G_ONTOLOGY.setdefault("per_article_keywords", {})
    _G_ONTOLOGY.setdefault("per_article_aliases", {})
    return _G_ONTOLOGY


# =========================
# Build index (PDF -> JSONL -> FAISS + ontology)
# =========================
def build_index(use_llm_concepts: bool = False, use_llm_synonyms: bool = False):
    if not os.path.exists(LAW_PDF):
        raise FileNotFoundError(f"Cannot find {LAW_PDF}. Put it next to this script.")

    export_articles_jsonl_from_pdf(LAW_PDF, ARTICLES_JSONL)

    metas = load_articles_jsonl(ARTICLES_JSONL)
    if not metas:
        raise RuntimeError("No articles loaded from articles_extracted.jsonl.")

    passages = ["passage: " + m["text"] for m in metas]
    emb_model = get_emb_model()
    embs = emb_model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    build_ontology(
        metas,
        use_llm_synonyms=use_llm_synonyms,
        use_llm_concepts=use_llm_concepts
    )

    global _G_INDEX, _G_METAS, _G_ONTOLOGY
    _G_INDEX, _G_METAS, _G_ONTOLOGY = None, None, None

    print("✅ Build completed (PDF -> JSONL artifact -> indexed by articles + ontology)")
    print(f"- PDF source: {LAW_PDF}")
    print(f"- artifact : {ARTICLES_JSONL}")
    print(f"- articles : {len(metas)}")
    print(f"- index    : {INDEX_PATH}")
    print(f"- meta     : {META_PATH}")
    print(f"- ontology : {ONTOLOGY_PATH}")
    print(f"- LLM concepts (all articles): {'ON' if use_llm_concepts else 'OFF'}")
    print(f"- LLM synonym expansion: {'ON' if use_llm_synonyms else 'OFF (seed + auto only)'}")


# =========================
# Query parsing + expansion
# =========================
def extract_article_numbers(query_zh: str) -> List[int]:
    nums = set()

    for m in re.finditer(r"第\s*(\d+)\s*條", query_zh):
        nums.add(int(m.group(1)))

    for m in re.finditer(r"第\s*([零〇一二兩三四五六七八九十百千]+)\s*條", query_zh):
        v = zh_num_to_int(m.group(1))
        if v is not None:
            nums.add(int(v))

    return sorted(nums)


def extract_keywords(query_zh: str, n_min: int = 2, n_max: int = 4) -> List[str]:
    q_cjk = cjk_only(query_zh)
    kws: List[str] = []

    for w in list(SEED_SYNONYMS.keys()):
        if w and w in query_zh:
            kws.append(w)

    cue_words = [
        "遷徙自由", "居住", "遷徙", "住居", "住所",
        "集會", "結社", "言論", "出版", "宗教", "信仰",
        "工作", "生存", "財產", "教育", "服兵役", "納稅", "平等",
        "三民主義", "民主共和國", "民有民治民享",
        "主權", "政權", "參政", "選舉", "罷免", "創制", "複決", "主權屬於國民全體"
    ]
    for w in cue_words:
        if w in query_zh:
            kws.append(w)

    L = len(q_cjk)
    for n in range(n_min, n_max + 1):
        for i in range(0, L - n + 1):
            g = q_cjk[i:i+n]
            if n == 2 and g in STOP_BIGRAMS:
                continue
            if g in GENERIC_WORDS:
                continue
            kws.append(g)

    seen = set()
    out = []
    for k in kws:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def rule_expand_query(query_zh: str) -> List[str]:
    extra = []

    if re.search(r"(是否|有無|可否|能否|得否)", query_zh):
        extra += ["是否", "得", "不得"]
    if re.search(r"(何謂|何者|為何|如何|怎樣|哪些)", query_zh):
        extra += ["規定", "內容"]
    if re.search(r"(義務|責任|應盡)", query_zh):
        extra += ["義務"]
    if re.search(r"(權利|自由|享有|保障)", query_zh):
        extra += ["權利", "自由", "保障"]

    # movement/residence
    if re.search(r"(選擇居住地|自由移動|自由移动|遷徙|移动|旅行|通行|居住)", query_zh):
        extra += ["居住及遷徙之自由", "遷徙自由", "第10條"]

    # preamble-ish / state idea
    if re.search(r"(開宗明義|宣示|治理理念|治國理念|建國理念|立國精神|國家治理)", query_zh):
        extra += ["三民主義", "民主共和國", "民有民治民享", "第1條"]

    # sovereignty / political power
    if re.search(r"(主權|國家主權|國民主權|主權在民|政權|參政|參與)", query_zh):
        extra += [
            "主權屬於國民全體", "國民主權", "主權在民", "政權",
            "參政權", "選舉", "罷免", "創制", "複決",
            "第2條", "第17條"
        ]

    out = []
    seen = set()
    for x in extra:
        if not x or x in query_zh:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def expand_query_with_ontology(query_zh: str) -> str:
    onto = load_ontology()
    syn = onto.get("synonym_map", {}) or {}

    extra_terms = []

    for k, vs in syn.items():
        if k and k in query_zh:
            extra_terms.extend(vs)

    extra_terms.extend(rule_expand_query(query_zh))

    extra_terms_u = []
    seen = set()
    for t in extra_terms:
        if not t or t in query_zh:
            continue
        if t not in seen:
            seen.add(t)
            extra_terms_u.append(t)

    if extra_terms_u:
        return query_zh + " " + " ".join(extra_terms_u)
    return query_zh


# =========================
# Retrieval
# =========================
def vector_scores(query_zh: str, top_k: int) -> Dict[int, float]:
    index, metas = load_index_and_meta()
    emb_model = get_emb_model()

    q_emb = emb_model.encode(["query: " + query_zh], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, ids = index.search(q_emb, top_k)

    out: Dict[int, float] = {}
    for sc, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        art_no = metas[int(i)].get("article_no")
        if art_no is not None:
            out[int(art_no)] = float(sc)
    return out


def lexical_score_article(text: str, keywords: List[str]) -> Tuple[int, int]:
    if not text:
        return 0, 0
    hit = 0
    score = 0
    for k in keywords:
        if k and k in text:
            hit += 1
            score += 3 if len(k) >= 3 else 1
    return hit, score


def keyword_index_score(article_no: int, query_terms: List[str]) -> Tuple[int, int]:
    onto = load_ontology()
    per = onto.get("per_article_keywords", {}) or {}
    kws = per.get(str(article_no), []) or []
    kws_set = set(kws)

    hit = 0
    s = 0
    for t in query_terms:
        if t in kws_set:
            hit += 1
            s += 4 if len(t) >= 3 else 1
    return hit, s


def alias_overlap_score(article_no: int, query_zh: str) -> Tuple[int, int]:
    onto = load_ontology()
    amap = onto.get("per_article_aliases", {}) or {}
    aliases = amap.get(str(article_no), []) or []
    if not aliases:
        return 0, 0

    hit = 0
    score = 0
    for a in aliases:
        if not isinstance(a, str):
            continue
        a = a.strip()
        if not a:
            continue
        if a in query_zh:
            hit += 1
            score += 6
    return hit, score


def retrieve_related_articles(query_zh: str, top_n: int) -> Tuple[List[Dict], str]:
    _, metas = load_index_and_meta()

    explicit = extract_article_numbers(query_zh)

    expanded_query = expand_query_with_ontology(query_zh)
    expanded_terms = extract_keywords(expanded_query)

    vmap = vector_scores(expanded_query, top_k=max(120, top_n)) if USE_VECTOR_ASSIST else {}

    scored = []
    for m in metas:
        art_no = m.get("article_no")
        if art_no is None:
            continue
        art_no = int(art_no)
        text = m.get("text", "")

        hit_txt, lex_txt = lexical_score_article(text, expanded_terms)
        hit_k, lex_k = keyword_index_score(art_no, expanded_terms)
        hit_a, lex_a = alias_overlap_score(art_no, expanded_query)

        lex_total = (lex_txt) + int(KEYIDX_WEIGHT * lex_k) + int(ALIAS_WEIGHT * lex_a)
        hit_total = hit_txt + hit_k + hit_a

        final = LEX_WEIGHT * float(lex_total)
        vec_sc = float(vmap.get(art_no, 0.0))
        if art_no in vmap:
            final += VECTOR_WEIGHT * vec_sc

        if explicit and art_no in explicit:
            final += EXPLICIT_ARTICLE_BOOST

        scored.append({
            "article_no": art_no,
            "article_zh": f"第{art_no}條",
            "id": m.get("id", ""),
            "text": text,
            "hit_count": int(hit_total),
            "lex_score": int(lex_total),
            "vec_score": vec_sc,
            "final_score": float(final)
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    supports: List[Dict] = []
    seen = set()

    # hard include explicit first
    if explicit:
        for a in explicit:
            for item in scored:
                if item["article_no"] == a and item["article_zh"] not in seen:
                    supports.append(item)
                    seen.add(item["article_zh"])
                    break

    # main fill: allow vec-strong to survive
    for item in scored:
        if len(supports) >= top_n:
            break
        if item["article_zh"] in seen:
            continue

        if item["article_no"] not in explicit:
            if item["lex_score"] < MIN_LEX_SCORE and item["vec_score"] < VEC_KEEP_TH:
                continue

        supports.append(item)
        seen.add(item["article_zh"])

    # ✅ force-keep top by vector if still not enough (critical for conceptual queries)
    if len(supports) < top_n:
        scored_by_vec = sorted(scored, key=lambda x: x["vec_score"], reverse=True)
        for it in scored_by_vec:
            if len(supports) >= top_n:
                break
            if it["article_zh"] in seen:
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
        found = set(int(s.get("article_no")) for s in supports if s.get("article_no") is not None)
        return all(a in found for a in explicit)

    total_hits = sum(int(s.get("hit_count", 0)) for s in supports)
    if any(float(s.get("vec_score", 0.0)) >= (VEC_KEEP_TH + 0.05) for s in supports):
        return True
    return total_hits >= MIN_TOTAL_HITS


# =========================
# Cheap LLM re-rank
# =========================
def llm_rerank_and_judge(query_zh: str, cands: List[Dict]) -> Optional[Dict]:
    if not cands:
        return None

    items = []
    for c in cands:
        art = int(c["article_no"])
        text = c.get("text", "")
        text_clip = text[:650].strip()
        items.append({
            "article_no": art,
            "title": f"第{art}條",
            "text": text_clip
        })

    prompt = (
        "你是一個「檢索重排序」與「可否回答判定」器。\n"
        "給你一個問題，以及多條憲法條文候選（每條只有截斷片段）。\n"
        "任務：\n"
        "1) 按與問題相關程度，重排序條號（最相關在前）。\n"
        "2) 判定：僅依這些條文，是否足以回答問題？\n"
        "   - 若不足，sufficient=false。\n"
        "3) 若問題語意等同於「主權在民/主權屬於國民」或「參政權」，通常需要第2條/第17條（若候選中存在）。\n"
        "輸出規則：\n"
        "- 只輸出 JSON，格式如下：\n"
        "  {\"order\":[...],\"sufficient\":true/false,\"need_articles\":[...]}\n"
        "- 不要解釋，不要額外文字。\n\n"
        f"問題：{query_zh}\n\n"
        f"候選條文：{json.dumps(items, ensure_ascii=False)}\n"
    )

    r = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你輸出必須是嚴格 JSON。"},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": RERANK_MODEL_TEMPERATURE},
    )
    txt = (r.get("message", {}) or {}).get("content", "").strip()

    try:
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            return None
        order = obj.get("order")
        sufficient = obj.get("sufficient")
        need_articles = obj.get("need_articles", [])
        if not isinstance(order, list) or not isinstance(sufficient, bool) or not isinstance(need_articles, list):
            return None
        order = [int(x) for x in order if isinstance(x, (int, float, str)) and str(x).isdigit()]
        need_articles = [int(x) for x in need_articles if isinstance(x, (int, float, str)) and str(x).isdigit()]
        return {"order": order, "sufficient": sufficient, "need_articles": need_articles}
    except Exception:
        return None


def apply_rerank(supports: List[Dict], query_zh: str) -> Optional[List[Dict]]:
    if not ENABLE_LLM_RERANK or not supports:
        return supports

    cands = supports[:max(RERANK_CAND_N, len(supports))]
    rr = llm_rerank_and_judge(query_zh, cands)
    if rr is None:
        return supports

    if rr.get("sufficient") is False:
        return []

    need = set(rr.get("need_articles", []) or [])
    cand_nums = set(int(c["article_no"]) for c in cands)
    if need and not need.issubset(cand_nums):
        # allow continue if missing, but don't hard-fail immediately
        need = set()

    order = rr.get("order", []) or []
    by_no = {int(c["article_no"]): c for c in cands}

    reranked = []
    seen = set()

    for a in list(need):
        if a in by_no and a not in seen:
            reranked.append(by_no[a])
            seen.add(a)

    for a in order:
        if a in by_no and a not in seen:
            reranked.append(by_no[a])
            seen.add(a)

    for c in cands:
        a = int(c["article_no"])
        if a not in seen:
            reranked.append(c)
            seen.add(a)

    return reranked[:max(1, RERANK_KEEP_TOP)]


# =========================
# LLM: answer + translation
# =========================
def translate_to_english(text_zh: str) -> str:
    text_zh = (text_zh or "").strip()
    if not text_zh:
        return "Translation error."

    prompt = (
        "Translate the following Traditional Chinese legal text into English.\n"
        "Output ENGLISH ONLY. No Chinese. No explanations.\n\n"
        f"{text_zh}"
    )
    r = ollama.chat(
        model=LLM_MODEL,
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


def llm_answer_zh_en(query_zh: str, supports: List[Dict]) -> Optional[Dict[str, str]]:
    evidence_text = build_evidence_text(supports)

    system_zh = (
        "你是法律文件問答助手。\n"
        "你只能根據【證據】回答，不得使用外部資料。\n"
        "若【證據】不足以回答問題，請只輸出一句：證據不足，無法回答。\n"
        "若可回答：請先概括總結（避免逐字長篇照抄），再回答。\n"
        "輸出格式：1~3 句 或 2~5 點條列。\n"
        "不得編造證據中不存在的數字、機關、程序或細節。\n"
        "回答必須是繁體中文。\n"
    )

    user_zh = f"""【證據】
{evidence_text}

問題：{query_zh}
"""

    r = ollama.chat(
        model=LLM_MODEL,
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

    ans_en = translate_to_english(ans_zh)
    if ans_en == "Translation error.":
        return None

    return {"answer_zh": ans_zh, "answer_en": ans_en}


# =========================
# Ask
# =========================
def answer_one(query_raw: str, topk: int = DEFAULT_TOPK) -> None:
    query_raw = (query_raw or "").strip()
    if not query_raw:
        print(FAIL_MSG)
        return

    query_zh = normalize_to_zh_tw(query_raw)

    supports, expanded_query = retrieve_related_articles(query_zh, top_n=max(topk, RERANK_CAND_N))
    if not supports or not enough_evidence(supports, expanded_query):
        print(FAIL_MSG)
        return

    reranked = apply_rerank(supports, expanded_query)
    if not reranked:
        print(FAIL_MSG)
        return

    final_supports = reranked[:max(1, min(topk, len(reranked)))]

    if not enough_evidence(final_supports, expanded_query):
        print(FAIL_MSG)
        return

    result = llm_answer_zh_en(expanded_query, final_supports)
    if not result:
        print(FAIL_MSG)
        return

    print("\n【Answer】")
    print("Chinese:", result["answer_zh"])
    print("Engsub :", result["answer_en"])

    print("\n【Extracted Articles】")
    for s in final_supports:
        print(f"- {s['article_zh']} (hit={s['hit_count']}, lex={s['lex_score']}, vec={s['vec_score']:.3f})")

    print("\n【Evidence】")
    for s in final_supports:
        ev_zh = s["text"]
        ev_en = translate_to_english(ev_zh)

        print("\nChinese:")
        print(f"{s['article_zh']}\n{ev_zh}")
        print("\nEngsub:")
        print(f"Article {s['article_no']}\n{ev_en}")


# =========================
# Batch
# =========================
def run_batch(file_path: str, topk: int):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    for q in lines:
        answer_one(q, topk=topk)


# =========================
# CLI
# =========================
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Build: python rag_ollama_min.py build [--llm_concepts] [--llm_syn]")
        print('  Ask  : python rag_ollama_min.py ask "國民是否有參與國家主權的權利" --topk 3')
        print("  Batch: python rag_ollama_min.py batch questions.txt --topk 8")
        return

    cmd = sys.argv[1].lower()

    if cmd == "build":
        use_llm_concepts = ("--llm_concepts" in sys.argv[2:])
        use_llm_syn = ("--llm_syn" in sys.argv[2:])
        build_index(use_llm_concepts=use_llm_concepts, use_llm_synonyms=use_llm_syn)
        return

    topk = DEFAULT_TOPK
    args = sys.argv[2:]
    clean_args = []
    i = 0
    while i < len(args):
        if args[i] == "--topk" and i + 1 < len(args):
            topk = int(args[i + 1])
            i += 2
        else:
            clean_args.append(args[i])
            i += 1

    if cmd == "ask":
        query = " ".join(clean_args).strip().strip('"')
        answer_one(query, topk=topk)
        return

    if cmd == "batch":
        if not clean_args:
            print("Missing questions file. Example: python rag_ollama_min.py batch questions.txt --topk 8")
            return
        run_batch(clean_args[0], topk=topk)
        return

    print(f"Unknown command: {cmd}")
    print("Commands: build | ask | batch")


if __name__ == "__main__":
    main()
