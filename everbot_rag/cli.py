# everbot_rag/cli.py
from __future__ import annotations
import argparse
import os
from typing import Optional

from .config import (
    LAW_PDF_DEFAULT, EMB_MODEL_DEFAULT, LLM_MODEL_DEFAULT, DEFAULT_TOPK, FAIL_MSG
)
from .index_build import build_all
from .pipeline import ask as ask_pipeline
from .qa import translate_to_english

def cmd_build(args: argparse.Namespace) -> int:
    artifacts = build_all(
        pdf_path=args.pdf,
        out_dir=args.out_dir,
        emb_model=args.emb_model,
        llm_model=args.llm_model,
        use_llm_concepts=args.llm_concepts,
        use_llm_synonyms=args.llm_synonyms,
    )
    print("✅ Build completed")
    for k, v in artifacts.items():
        print(f"- {k}: {v}")
    print("Notes:")
    print("- articles_extracted.jsonl is an audit artifact generated FROM the PDF (not an external source).")
    return 0

def cmd_ask(args: argparse.Namespace) -> int:
    res = ask_pipeline(
        query_raw=args.question,
        topk=args.topk,
        base_dir=args.base_dir,
        emb_model=args.emb_model,
        llm_model=args.llm_model,
        enable_rerank=(not args.no_rerank),
    )
    if not res:
        print(FAIL_MSG)
        return 1

    print("\n【Answer】")
    print("Chinese:", res["answer_zh"])
    print("Engsub :", res["answer_en"])

    print("\n【Extracted Articles】")
    for s in res["supports"][:args.topk]:
        print(f"- {s['article_zh']} (hit={s['hit_count']}, lex={s['lex_score']}, vec={float(s['vec_score']):.3f})")

    print("\n【Evidence】")
    for s in res["supports"][:args.topk]:
        ev_zh = s["text_zh"]
        ev_en = translate_to_english(ev_zh, llm_model=args.llm_model)
        print("\nChinese:")
        print(f"{s['article_zh']}\n{ev_zh}")
        print("\nEngsub:")
        print(f"Article {s['article_no']}\n{ev_en}")
    return 0

def cmd_batch(args: argparse.Namespace) -> int:
    if not os.path.exists(args.file):
        raise FileNotFoundError(args.file)
    with open(args.file, "r", encoding="utf-8") as f:
        questions = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    for q in questions:
        print("\n" + "="*80)
        print("Q:", q)
        _ = cmd_ask(argparse.Namespace(
            question=q, topk=args.topk, base_dir=args.base_dir,
            emb_model=args.emb_model, llm_model=args.llm_model,
            no_rerank=args.no_rerank
        ))
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="everbot_rag", description="Offline RAG for law PDF (article-based).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Extract articles from PDF -> JSONL -> build FAISS index + ontology.")
    pb.add_argument("--pdf", default=LAW_PDF_DEFAULT, help="Path to law PDF.")
    pb.add_argument("--out-dir", default=".", help="Output directory for artifacts.")
    pb.add_argument("--emb-model", default=EMB_MODEL_DEFAULT, help="SentenceTransformer embedding model.")
    pb.add_argument("--llm-model", default=LLM_MODEL_DEFAULT, help="Ollama model for normalization/QA.")
    pb.add_argument("--llm-concepts", action="store_true", help="Use LLM to generate per-article concept aliases (build-time).")
    pb.add_argument("--llm-synonyms", action="store_true", help="Use LLM to expand synonym map (build-time).")
    pb.set_defaults(func=cmd_build)

    pa = sub.add_parser("ask", help="Ask a question; answer from extracted articles only.")
    pa.add_argument("question", help="Question text (any language; normalized to Traditional Chinese).")
    pa.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Number of candidate articles.")
    pa.add_argument("--base-dir", default=".", help="Directory containing built artifacts.")
    pa.add_argument("--emb-model", default=EMB_MODEL_DEFAULT)
    pa.add_argument("--llm-model", default=LLM_MODEL_DEFAULT)
    pa.add_argument("--no-rerank", action="store_true", help="Disable LLM cheap re-rank.")
    pa.set_defaults(func=cmd_ask)

    pbatch = sub.add_parser("batch", help="Ask multiple questions from a text file.")
    pbatch.add_argument("file", help="Path to questions.txt")
    pbatch.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    pbatch.add_argument("--base-dir", default=".")
    pbatch.add_argument("--emb-model", default=EMB_MODEL_DEFAULT)
    pbatch.add_argument("--llm-model", default=LLM_MODEL_DEFAULT)
    pbatch.add_argument("--no-rerank", action="store_true")
    pbatch.set_defaults(func=cmd_batch)

    return p

def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)

if __name__ == "__main__":
    raise SystemExit(main())
