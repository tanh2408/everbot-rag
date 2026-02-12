# rag_ollama_min.py
# Thin wrapper for Everbot RAG package.
# Usage:
#   python rag_ollama_min.py build --pdf "中華民國憲法.pdf"
#   python rag_ollama_min.py ask "根據憲法第20條的規定，國民有什麼樣的義務？" --topk 3
#
from everbot_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
