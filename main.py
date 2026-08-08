"""Mini Movie RAG System - CLI entry point.

Usage:
    python main.py build-index                      # subset -> chunk -> embed -> store
    python main.py query "your question"            # retrieve + generate structured JSON
    python main.py query "..." --top-k 5 --no-llm   # retrieval-only JSON (no API needed)
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from src.rag_graph import RAGPipeline
from src.utils import load_config

load_dotenv()

# Windows consoles default to cp1252; force UTF-8 so plot snippets with
# non-ASCII characters (e.g. "π") don't crash the CLI.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def build_index(args) -> None:
    pipeline = RAGPipeline(load_config())
    embeddings = pipeline.build_index(force=args.force)
    if embeddings is None:
        print(json.dumps({"status": "ok", "chunks_stored": pipeline.retriever.count}))
        return
    print(
        json.dumps(
            {
                "status": "ok",
                "chunks_stored": pipeline.retriever.count,
                "embedding_dim": int(embeddings.shape[1]),
            },
            indent=2,
        )
    )


def query(args) -> None:
    pipeline = RAGPipeline(load_config())
    if args.no_llm:
        chunks = pipeline.retrieve(args.question, top_k=args.top_k)
        result = {
            "answer": "(LLM skipped with --no-llm)",
            "contexts": [c["chunk_text"][:400] for c in chunks],
            "reasoning": f"Retrieved {len(chunks)} chunks without generating an answer.",
        }
    else:
        result = pipeline.answer(args.question, top_k=args.top_k)

    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini Movie RAG System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build the vector index")
    build_parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the subset and embeddings from scratch (default: reuse the cached index)",
    )
    build_parser.set_defaults(func=build_index)

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str, help="Your question about movies")
    query_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    query_parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the LLM call and return retrieval-only JSON (no API key needed)",
    )
    query_parser.set_defaults(func=query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
