"""Command line entry point: `rag ingest`, `rag ask "question"`, `rag serve`."""

from __future__ import annotations

import argparse
import json
import sys

from rag.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Docs RAG assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="(Re)build the index from the docs directory")

    ask = sub.add_parser("ask", help="Ask a question against the index")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Print the raw response as JSON")

    serve = sub.add_parser("serve", help="Run the FastAPI server")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "ingest":
        from rag.service import build_index

        count = build_index(settings)
        print(f"Indexed {count} chunks from {settings.docs_dir} -> {settings.db_path}")
        return 0

    if args.command == "ask":
        from rag.service import RagService

        result = RagService(settings).ask(args.question)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        print(result.answer)
        if result.citations:
            print("\nSources:")
            for c in result.citations:
                print(f"  - {c.doc} p.{c.page}")
        print(f"\nconfidence={result.confidence:.2f} abstained={result.abstained}")
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("rag.api:app", host="127.0.0.1", port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
