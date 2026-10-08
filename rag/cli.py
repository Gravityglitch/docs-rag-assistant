"""Command line entry point: ``rag ingest [--full]``, ``rag ask "question"``, ``rag serve``."""

from __future__ import annotations

import argparse
import json
import sys

from rag.config import Settings
from rag.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Docs RAG assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Sync the index with the docs directory")
    ingest.add_argument("--full", action="store_true", help="Discard the index and rebuild")

    ask = sub.add_parser("ask", help="Ask a question against the index")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Print the raw response as JSON")
    ask.add_argument("--debug", action="store_true", help="Include retrieval diagnostics")

    serve = sub.add_parser("serve", help="Run the FastAPI server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    configure_logging(json_logs=settings.log_json, level=settings.log_level)

    if args.command == "ingest":
        from rag.service import build_index

        result = build_index(settings, full=args.full)
        target = settings.database_url or str(settings.db_path)
        print(
            f"Index has {result.chunks} chunks ({target}): "
            f"{len(result.indexed)} indexed, {len(result.skipped)} unchanged, "
            f"{len(result.pruned)} pruned, {len(result.failed)} failed"
        )
        for doc, error in result.failed.items():
            print(f"  FAILED {doc}: {error}", file=sys.stderr)
        return 1 if result.failed else 0

    if args.command == "ask":
        from rag.service import RagService

        result = RagService(settings).ask(args.question)
        if args.json:
            print(json.dumps(result.to_dict(debug=args.debug), indent=2))
            return 0
        print(result.answer)
        if result.citations:
            print("\nSources:")
            for c in result.citations:
                print(f"  - {c.doc} p.{c.page}")
        print(f"\nconfidence={result.confidence:.2f} abstained={result.abstained}")
        if args.debug and result.diagnostics:
            print(json.dumps(result.diagnostics.to_dict(), indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "rag.api:app",
            host=args.host or settings.host,
            port=args.port or settings.port,
            log_config=None,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
