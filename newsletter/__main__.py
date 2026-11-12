"""CLI entry point: `python -m newsletter "topic" --template compact --out runs/`.

Deliberately does not import streamlit (see newsletter/pipeline.py's own
no-streamlit-import requirement), so this works with no Streamlit installed.

Setting the environment variable LLM_PROVIDER=fake swaps in
newsletter.fakes.build_golden_path instead of a real LLM/search tool, so
the whole CLI can run end to end offline with no API key. That is exactly
what tests/test_cli_integration.py does (it runs this module as a real
subprocess with LLM_PROVIDER=fake set) and what a human can run locally to
sanity check the pipeline before wiring up real credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from newsletter.errors import RenderFailed, SearchFailed, Timeout
from newsletter.pipeline import generate

# Exit codes: 0 success, 1 generic pipeline failure, 2 timeout specifically
# (so a CI job or cron can distinguish "the crew is stuck" from "it failed").
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_TIMEOUT = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsletter",
        description="Generate a researched newsletter brief, HTML and PDF for a topic.",
    )
    parser.add_argument("topic", help="Topic to research and write a newsletter about")
    parser.add_argument(
        "--template",
        choices=["classic", "compact"],
        default="classic",
        help="Which template to render the HTML/PDF with (default: classic)",
    )
    parser.add_argument(
        "--out",
        default="runs",
        help="Directory under which a fresh run folder is created (default: runs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the crew before giving up (default: 300)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Freshness window in days for the research guardrail (default: 30)",
    )
    parser.add_argument(
        "--max-per-domain",
        type=int,
        default=2,
        help="Max items allowed from one domain (default: 2)",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=3,
        help="Minimum verified items required for a successful run (default: 3)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    llm = None
    search_tool = None
    if os.environ.get("LLM_PROVIDER", "").strip().lower() == "fake":
        from newsletter.fakes import build_golden_path

        llm, search_tool = build_golden_path(args.topic, date.today())

    try:
        result = generate(
            args.topic,
            out_dir=Path(args.out),
            llm=llm,
            search_tool=search_tool,
            template=args.template,
            timeout_seconds=args.timeout,
            window_days=args.window_days,
            max_per_domain=args.max_per_domain,
            min_items=args.min_items,
        )
    except Timeout as exc:
        print(json.dumps({"status": "timeout", "error": str(exc)}), file=sys.stdout)
        print(f"newsletter: timed out: {exc}", file=sys.stderr)
        return EXIT_TIMEOUT
    except (SearchFailed, RenderFailed) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stdout)
        print(f"newsletter: failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except OSError as exc:
        # generate() has several unguarded filesystem calls (making the
        # run dir, writing brief.json/newsletter.html/trace.json) that are
        # none of the typed pipeline errors above. Without this, a bad
        # --out path (permission denied, disk full) raised a bare
        # traceback on stderr with nothing on stdout, breaking the
        # documented "always a JSON summary on stdout" contract this CLI
        # promises.
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stdout)
        print(f"newsletter: failed: {exc}", file=sys.stderr)
        return EXIT_FAILED

    summary = {
        "status": "success",
        "topic": args.topic,
        "run_dir": str(result.run_dir),
        "html_path": str(result.html_path),
        "pdf_path": str(result.pdf_path),
        "item_count": len(result.brief.items),
        "trace": result.trace.to_dict(),
    }
    print(json.dumps(summary, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
