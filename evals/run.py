"""Newsletter quality evals: run the real two-agent crew over a fixed set
of fixture topics and score the resulting Brief against citation validity,
freshness, source diversity, faithfulness and layout.

Usage:
    python -m evals.run --llm fake
    python -m evals.run --llm do_serverless --model gpt-4o-mini
    python -m evals.run --llm claude_cli

`--llm fake` needs no credentials and is what CI runs on every pull
request (see tests/test_evals_integration.py). `--llm do_serverless` and
`--llm claude_cli` exercise a real model's research-writing quality and are
meant to be run locally/manually, not in CI, since they need either a
DigitalOcean serverless inference key or a logged-in `claude` CLI.

All three modes use the same fixture search results (evals/fixtures/*.json)
as the search tool's return value, regardless of which LLM is under test:
this isolates what is being evaluated to the LLM's research-summarizing
and writing quality, not whether a real web search API is reachable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from evals.checks import CheckResult, run_all_checks
from newsletter.fakes import FakeLLM, FakeSearchTool
from newsletter.guardrails import normalize_url
from newsletter.pipeline import generate

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"

_RELATIVE_DATE_RE = re.compile(r"(\d+)\s*(day|week|month)s?\s*ago")


def _resolve_relative_date(text: str, today: date) -> date:
    match = _RELATIVE_DATE_RE.search(text)
    if not match:
        return today
    amount, unit = int(match.group(1)), match.group(2)
    days = {"day": 1, "week": 7, "month": 30}[unit] * amount
    return today - timedelta(days=days)


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_fake_llm(fixture: dict[str, Any], today: date) -> FakeLLM:
    """Script a FakeLLM that cites exactly the fixture's own results, so
    --llm fake exercises the full real pipeline end to end offline without
    needing a real model to write faithful prose."""
    topic = fixture["topic"]
    results = fixture["search_results"]
    items = [
        {
            "title": r["title"],
            "summary": r["snippet"],
            "source_url": r["link"],
            "source_name": r["source"],
            "published": _resolve_relative_date(r["date"], today).isoformat(),
        }
        for r in results
    ]
    research_json = json.dumps({"topic": topic, "items": items})
    brief_json = json.dumps(
        {
            "topic": topic,
            "generated_on": today.isoformat(),
            "intro": f"Here is what is new in {topic} this week.",
            "items": items,
            "outro": "That is all for this edition.",
        }
    )
    return FakeLLM(
        model="fake-llm",
        scripted_replies={
            "researcher": [
                "Thought: I should search for recent news on this topic.\n"
                "Action: Search the internet with Serper\n"
                f'Action Input: {{"search_query": "{topic} news"}}',
                f"Thought: I have enough verified items.\nFinal Answer: {research_json}",
            ],
            "writer": [
                f"Thought: I will summarize the supplied items only.\nFinal Answer: {brief_json}",
            ],
        },
    )


def _build_llm(llm_mode: str, model: str | None, fixture: dict[str, Any], today: date) -> Any:
    if llm_mode == "fake":
        return _build_fake_llm(fixture, today)
    if llm_mode == "do_serverless":
        from evals.llms import build_do_serverless_llm

        return build_do_serverless_llm(model)
    if llm_mode == "claude_cli":
        from evals.llms import build_claude_cli_llm

        return build_claude_cli_llm()
    raise ValueError(f"unknown --llm mode: {llm_mode!r}")


@dataclass
class FixtureEvalResult:
    topic: str
    fixture_name: str
    checks: list[CheckResult]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)


def run_fixture(
    fixture_path: Path,
    *,
    llm_mode: str,
    model: str | None,
    out_dir: Path,
    today: date,
    window_days: int,
    max_per_domain: int,
    min_items: int,
) -> FixtureEvalResult:
    fixture = _load_fixture(fixture_path)
    topic = fixture["topic"]
    results = fixture["search_results"]
    fixture_urls = {r["link"] for r in results}
    source_snippets = {normalize_url(r["link"]): r["snippet"] for r in results}

    llm = _build_llm(llm_mode, model, fixture, today)
    search_tool = FakeSearchTool(fixture_results=results)

    try:
        result = generate(
            topic,
            out_dir=out_dir,
            llm=llm,
            search_tool=search_tool,
            today=today,
            window_days=window_days,
            max_per_domain=max_per_domain,
            min_items=min_items,
            template="classic",
        )
    except Exception as exc:  # noqa: BLE001 - recorded as a failed fixture, not a crash
        return FixtureEvalResult(
            topic=topic, fixture_name=fixture_path.stem, checks=[], error=str(exc)
        )

    checks = run_all_checks(
        brief=result.brief,
        pdf_path=result.pdf_path,
        fixture_urls=fixture_urls,
        source_snippets=source_snippets,
        today=today,
        window_days=window_days,
        max_per_domain=max_per_domain,
    )
    return FixtureEvalResult(topic=topic, fixture_name=fixture_path.stem, checks=checks)


def render_markdown_report(
    *, llm_mode: str, model: str | None, results: list[FixtureEvalResult], today: date
) -> str:
    model_label = model or llm_mode
    lines = [
        f"# Newsletter eval run: {llm_mode} ({model_label})",
        "",
        f"Date: {today.isoformat()}",
        "",
        "| Topic | Status | " + " | ".join(
            sorted({c.name for r in results for c in r.checks})
        ) + " |",
    ]
    all_check_names = sorted({c.name for r in results for c in r.checks})
    lines.append("| --- | --- | " + " | ".join("---" for _ in all_check_names) + " |")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.error is not None:
            lines.append(f"| {r.topic} | ERROR | " + " | ".join("-" for _ in all_check_names) + " |")
            continue
        by_name = {c.name: c for c in r.checks}
        cells = ["ok" if by_name[name].passed else "FAIL" for name in all_check_names]
        lines.append(f"| {r.topic} | {status} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Detail")
    for r in results:
        lines.append(f"### {r.topic} ({r.fixture_name})")
        if r.error is not None:
            lines.append(f"- pipeline error: {r.error}")
        else:
            for c in r.checks:
                marker = "PASS" if c.passed else "FAIL"
                lines.append(f"- [{marker}] {c.name}: {c.detail}")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals.run", description="Run newsletter quality evals")
    parser.add_argument(
        "--llm", choices=["fake", "do_serverless", "claude_cli"], default="fake"
    )
    parser.add_argument("--model", default=None, help="Model name, used by --llm do_serverless")
    parser.add_argument(
        "--topics", nargs="*", default=None, help="Fixture stems to run (default: all)"
    )
    parser.add_argument("--out", default=None, help="Run output dir (default: a temp runs/ dir)")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-per-domain", type=int, default=2)
    parser.add_argument("--min-items", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fixture_paths = sorted(FIXTURES_DIR.glob("*.json"))
    if args.topics:
        fixture_paths = [p for p in fixture_paths if p.stem in args.topics]
    if not fixture_paths:
        print("evals.run: no fixtures matched", file=sys.stderr)
        return 1

    today = date.today()
    out_dir = Path(args.out) if args.out else Path("evals") / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[FixtureEvalResult] = []
    for fixture_path in fixture_paths:
        print(f"evals.run: running {fixture_path.stem} ({args.llm})", file=sys.stderr)
        results.append(
            run_fixture(
                fixture_path,
                llm_mode=args.llm,
                model=args.model,
                out_dir=out_dir,
                today=today,
                window_days=args.window_days,
                max_per_domain=args.max_per_domain,
                min_items=args.min_items,
            )
        )

    report = render_markdown_report(llm_mode=args.llm, model=args.model, results=results, today=today)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_label = (args.model or args.llm).replace("/", "-")
    timestamp = datetime.now(UTC).strftime("%Y%m%d")
    report_path = RESULTS_DIR / f"{timestamp}-{model_label}.md"
    report_path.write_text(report, encoding="utf-8")

    n_passed = sum(1 for r in results if r.passed)
    print(f"evals.run: {n_passed}/{len(results)} fixtures passed all checks", file=sys.stderr)
    print(f"evals.run: report written to {report_path}", file=sys.stderr)
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
