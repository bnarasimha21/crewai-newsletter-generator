"""The single entry point: generate(topic) -> NewsletterResult.

Never touches the process's working directory (issue 1's bug: the old
code wrote report.html/report.pdf into cwd, so two concurrent runs
overwrote each other). Every run gets its own `runs/<timestamp>-<slug>/`
directory, created fresh, and only ever writes inside it.

Importing this module must not import Streamlit (checked by
tests/test_pipeline_integration.py), so the CLI in newsletter/__main__.py
can run with no Streamlit installed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from newsletter.crew import NewsletterCrew
from newsletter.errors import RenderFailed, SearchFailed, Timeout
from newsletter.history import estimate_cost, finish_run, start_run
from newsletter.models import Brief
from newsletter.render import html_to_pdf
from newsletter.trace import Trace, TraceRecorder

logger = logging.getLogger("newsletter.pipeline")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DEFAULT_RUNS_DIR = Path("runs")
_DEFAULT_HISTORY_DB = _DEFAULT_RUNS_DIR / "history.db"


def _safe_start_run(db_path: Path, **kwargs: Any) -> int | None:
    """Record a run's start without letting a history-db hiccup (e.g. a
    locked SQLite file under concurrent generate() calls) crash the whole
    pipeline with an untyped exception. History is supplementary to the
    actual brief/HTML/PDF a caller gets back, so a failure here is logged
    and swallowed, never raised.
    """
    try:
        return start_run(db_path, **kwargs)
    except Exception:  # noqa: BLE001 - history writes must never crash generate()
        logger.warning("failed to record run start in %s", db_path, exc_info=True)
        return None


def _safe_finish_run(db_path: Path, run_id: int | None, **kwargs: Any) -> None:
    if run_id is None:
        return
    try:
        finish_run(db_path, run_id, **kwargs)
    except Exception:  # noqa: BLE001 - history writes must never crash generate()
        logger.warning("failed to record run finish in %s", db_path, exc_info=True)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class NewsletterResult:
    run_dir: Path
    brief: Brief
    html_path: Path
    pdf_path: Path
    trace: Trace
    history_id: int | None = None


def _slugify(topic: str) -> str:
    slug = _SLUG_RE.sub("-", topic.strip().lower()).strip("-")
    return slug or "topic"


def _make_run_dir(out_dir: Path, topic: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    # A uuid4 suffix, not just the timestamp: two generate() calls for the
    # same topic in the same second (the parallel-run test in
    # tests/test_pipeline_integration.py does exactly this) must never
    # collide on the same folder.
    unique = uuid.uuid4().hex[:8]
    run_dir = out_dir / f"{timestamp}-{_slugify(topic)}-{unique}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _render_template(brief: Brief, template: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    jinja_template = env.get_template("newsletter.html.j2")
    return jinja_template.render(brief=brief, css_class=template)


def _run_crew(
    *,
    topic: str,
    today: date,
    llm: Any | None,
    search_tool: Any | None,
    window_days: int,
    max_per_domain: int,
    min_items: int,
    guardrail_max_retries: int,
    recorder: TraceRecorder,
) -> tuple[Brief, int]:
    newsletter_crew = NewsletterCrew(
        topic=topic,
        today=today,
        llm=llm,
        search_tool=search_tool,
        window_days=window_days,
        max_per_domain=max_per_domain,
        min_items=min_items,
        guardrail_max_retries=guardrail_max_retries,
    )
    try:
        with recorder.stage("research_and_write"):
            crew_output = newsletter_crew.crew().kickoff(inputs=newsletter_crew.inputs())
    except Exception as exc:
        raise SearchFailed(
            f"The research stage could not produce a valid brief: {exc}"
        ) from exc
    finally:
        newsletter_crew.detach_provenance()

    search_calls = (
        newsletter_crew.provenance_tracker.calls
        if newsletter_crew.provenance_tracker is not None
        else 0
    )

    brief_pydantic = crew_output.pydantic
    if not isinstance(brief_pydantic, Brief):
        raise SearchFailed(
            "The writer task did not produce a structured Brief; "
            f"raw output was: {crew_output.raw[:500]!r}"
        )
    return brief_pydantic, search_calls


def generate(
    topic: str,
    out_dir: str | Path | None = None,
    llm: Any | None = None,
    *,
    search_tool: Any | None = None,
    template: str = "classic",
    today: date | None = None,
    window_days: int = 30,
    max_per_domain: int = 2,
    min_items: int = 3,
    guardrail_max_retries: int = 2,
    timeout_seconds: float = 300.0,
    history_db: str | Path | None = None,
) -> NewsletterResult:
    """Research, write and render one newsletter brief.

    Raises:
        SearchFailed: the research stage never produced a valid brief.
        RenderFailed: WeasyPrint could not turn the brief's HTML into a PDF.
        Timeout: the crew did not finish within `timeout_seconds`.
    """
    out_path = Path(out_dir) if out_dir is not None else _DEFAULT_RUNS_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    history_path = Path(history_db) if history_db is not None else out_path / "history.db"
    today = today or date.today()

    run_dir = _make_run_dir(out_path, topic)
    started_at = datetime.now(UTC).isoformat()
    history_id = _safe_start_run(
        history_path, topic=topic, started_at=started_at, run_dir=str(run_dir)
    )

    recorder = TraceRecorder()
    result_holder: dict[str, Any] = {}
    error_holder: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            brief, search_calls = _run_crew(
                topic=topic,
                today=today,
                llm=llm,
                search_tool=search_tool,
                window_days=window_days,
                max_per_domain=max_per_domain,
                min_items=min_items,
                guardrail_max_retries=guardrail_max_retries,
                recorder=recorder,
            )
            result_holder["brief"] = brief
            result_holder["search_calls"] = search_calls
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            error_holder["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        recorder.detach()
        _safe_finish_run(
            history_path,
            history_id,
            finished_at=datetime.now(UTC).isoformat(),
            status="timeout",
            error=f"exceeded {timeout_seconds}s timeout",
        )
        raise Timeout(f"Newsletter generation for {topic!r} exceeded {timeout_seconds}s")

    if "error" in error_holder:
        exc = error_holder["error"]
        status = "failed"
        recorder.detach()
        _safe_finish_run(
            history_path,
            history_id,
            finished_at=datetime.now(UTC).isoformat(),
            status=status,
            error=str(exc),
        )
        if isinstance(exc, (SearchFailed, RenderFailed, Timeout)):
            raise exc
        raise SearchFailed(f"Newsletter generation failed: {exc}") from exc

    brief: Brief = result_holder["brief"]
    search_calls: int = result_holder["search_calls"]

    brief_path = run_dir / "brief.json"
    brief_path.write_text(brief.model_dump_json(indent=2), encoding="utf-8")

    with recorder.stage("render"):
        html = _render_template(brief, template)
        html_path = run_dir / "newsletter.html"
        html_path.write_text(html, encoding="utf-8")

        pdf_path = run_dir / "newsletter.pdf"
        try:
            html_to_pdf(html, pdf_path)
        except RenderFailed as exc:
            recorder.detach()
            _safe_finish_run(
                history_path,
                history_id,
                finished_at=datetime.now(UTC).isoformat(),
                status="failed",
                error=str(exc),
            )
            raise

    trace = recorder.finish(search_calls=search_calls)
    (run_dir / "trace.json").write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")

    _safe_finish_run(
        history_path,
        history_id,
        finished_at=datetime.now(UTC).isoformat(),
        status="success",
        tokens=trace.prompt_tokens + trace.completion_tokens,
        estimated_cost=estimate_cost(trace.prompt_tokens, trace.completion_tokens),
    )

    return NewsletterResult(
        run_dir=run_dir,
        brief=brief,
        html_path=html_path,
        pdf_path=pdf_path,
        trace=trace,
        history_id=history_id,
    )
