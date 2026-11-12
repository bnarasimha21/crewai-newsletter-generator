"""Issue 1 acceptance: generate() end to end, real file output, no shared state."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from newsletter.errors import Timeout
from newsletter.fakes import FakeLLM, build_golden_path
from newsletter.history import list_runs
from newsletter.pipeline import generate

pytestmark = pytest.mark.integration


class _SlowFakeLLM(FakeLLM):
    """A real FakeLLM whose first call blocks past the caller's timeout, so
    the timeout test below exercises generate()'s actual threading.Thread +
    join(timeout=...) mechanism (newsletter/pipeline.py lines ~177-189),
    not a mocked stand-in for it."""

    sleep_seconds: float = 2.0

    def call(self, *args: Any, **kwargs: Any) -> str:
        time.sleep(self.sleep_seconds)
        return super().call(*args, **kwargs)


def test_generate_writes_real_run_folder(tmp_path: Path, today) -> None:
    llm, search_tool = build_golden_path("quantum computing", today, n_items=4)

    result = generate(
        "quantum computing",
        out_dir=tmp_path,
        llm=llm,
        search_tool=search_tool,
        today=today,
    )

    assert result.run_dir.parent == tmp_path
    brief_path = result.run_dir / "brief.json"
    html_path = result.run_dir / "newsletter.html"
    pdf_path = result.run_dir / "newsletter.pdf"
    trace_path = result.run_dir / "trace.json"

    assert brief_path.exists()
    assert html_path.exists()
    assert pdf_path.exists()
    assert trace_path.exists()

    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 1024

    # Only 2 LLM calls total (one researcher Action, one Final Answer, one
    # writer Final Answer => 3 llm.call()s), never an extra conversion call:
    # output_pydantic conversion took the direct-JSON path (see
    # newsletter/fakes.py's module docstring and crewai/utilities/converter.py).
    assert llm.total_calls() == 3
    assert search_tool.calls == ["quantum computing news"]


def test_generate_imports_no_streamlit() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-I", "-c", "import newsletter.pipeline; import sys; "
         "assert 'streamlit' not in sys.modules, 'streamlit got imported'"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_parallel_generate_calls_do_not_collide(tmp_path: Path, today) -> None:
    """Two concurrent generate() calls for the same topic must not collide."""
    results: dict[str, object] = {}
    errors: dict[str, BaseException] = {}

    def run(key: str) -> None:
        try:
            llm, search_tool = build_golden_path("quantum computing", today, n_items=4)
            results[key] = generate(
                "quantum computing",
                out_dir=tmp_path,
                llm=llm,
                search_tool=search_tool,
                today=today,
            )
        except BaseException as exc:  # noqa: BLE001
            errors[key] = exc

    threads = [threading.Thread(target=run, args=(f"t{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    run_dirs = {result.run_dir for result in results.values()}
    assert len(run_dirs) == 2, "both runs must land in distinct folders"
    for result in results.values():
        assert (result.run_dir / "newsletter.pdf").exists()


def test_generate_raises_timeout_and_records_it_in_history_when_the_crew_hangs(
    tmp_path: Path, today
) -> None:
    """A real slow LLM call, a real short timeout_seconds, a real
    threading.Thread left running in the background: generate() must give
    up and raise Timeout rather than wait forever, and must record
    status="timeout" in history before doing so."""
    llm, search_tool = build_golden_path("quantum computing", today, n_items=4)
    slow_llm = _SlowFakeLLM(
        model="fake-llm", scripted_replies=llm.scripted_replies, sleep_seconds=1.0
    )

    with pytest.raises(Timeout):
        generate(
            "quantum computing",
            out_dir=tmp_path,
            llm=slow_llm,
            search_tool=search_tool,
            today=today,
            timeout_seconds=0.2,
        )

    runs = list_runs(tmp_path / "history.db")
    assert runs, "a history row must exist even for a run that timed out"
    latest = runs[0]
    assert latest.status == "timeout"
    assert latest.error is not None and "0.2" in latest.error

    # The abandoned worker thread is a daemon thread (pipeline.py's
    # threading.Thread(..., daemon=True)): it is never joined after the
    # timeout fires, so it is still sleeping in the background here. That
    # is a deliberate, documented tradeoff (a hung crew call cannot be
    # force-killed from Python), not a resource leak this test needs to
    # clean up: the thread dies on its own once _SlowFakeLLM.call()'s
    # time.sleep(2.0) returns, well before the test process exits.


def test_generate_still_succeeds_when_the_history_db_cannot_be_opened(
    tmp_path: Path, today
) -> None:
    """A real, reproducible history-write failure (history_db points at a
    directory, not a file, so sqlite3.connect() really raises
    OperationalError): generate() must still return a full, successful
    NewsletterResult with history_id=None, never let a history-db problem
    take down the actual brief/HTML/PDF a caller gets back. Exercises
    newsletter/pipeline.py's _safe_start_run/_safe_finish_run wrappers for
    real, not by mocking newsletter.history."""
    llm, search_tool = build_golden_path("quantum computing", today, n_items=4)
    unopenable_db = tmp_path / "history.db"
    unopenable_db.mkdir()  # a directory where sqlite3 expects a file

    result = generate(
        "quantum computing",
        out_dir=tmp_path,
        llm=llm,
        search_tool=search_tool,
        today=today,
        history_db=unopenable_db,
    )

    assert result.history_id is None
    assert result.pdf_path.exists()
    assert result.pdf_path.read_bytes()[:4] == b"%PDF"
