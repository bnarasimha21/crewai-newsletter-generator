"""Issue 1 acceptance: generate() end to end, real file output, no shared state."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from newsletter.fakes import build_golden_path
from newsletter.pipeline import generate

pytestmark = pytest.mark.integration


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
