"""Issue 6: `python -m newsletter` as a real subprocess, offline via
LLM_PROVIDER=fake, checking exit codes and the JSON summary it prints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parent.parent


def _run_cli(args: list[str], *, env_overrides: dict[str, str], timeout: float = 60) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [sys.executable, "-m", "newsletter", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _summary_json(stdout: str) -> dict:
    """The CLI's one `json.dumps(...)` call is the last thing it prints, but
    crewai's own rich-console logging (progress panels, emoji banners) also
    writes to stdout ahead of it. The JSON summary is the only part starting
    with "{" at the start of a line, and json.dumps(indent=2) means it can
    span multiple lines, so take everything from the first such line onward.
    """
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("{"):
            return json.loads("\n".join(lines[i:]))
    raise AssertionError(f"no JSON summary line found in CLI stdout:\n{stdout}")


def test_cli_success_prints_json_summary_and_writes_real_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    result = _run_cli(
        ["quantum computing", "--out", str(out_dir), "--template", "compact"],
        env_overrides={"LLM_PROVIDER": "fake"},
    )

    assert result.returncode == 0, result.stderr
    summary = _summary_json(result.stdout)
    assert summary["status"] == "success"
    assert summary["item_count"] >= 1

    run_dir = Path(summary["run_dir"])
    assert run_dir.is_dir()
    assert (run_dir / "brief.json").exists()
    assert (run_dir / "newsletter.html").exists()
    pdf_path = Path(summary["pdf_path"])
    assert pdf_path.read_bytes()[:4] == b"%PDF"
    assert pdf_path.stat().st_size > 1024


def test_cli_exits_with_timeout_code_when_the_deadline_is_too_tight(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    result = _run_cli(
        ["quantum computing", "--out", str(out_dir), "--timeout", "0.001"],
        env_overrides={"LLM_PROVIDER": "fake"},
    )

    assert result.returncode == 2, result.stderr
    summary = _summary_json(result.stdout)
    assert summary["status"] == "timeout"


def test_cli_without_llm_provider_fake_and_no_api_key_fails_cleanly(tmp_path: Path) -> None:
    """With no LLM_PROVIDER=fake and no reachable LLM endpoint, the crew's
    LLM call must fail and the CLI must exit 1 with a JSON error, not crash
    with an unhandled traceback or time out."""
    out_dir = tmp_path / "runs"
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-test-invalid-not-a-real-key",
        "SERPER_API_KEY": "test-invalid-not-a-real-key",
        # Point the OpenAI client at a closed local port instead of the
        # real api.openai.com. Originally this test relied on a genuine
        # network round trip to get a 401, which an adversarial review
        # flagged as flaky: in a no-egress sandbox or CI runner, the
        # connection attempt doesn't fail fast with 401, it hangs or fails
        # with a different error (or even times out past the 20s pipeline
        # deadline, flipping the result to the "timeout" exit code 2 this
        # test does not expect). A closed loopback port refuses the
        # connection immediately and deterministically everywhere,
        # regardless of egress policy, while still exercising the same
        # "LLM call fails -> CLI exits 1 with a JSON error" path.
        "OPENAI_BASE_URL": "http://127.0.0.1:1",
        "OPENAI_API_BASE": "http://127.0.0.1:1",
    }
    # Explicitly deleted, not just absent from the overrides dict: `pop` on
    # a dict of overrides never touches a value already inherited from
    # os.environ, so if the ambient shell (or a CI job step) happens to
    # export LLM_PROVIDER=fake, the subprocess would silently inherit it
    # and this test would stop testing the real-credentials failure path.
    env.pop("LLM_PROVIDER", None)
    result = subprocess.run(
        [sys.executable, "-m", "newsletter", "quantum computing", "--out", str(out_dir), "--timeout", "20"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, result.stderr
    summary = _summary_json(result.stdout)
    assert summary["status"] == "failed"
    assert summary["error"]


def test_cli_help_does_not_import_streamlit() -> None:
    """A plain --help run must work even without Streamlit installed, since
    this is the offline entry point newsletter/pipeline.py promises."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import newsletter.__main__"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "streamlit" not in sys.modules
