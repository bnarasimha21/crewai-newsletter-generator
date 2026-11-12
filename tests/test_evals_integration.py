"""Issue 5: the eval harness runs the real pipeline end to end offline
(--llm fake) over every fixture and produces a real markdown report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from evals.checks import (
    check_citation_validity,
    check_faithfulness,
    check_freshness,
    check_layout,
    check_source_diversity,
)
from evals.run import FIXTURES_DIR, _load_fixture, render_markdown_report, run_fixture
from newsletter.models import Brief, Item

pytestmark = pytest.mark.integration


def test_six_fixtures_exist_and_parse():
    fixture_paths = sorted(FIXTURES_DIR.glob("*.json"))
    assert len(fixture_paths) == 6
    for path in fixture_paths:
        fixture = _load_fixture(path)
        assert fixture["topic"]
        assert len(fixture["search_results"]) >= 3
        for r in fixture["search_results"]:
            assert r["link"].startswith("https://")


def test_fake_mode_runs_one_fixture_end_to_end_and_passes(tmp_path: Path) -> None:
    fixture_path = FIXTURES_DIR / "quantum_computing.json"
    result = run_fixture(
        fixture_path,
        llm_mode="fake",
        model=None,
        out_dir=tmp_path,
        today=date(2026, 11, 12),
        window_days=30,
        max_per_domain=2,
        min_items=3,
    )
    assert result.error is None
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]
    assert {c.name for c in result.checks} == {
        "citation_validity",
        "freshness",
        "source_diversity",
        "faithfulness",
        "layout",
    }


def test_fake_mode_runs_all_six_fixtures_and_writes_a_real_report(tmp_path: Path) -> None:
    results = [
        run_fixture(
            path,
            llm_mode="fake",
            model=None,
            out_dir=tmp_path,
            today=date(2026, 11, 12),
            window_days=30,
            max_per_domain=2,
            min_items=3,
        )
        for path in sorted(FIXTURES_DIR.glob("*.json"))
    ]
    assert len(results) == 6
    for r in results:
        assert r.passed, f"{r.topic}: {[(c.name, c.detail) for c in r.checks if not c.passed]}"

    report = render_markdown_report(
        llm_mode="fake", model=None, results=results, today=date(2026, 11, 12)
    )
    assert "citation_validity" in report
    assert "PASS" in report
    for r in results:
        assert r.topic in report


def test_citation_validity_catches_a_fabricated_url():
    brief = Brief(
        topic="t",
        generated_on=date(2026, 11, 12),
        intro="i",
        items=[
            Item(
                title="A",
                summary="s",
                source_url="https://example.com/fabricated-not-in-fixture",
                source_name="Example",
                published=None,
            )
        ],
        outro="o",
    )
    result = check_citation_validity(brief, {"https://example.com/real-article"})
    assert not result.passed


def test_freshness_catches_a_stale_item():
    brief = Brief(
        topic="t",
        generated_on=date(2026, 11, 12),
        intro="i",
        items=[
            Item(
                title="A",
                summary="s",
                source_url="https://example.com/a",
                source_name="Example",
                published=date(2026, 1, 1),
            )
        ],
        outro="o",
    )
    result = check_freshness(brief, date(2026, 11, 12), window_days=30)
    assert not result.passed


def test_source_diversity_catches_domain_overflow():
    brief = Brief(
        topic="t",
        generated_on=date(2026, 11, 12),
        intro="i",
        items=[
            Item(
                title=f"Item {i}",
                summary="s",
                source_url=f"https://example.com/item-{i}",
                source_name="Example",
                published=None,
            )
            for i in range(3)
        ],
        outro="o",
    )
    result = check_source_diversity(brief, max_per_domain=2)
    assert not result.passed


def test_faithfulness_catches_a_summary_unrelated_to_its_source():
    brief = Brief(
        topic="t",
        generated_on=date(2026, 11, 12),
        intro="i",
        items=[
            Item(
                title="A",
                summary="Completely unrelated sentence about cooking recipes and baking bread.",
                source_url="https://example.com/a",
                source_name="Example",
                published=None,
            )
        ],
        outro="o",
    )
    source_snippets = {
        "https://example.com/a": "Researchers announced a breakthrough in fusion reactor containment."
    }
    result = check_faithfulness(brief, source_snippets)
    assert not result.passed


def test_layout_catches_a_missing_pdf(tmp_path: Path):
    brief = Brief(
        topic="t", generated_on=date(2026, 11, 12), intro="i", items=[], outro="o"
    )
    result = check_layout(tmp_path / "does_not_exist.pdf", brief)
    assert not result.passed
