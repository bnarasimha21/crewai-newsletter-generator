"""Issue 2: template rendering is byte-identical for a fixed Brief, escapes
hostile content, and never hardcodes a year."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from newsletter.models import Brief, Item
from newsletter.pipeline import _render_template

FIXED_BRIEF = Brief(
    topic="quantum computing",
    generated_on=date(2026, 11, 12),
    intro="Here is what is new in quantum computing this week.",
    items=[
        Item(
            title="Quantum Error Correction Hits New Milestone",
            summary="Researchers report a breakthrough in fault-tolerant qubits.",
            source_url="https://www.technews.example.com/quantum-error-correction",
            source_name="Tech News Example",
            published=date(2026, 11, 9),
        ),
        Item(
            title="Startup Raises Funding For Quantum Chips",
            summary="A new chipmaker closed a funding round.",
            source_url="https://venture.example.com/quantum-chip-funding",
            source_name="Venture Example",
            published=None,
        ),
    ],
    outro="That is all for this edition.",
)


def test_rendering_a_fixed_brief_is_byte_identical_across_runs():
    html_a = _render_template(FIXED_BRIEF, "classic")
    html_b = _render_template(FIXED_BRIEF, "classic")
    assert html_a == html_b


def test_rendering_a_fixed_brief_matches_a_committed_snapshot():
    snapshot_path = Path(__file__).parent / "snapshots" / "newsletter_classic.html"
    rendered = _render_template(FIXED_BRIEF, "classic")
    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(rendered, encoding="utf-8")
    assert rendered == snapshot_path.read_text(encoding="utf-8")


def test_xss_payload_in_a_summary_is_escaped_not_executed():
    hostile = Brief(
        topic="t",
        generated_on=date(2026, 11, 12),
        intro="intro",
        items=[
            Item(
                title="<script>alert(1)</script>",
                summary="<img src=x onerror=alert(1)>",
                source_url="https://example.com/a",
                source_name="Example",
                published=None,
            )
        ],
        outro="outro",
    )
    html = _render_template(hostile, "classic")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html


def test_date_unknown_badge_shown_for_items_with_no_published_date():
    html = _render_template(FIXED_BRIEF, "classic")
    assert "date unknown" in html


def test_sources_list_links_every_item():
    html = _render_template(FIXED_BRIEF, "classic")
    for item in FIXED_BRIEF.items:
        assert f'href="{item.source_url}"' in html


def test_compact_template_uses_the_compact_css_class():
    html = _render_template(FIXED_BRIEF, "compact")
    assert 'class="compact"' in html


def test_no_hardcoded_year_in_task_yaml():
    """The old code hardcoded '2025' in tasks.yaml; today's date must come
    from code (newsletter/pipeline.py injecting {today}), never a literal
    year baked into the prompt."""
    tasks_yaml = Path(__file__).parent.parent / "newsletter" / "config" / "tasks.yaml"
    content = tasks_yaml.read_text(encoding="utf-8")
    assert not re.search(r"\b20\d{2}\b", content), (
        "tasks.yaml must not contain a hardcoded year; found one in: " + content
    )
