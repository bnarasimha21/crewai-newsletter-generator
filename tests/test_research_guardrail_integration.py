"""Issue 3: adversarial tests for the research guardrail.

Exercises newsletter.guardrails.make_research_guardrail end to end (building
a real TaskOutput, running the real check_research callable) rather than
unit-testing normalize_url/domain_of in isolation, since the thing that can
actually be gamed is their combination inside check_research.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from crewai.tasks.task_output import TaskOutput

from newsletter.guardrails import domain_of, make_research_guardrail, normalize_title, normalize_url
from newsletter.models import Item, ResearchItems

TODAY = date(2026, 11, 12)


def _output(research: ResearchItems) -> TaskOutput:
    return TaskOutput(description="research", agent="researcher", raw=research.model_dump_json())


def _item(**overrides) -> Item:
    defaults = dict(
        title="Quantum Error Correction Hits New Milestone",
        summary="A breakthrough in fault-tolerant qubits.",
        source_url="https://www.technews.example.com/quantum-error-correction",
        source_name="Tech News Example",
        published=TODAY - timedelta(days=3),
    )
    defaults.update(overrides)
    return Item(**defaults)


def _guardrail(returned_urls: set[str], **kwargs):
    return make_research_guardrail(returned_urls=returned_urls, today=TODAY, **kwargs)


REAL_URL = "https://www.technews.example.com/quantum-error-correction"


def test_happy_path_passes():
    items = [
        _item(),
        _item(title="Startup Raises Funding", source_url="https://venture.example.com/a"),
        _item(title="Lab Opens Research Center", source_url="https://labs.example.org/b"),
    ]
    returned = {i.source_url for i in items}
    check = _guardrail(returned)
    ok, result = check(_output(ResearchItems(topic="quantum computing", items=items)))
    assert ok is True
    assert len(result.pydantic.items) == 3


def test_fabricated_url_that_is_a_substring_of_a_real_one_is_rejected():
    """A fabricated URL ('.../quantum') that is a strict prefix of the real,
    returned URL ('.../quantum-error-correction') must never pass: exact
    match only, never substring/prefix containment."""
    fabricated = "https://www.technews.example.com/quantum"
    assert fabricated in REAL_URL  # sanity: it really is a substring
    items = [
        _item(source_url=fabricated),
        _item(title="B", source_url="https://venture.example.com/a"),
        _item(title="C", source_url="https://labs.example.org/b"),
    ]
    check = _guardrail({REAL_URL, "https://venture.example.com/a", "https://labs.example.org/b"})
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is False
    assert "quantum" in result or "not among" in result


@pytest.mark.parametrize(
    "variant",
    [
        "HTTPS://WWW.TECHNEWS.EXAMPLE.COM/quantum-error-correction",
        "https://www.technews.example.com/quantum-error-correction/",
    ],
)
def test_casing_and_trailing_slash_variants_are_treated_as_the_same_url(variant):
    """normalize_url lowercases scheme/host and strips a trailing slash, so
    these variants of the real URL must still pass."""
    items = [
        _item(source_url=variant),
        _item(title="B", source_url="https://venture.example.com/a"),
        _item(title="C", source_url="https://labs.example.org/b"),
    ]
    check = _guardrail({REAL_URL, "https://venture.example.com/a", "https://labs.example.org/b"})
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is True
    assert len(result.pydantic.items) == 3


def test_near_duplicate_titles_are_deduplicated():
    """Differently-punctuated/cased versions of the same title must collapse
    to one kept item, not two."""
    items = [
        _item(title="AI Breakthrough!"),
        _item(
            title="ai breakthrough",
            source_url="https://venture.example.com/a",
        ),
        _item(title="C", source_url="https://labs.example.org/b"),
        _item(title="D", source_url="https://another.example.net/c"),
    ]
    returned = {i.source_url for i in items}
    check = _guardrail(returned, min_items=3)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is True
    titles = {normalize_title(i.title) for i in result.pydantic.items}
    assert "ai breakthrough" in titles
    assert len(result.pydantic.items) == 3  # the duplicate was dropped, not both kept


@pytest.mark.parametrize(
    "age_days,should_pass",
    [
        (30, True),   # exactly at the boundary: inclusive
        (29, True),   # one day inside the window
        (31, False),  # one day outside the window
    ],
)
def test_freshness_window_boundary_is_inclusive(age_days, should_pass):
    items = [
        _item(published=TODAY - timedelta(days=age_days)),
        _item(title="B", source_url="https://venture.example.com/a"),
        _item(title="C", source_url="https://labs.example.org/b"),
    ]
    returned = {i.source_url for i in items}
    check = _guardrail(returned, window_days=30)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    if should_pass:
        assert ok is True
        assert len(result.pydantic.items) == 3
    else:
        assert ok is False
        assert len(result) and "freshness" in result.lower() or "window" in result.lower()


def test_unverified_date_is_allowed_through_not_rejected():
    items = [
        _item(published=None),
        _item(title="B", source_url="https://venture.example.com/a"),
        _item(title="C", source_url="https://labs.example.org/b"),
    ]
    returned = {i.source_url for i in items}
    check = _guardrail(returned)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is True
    assert any(i.published is None for i in result.pydantic.items)


def test_same_domain_different_subdomain_counts_as_a_different_domain():
    """blog.example.com and example.com are editorially distinct, so both
    should be able to contribute up to max_per_domain items independently."""
    items = [
        _item(source_url="https://example.com/a", title="A"),
        _item(source_url="https://example.com/b", title="B"),
        _item(source_url="https://blog.example.com/c", title="C"),
        _item(source_url="https://blog.example.com/d", title="D"),
    ]
    returned = {i.source_url for i in items}
    check = _guardrail(returned, max_per_domain=2, min_items=4)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is True
    assert len(result.pydantic.items) == 4
    assert domain_of("https://example.com/a") == "example.com"
    assert domain_of("https://blog.example.com/c") == "blog.example.com"


def test_www_prefix_counts_as_the_same_domain_as_bare_domain():
    items = [
        _item(source_url="https://www.example.com/a", title="A"),
        _item(source_url="https://example.com/b", title="B"),
        _item(source_url="https://example.com/c", title="C"),
        _item(source_url="https://venture.example.net/d", title="D"),
    ]
    returned = {i.source_url for i in items}
    check = _guardrail(returned, max_per_domain=2, min_items=3)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is True
    # www.example.com + example.com + example.com = 3 items on one domain,
    # over the cap of 2, so the third must be dropped.
    kept_domains = [domain_of(i.source_url) for i in result.pydantic.items]
    assert kept_domains.count("example.com") == 2


def test_domain_cap_failure_names_the_domain():
    items = [
        _item(source_url="https://example.com/a", title="A"),
        _item(source_url="https://example.com/b", title="B"),
        _item(source_url="https://example.com/c", title="C"),
    ]
    returned = {i.source_url for i in items}
    check = _guardrail(returned, max_per_domain=2, min_items=3)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is False
    assert "example.com" in result


def test_below_min_items_fails_with_detail():
    items = [_item()]
    check = _guardrail({i.source_url for i in items}, min_items=3)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is False
    assert "Only 1 of 3" in result


def test_invalid_json_raw_output_fails_gracefully():
    check = _guardrail({REAL_URL})
    bad_output = TaskOutput(description="research", agent="researcher", raw="not json at all")
    ok, result = check(bad_output)
    assert ok is False
    assert "research output was not valid JSON" in result


def test_normalize_url_strips_fragment_but_keeps_query():
    assert normalize_url("https://Example.com/Page#section") == "https://example.com/Page"
    assert normalize_url("https://example.com/page?x=1") == "https://example.com/page?x=1"


def test_returned_urls_set_is_read_live_not_snapshotted():
    """The guardrail must read `returned_urls` fresh on every call, since a
    retry can add more URLs to the same mutable set after the guardrail was
    built (see the module docstring in newsletter/guardrails.py)."""
    returned: set[str] = set()
    check = _guardrail(returned, min_items=1)
    items = [_item()]
    ok, _ = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is False  # nothing in `returned` yet

    returned.add(REAL_URL)
    ok, result = check(_output(ResearchItems(topic="t", items=items)))
    assert ok is True
    assert len(result.pydantic.items) == 1
