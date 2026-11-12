"""Quality checks run against one generated Brief for the eval harness.

Each check returns a CheckResult (passed: bool, detail: str) rather than
raising, so evals/run.py can run every check for a fixture even after one
fails, and report all of them in the markdown summary.

These are deliberately independent of newsletter.guardrails: the guardrail
runs *during* the crew, gating what the writer is allowed to keep; these
checks run *after*, against the finished Brief, as an acceptance check on
the whole pipeline's output (including a real LLM's prose, which the
guardrail does not evaluate at all).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from pypdf import PdfReader

from newsletter.guardrails import domain_of, normalize_url
from newsletter.models import Brief


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_citation_validity(brief: Brief, fixture_urls: set[str]) -> CheckResult:
    """Every item's source_url must be one of the fixture's real URLs.

    Mirrors newsletter.guardrails' exact-match-after-normalization rule: a
    fabricated URL that is a substring or near-miss of a real one must not
    pass. If the writer paraphrased or invented a URL not present in the
    fixture, that is exactly what this check exists to catch.
    """
    normalized_fixture = {normalize_url(u) for u in fixture_urls}
    bad = [
        item.source_url
        for item in brief.items
        if normalize_url(item.source_url) not in normalized_fixture
    ]
    if bad:
        return CheckResult(
            "citation_validity",
            False,
            f"{len(bad)} item(s) cite a URL not in the fixture's real results: {bad}",
        )
    return CheckResult("citation_validity", True, f"all {len(brief.items)} citations verified")


def check_freshness(brief: Brief, today: date, window_days: int = 30) -> CheckResult:
    """Every item with a known published date must fall inside the window.

    Unverified (None) dates are allowed through here, same as the research
    guardrail: this checks dates the brief actually claims, not whether
    every item has one.
    """
    earliest = today - timedelta(days=window_days)
    stale = [
        (item.title, item.published)
        for item in brief.items
        if item.published is not None and not (earliest <= item.published <= today)
    ]
    if stale:
        return CheckResult(
            "freshness",
            False,
            f"{len(stale)} item(s) outside the {window_days}-day window: {stale}",
        )
    return CheckResult("freshness", True, f"all dated items inside the {window_days}-day window")


def check_source_diversity(brief: Brief, max_per_domain: int = 2) -> CheckResult:
    """No single domain should supply more than max_per_domain items."""
    counts: dict[str, int] = {}
    for item in brief.items:
        domain = domain_of(item.source_url)
        counts[domain] = counts.get(domain, 0) + 1
    overflowing = {d: c for d, c in counts.items() if c > max_per_domain}
    if overflowing:
        return CheckResult(
            "source_diversity",
            False,
            f"domain(s) over the cap of {max_per_domain}: {overflowing}",
        )
    return CheckResult(
        "source_diversity", True, f"{len(counts)} distinct domain(s), none over the cap"
    )


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "is", "are", "was", "were", "be", "been", "that",
    "this", "it", "its", "as", "has", "have", "had", "than", "then",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def check_faithfulness(brief: Brief, source_snippets: dict[str, str], min_overlap: float = 0.15) -> CheckResult:
    """A cheap proxy for "did the writer stick to the source": each item's
    summary must share at least `min_overlap` of its content words with the
    fixture snippet it is attributed to. This will not catch subtle
    misrepresentation, but it does catch an item whose summary is about
    something the cited source snippet never mentions at all, which is the
    failure mode most worth flagging automatically; anything subtler needs
    a human or an LLM-judge pass, out of scope for this offline check.
    """
    weak: list[tuple[str, float]] = []
    checked = 0
    for item in brief.items:
        snippet = source_snippets.get(normalize_url(item.source_url))
        if snippet is None:
            continue
        checked += 1
        source_words = _content_words(snippet)
        summary_words = _content_words(item.summary)
        if not source_words or not summary_words:
            continue
        overlap = len(source_words & summary_words) / len(source_words)
        if overlap < min_overlap:
            weak.append((item.title, round(overlap, 2)))
    if weak:
        return CheckResult(
            "faithfulness",
            False,
            f"{len(weak)}/{checked} item(s) below {min_overlap:.0%} word overlap with their source: {weak}",
        )
    return CheckResult("faithfulness", True, f"{checked} item(s) checked, all above {min_overlap:.0%} overlap")


def check_layout(pdf_path: Path, brief: Brief, max_pages: int = 4) -> CheckResult:
    """The rendered PDF must be a real, readable PDF with every item's
    title present as extractable text, and not run away in page count."""
    if not pdf_path.exists():
        return CheckResult("layout", False, f"{pdf_path} does not exist")
    data = pdf_path.read_bytes()
    if data[:4] != b"%PDF":
        return CheckResult("layout", False, "file does not start with the %PDF header")
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) == 0 or len(reader.pages) > max_pages:
        return CheckResult(
            "layout", False, f"page count {len(reader.pages)} outside 1..{max_pages}"
        )
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    missing_titles = [item.title for item in brief.items if item.title not in full_text]
    if missing_titles:
        return CheckResult("layout", False, f"title(s) not found as text in the PDF: {missing_titles}")
    return CheckResult("layout", True, f"{len(reader.pages)} page(s), all titles present as text")


def run_all_checks(
    *,
    brief: Brief,
    pdf_path: Path,
    fixture_urls: set[str],
    source_snippets: dict[str, str],
    today: date,
    window_days: int = 30,
    max_per_domain: int = 2,
) -> list[CheckResult]:
    return [
        check_citation_validity(brief, fixture_urls),
        check_freshness(brief, today, window_days),
        check_source_diversity(brief, max_per_domain),
        check_faithfulness(brief, source_snippets),
        check_layout(pdf_path, brief),
    ]
