"""Research-quality guardrails for the research task.

Design decisions, made explicit because the adversarial-testing lesson from
prior repos in this batch is that a sloppy heuristic here is exactly what
gets gamed:

- URL matching is EXACT STRING EQUALITY after normalization, never a
  substring/prefix check. A fabricated URL that happens to be a substring
  of a real one (e.g. "https://example.com/a" vs the real
  "https://example.com/article") must never pass, so we never use `in`
  on the raw returned-URL set and never check containment either way.
- normalize_url lowercases scheme and host, strips a trailing "/" from the
  path (so ".../page" and ".../page/" are the same URL), strips the
  fragment, and leaves the query string untouched (two URLs that differ
  only by query params are treated as different resources, which is the
  safer default for provenance checking).
- Domain/subdomain handling is a deliberate choice, documented and tested
  both ways in tests/test_research_guardrail_integration.py: a leading
  "www." is stripped (so "www.example.com" and "example.com" count as the
  SAME domain), but any other subdomain is treated as a DIFFERENT domain
  ("blog.example.com" != "example.com"), because a news aggregator and a
  publisher's own blog subdomain are editorially distinct sources even
  when they share a root domain.
- Title normalization lowercases, strips all punctuation, and collapses
  whitespace, so "AI Breakthrough!" and "ai breakthrough" are the same
  title but unrelated titles are not falsely merged.
- The freshness window is inclusive on both ends: a published date exactly
  `window_days` before `today` passes, `window_days + 1` days before fails.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from datetime import date, timedelta
from urllib.parse import urlsplit, urlunsplit

from crewai.tasks.task_output import TaskOutput
from pydantic import ValidationError

from newsletter.models import Item, ResearchItems

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_url(url: str) -> str:
    """Normalize a URL for exact-match comparison (see module docstring)."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def normalize_title(title: str) -> str:
    """Normalize a title for near-duplicate detection (see module docstring)."""
    lowered = title.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", lowered).strip()


def domain_of(url: str) -> str:
    """Extract the comparison domain for the per-domain item cap.

    Strips a leading "www." so "www.example.com" and "example.com" count
    as one domain; any other subdomain ("blog.example.com") is kept as a
    distinct domain. See module docstring for why.
    """
    netloc = urlsplit(url.strip()).netloc.lower()
    netloc = netloc.split("@")[-1]  # drop userinfo if present
    netloc = netloc.split(":")[0]  # drop port
    if netloc.startswith("www."):
        netloc = netloc[len("www."):]
    return netloc


def make_research_guardrail(
    *,
    returned_urls: set[str],
    today: date,
    window_days: int = 30,
    max_per_domain: int = 2,
    min_items: int = 3,
):
    """Build the `guardrail=` callable for the research task.

    `returned_urls` and `today` are captured by closure per-run (never a
    module-level global) so concurrent pipeline runs do not share state.
    Returns a callable `(TaskOutput) -> tuple[bool, Any]` as required by
    crewai.task.Task's guardrail validator (confirmed by reading
    crewai/utilities/guardrail.py: `GuardrailResult.from_tuple` expects
    `(success, data)` where `data` is the validated result on success or
    an error string on failure; returning a TaskOutput as the success data
    replaces the task's output outright, which we use to drop the items
    that failed a per-item check while keeping the ones that passed).
    """
    def check_research(output: TaskOutput) -> tuple[bool, object]:
        # Recomputed on every call (not cached at closure-build time): the
        # guardrail may run again after a retry, and `returned_urls` is a
        # live, mutable set populated by the provenance tracker as the
        # agent makes more tool calls across attempts, so a stale snapshot
        # taken before kickoff would always be empty.
        normalized_returned = {normalize_url(u) for u in returned_urls}
        # `output.pydantic` is NOT populated here: confirmed by reading
        # crewai/task.py's execute path (both sync and async variants) -
        # whenever a guardrail is attached, output_pydantic conversion is
        # skipped at TaskOutput-build time ("elif not self._guardrails and
        # not self._guardrail: ... else: pydantic_output, json_output =
        # None, None"), precisely so the guardrail can inspect/replace the
        # raw text before conversion happens. So we parse `output.raw`
        # ourselves; a real (non-fake) LLM's ReAct "Final Answer:" text also
        # lands here as plain JSON, not as a pre-built pydantic instance.
        try:
            research = ResearchItems.model_validate_json(output.raw)
        except (ValidationError, ValueError) as exc:
            return False, (
                "Guardrail error: research output was not valid JSON matching "
                f"the research items schema: {exc}"
            )

        kept: list[Item] = []
        seen_titles: set[str] = set()
        domain_counts: Counter[str] = Counter()
        errors: list[str] = []

        for item in research.items:
            norm_url = normalize_url(item.source_url)
            if norm_url not in normalized_returned:
                errors.append(
                    f"Item '{item.title}' cites source_url {item.source_url!r} "
                    "which was not among the URLs the search tool actually "
                    "returned. Only use URLs you got back from the search tool."
                )
                continue

            norm_title = normalize_title(item.title)
            if norm_title in seen_titles:
                errors.append(
                    f"Item '{item.title}' is a duplicate (or near-duplicate) "
                    "of a title already used. Remove the duplicate."
                )
                continue

            domain = domain_of(item.source_url)
            if domain_counts[domain] >= max_per_domain:
                errors.append(
                    f"Domain '{domain}' already has {max_per_domain} items; "
                    f"item '{item.title}' exceeds the per-domain limit of "
                    f"{max_per_domain}. Use a different source."
                )
                continue

            if item.published is not None:
                age_days = (today - item.published).days
                if age_days < 0 or age_days > window_days:
                    errors.append(
                        f"Item '{item.title}' is dated {item.published.isoformat()}, "
                        f"outside the {window_days}-day freshness window "
                        f"ending {today.isoformat()}. Drop it or find a newer source."
                    )
                    continue
            # item.published is None: allowed through, flagged "date unknown"
            # by the template rather than rejected here.

            seen_titles.add(norm_title)
            domain_counts[domain] += 1
            kept.append(item)

        if len(kept) < min_items:
            error_detail = " ".join(errors) if errors else "No valid items found."
            return False, (
                f"Only {len(kept)} of {min_items} required verifiable items "
                f"passed the research guardrail. {error_detail}"
            )

        filtered = ResearchItems(topic=research.topic, items=kept)
        new_output = output.model_copy(
            update={
                "pydantic": filtered,
                "raw": filtered.model_dump_json(),
                "json_dict": filtered.model_dump(mode="json"),
            }
        )
        return True, new_output

    return check_research


def freshness_window(today: date, window_days: int = 30) -> tuple[date, date]:
    """The inclusive [start, end] date range considered "fresh"."""
    return today - timedelta(days=window_days), today
