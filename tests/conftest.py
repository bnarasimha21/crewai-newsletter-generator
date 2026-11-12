"""Shared test fixtures.

FakeLLM/FakeSearchTool/FIXTURE_SEARCH_RESULTS themselves live in
newsletter.fakes (shipped in the real package, not here) because the CLI's
`LLM_PROVIDER=fake` mode and evals/run.py's `--llm fake` mode need them
outside of pytest too. This file re-exports them for convenience and adds
pytest-only fixtures plus helpers for building deliberately *broken*
scripted replies (fabricated URL, duplicate titles, stale dates, per-domain
overflow) that the shared golden path in newsletter.fakes does not cover.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import pytest

from newsletter.fakes import (  # noqa: F401 - re-exported for tests
    FIXTURE_SEARCH_RESULTS,
    FakeLLM,
    FakeSearchTool,
    build_golden_path,
)


@pytest.fixture
def today() -> date:
    """A fixed "today" so freshness-window tests are not calendar-dependent."""
    return date(2026, 11, 12)


def make_research_llm(
    research_replies: list[str], writer_replies: list[str] | None = None
) -> FakeLLM:
    """Build a FakeLLM with explicit researcher/writer scripts for one test."""
    scripted = {"researcher": research_replies}
    if writer_replies is not None:
        scripted["writer"] = writer_replies
    # `model=` must be passed explicitly: BaseLLM's own mode="before"
    # model_validator rejects a missing "model" key in the raw input dict
    # before pydantic ever applies FakeLLM's `model: str = "fake-llm"` field
    # default (see newsletter/fakes.py's build_golden_path, which hit the
    # same ValidationError until it started passing model= explicitly too).
    return FakeLLM(model="fake-llm", scripted_replies=scripted)


def research_final_answer(topic: str, items: list[dict[str, Any]]) -> str:
    """A "Final Answer:" ReAct reply carrying a ResearchItems-shaped JSON body."""
    return "Thought: Done.\nFinal Answer: " + json.dumps({"topic": topic, "items": items})


def search_action(query: str) -> str:
    """An "Action:" ReAct reply that calls the (fake) search tool once."""
    return (
        "Thought: Let me search for recent items.\n"
        "Action: Search the internet with Serper\n"
        f'Action Input: {{"search_query": "{query}"}}'
    )


def writer_final_answer(topic: str, today_: date, items: list[dict[str, Any]]) -> str:
    """A "Final Answer:" ReAct reply carrying a Brief-shaped JSON body."""
    body = {
        "topic": topic,
        "generated_on": today_.isoformat(),
        "intro": f"Here is what is new in {topic}.",
        "items": items,
        "outro": "That is all for this edition.",
    }
    return "Thought: Done.\nFinal Answer: " + json.dumps(body)


def days_ago(today_: date, n: int) -> str:
    return (today_ - timedelta(days=n)).isoformat()
