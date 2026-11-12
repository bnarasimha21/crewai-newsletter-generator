"""Deterministic, offline stand-ins for the LLM and the search tool.

Shipped as part of the installed package, not a pytest-only fixture, because
two things outside of pytest need a self-contained "golden path" through the
real two-agent crew with no network and no API key:

- the CLI's `LLM_PROVIDER=fake` mode (see newsletter/__main__.py), used by
  tests/test_cli_integration.py to run the real CLI as a subprocess, and
  available to Narsi for a local dry run with no credentials configured;
- the eval runner's `--llm fake` mode (see evals/run.py), which must work
  offline in CI on every pull request.

`tests/conftest.py` imports `FakeLLM`/`FakeSearchTool` from here rather than
redefining them, then builds its own scripted replies for the
guardrail-violation scenarios (fabricated URL, duplicate titles, stale
dates, per-domain overflow) that this module's single golden path does not
need to cover.

Confirmed on crewai==1.15.40 by reading crewai/utilities/agent_utils.py
and crewai/agents/parser.py: an attached LLM with no callable
`supports_function_calling` method (BaseLLM does not define one; only the
concrete litellm-backed `LLM` class does) makes CrewAI fall back to its
text-based ReAct loop, parsing "Action: <tool>\\nAction Input: <json>" or
"Final Answer: <text>" out of the plain string `llm.call()` returns. So
FakeLLM only ever needs to return ReAct-format text, never implement native
function-calling.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


def _role_key(role: str) -> str:
    """Map an agent's (topic-interpolated) role string to a stable fixture key."""
    lowered = role.lower()
    if "research" in lowered:
        return "researcher"
    if "writ" in lowered:
        return "writer"
    return lowered


class FakeLLM(BaseLLM):
    """A BaseLLM that returns pre-scripted ReAct-format replies per agent role.

    `scripted_replies={"researcher": ["Action: ...\\nAction Input: {...}",
    "Final Answer: {...}"], "writer": ["Final Answer: {...}"]}`.
    Each role keeps its own call counter, so an agent's second call (after
    its tool Observation is appended to the conversation) gets
    `scripted_replies[role][1]`.
    """

    model: str = "fake-llm"
    scripted_replies: dict[str, list[str]] = Field(default_factory=dict)

    _call_counts: dict[str, int] = PrivateAttr(default_factory=dict)
    _recorded_prompts: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any) -> None:
        # BaseLLM has a `mode="before"` model_validator that reads the raw
        # constructor kwargs dict and raises "Model name is required" if
        # "model" is absent from it - a subclass field default never
        # satisfies that check, since before-validators run on the raw
        # input, before defaults are applied. Confirmed by reading
        # crewai/llms/base_llm.py's `_validate_init_fields`. So we inject
        # the default into the kwargs dict itself, here, before `BaseLLM`
        # ever sees it.
        data.setdefault("model", "fake-llm")
        super().__init__(**data)

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        role = getattr(from_agent, "role", "") or ""
        key = _role_key(role)
        self._recorded_prompts.append({"role": key, "raw_role": role, "messages": messages})

        idx = self._call_counts.get(key, 0)
        self._call_counts[key] = idx + 1

        replies = self.scripted_replies.get(key)
        if not replies:
            raise AssertionError(f"FakeLLM: no scripted replies registered for role {key!r}")
        if idx >= len(replies):
            raise AssertionError(
                f"FakeLLM: role {key!r} called {idx + 1} times but only "
                f"{len(replies)} scripted replies were provided"
            )
        return replies[idx]

    @property
    def call_count(self) -> dict[str, int]:
        return dict(self._call_counts)

    @property
    def recorded_prompts(self) -> list[dict[str, Any]]:
        return list(self._recorded_prompts)

    def total_calls(self) -> int:
        return sum(self._call_counts.values())


class _FakeSearchToolSchema(BaseModel):
    search_query: str = Field(description="The query to search for")


class FakeSearchTool(BaseTool):
    """Stands in for SerperDevTool: returns fixture results, never hits the network.

    Keeps the same tool name as SerperDevTool so ProvenanceTracker's default
    tool_names set still matches it, and so the researcher's prompt (which
    lists available tool names) looks the same shape a real run would see.
    """

    name: str = "Search the internet with Serper"
    description: str = "A tool that can be used to search the internet with a search_query."
    args_schema: type[BaseModel] = _FakeSearchToolSchema

    fixture_results: list[dict[str, Any]] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("search_query") or kwargs.get("query") or ""
        self.calls.append(query)
        return {"news": list(self.fixture_results)}


FIXTURE_SEARCH_RESULTS: list[dict[str, Any]] = [
    {
        "title": "Quantum Error Correction Hits New Milestone",
        "link": "https://www.technews.example.com/quantum-error-correction",
        "snippet": "Researchers report a breakthrough in fault-tolerant qubits.",
        "date": "3 days ago",
        "source": "Tech News Example",
    },
    {
        "title": "Startup Raises Funding For Quantum Chips",
        "link": "https://venture.example.com/quantum-chip-funding",
        "snippet": "A new chipmaker closed a funding round.",
        "date": "5 days ago",
        "source": "Venture Example",
    },
    {
        "title": "Government Lab Opens Quantum Research Center",
        "link": "https://labs.example.org/quantum-research-center",
        "snippet": "A national lab opens a dedicated quantum computing wing.",
        "date": "1 week ago",
        "source": "Labs Example",
    },
    {
        "title": "Quantum Networking Demo Over 100km Fiber",
        "link": "https://www.technews.example.com/quantum-networking-demo",
        "snippet": "Researchers demonstrate entanglement distribution over fiber.",
        "date": "2 weeks ago",
        "source": "Tech News Example",
    },
]

_RELATIVE_DATE_RE = re.compile(r"(\d+)\s*(day|week|month)s?\s*ago")


def _parse_relative_date(text: str, today: date) -> date:
    """Turn a fixture's "N days/weeks ago" string into a real date near `today`.

    Only used by the golden path builder below; real research/writer tasks
    never see this text (it is template text in fixtures, not a tool return
    value that gets fed to the guardrail directly - the golden path below
    resolves it once, up front, into an ISO date string).
    """
    match = _RELATIVE_DATE_RE.search(text)
    if not match:
        return today
    amount, unit = int(match.group(1)), match.group(2)
    days = {"day": 1, "week": 7, "month": 30}[unit] * amount
    return today - timedelta(days=days)


def build_golden_path(
    topic: str, today: date, *, n_items: int = 4
) -> tuple[FakeLLM, FakeSearchTool]:
    """Build a FakeLLM + FakeSearchTool pair that sails through the real crew.

    Used by the CLI's `LLM_PROVIDER=fake` mode and by `evals/run.py --llm
    fake`. The scripted researcher reply cites only URLs the FakeSearchTool
    itself returns, uses `n_items` distinct domains well under the
    per-domain cap, distinct titles, and dates inside the default 30-day
    freshness window, so it passes newsletter.guardrails.check_research on
    the first attempt with zero retries.
    """
    if not 3 <= n_items <= len(FIXTURE_SEARCH_RESULTS):
        raise ValueError(f"n_items must be between 3 and {len(FIXTURE_SEARCH_RESULTS)}")
    results = FIXTURE_SEARCH_RESULTS[:n_items]
    search_tool = FakeSearchTool(fixture_results=results)

    items = [
        {
            "title": r["title"],
            "summary": r["snippet"],
            "source_url": r["link"],
            "source_name": r["source"],
            "published": _parse_relative_date(r["date"], today).isoformat(),
        }
        for r in results
    ]
    research_json = json.dumps({"topic": topic, "items": items})
    brief_json = json.dumps(
        {
            "topic": topic,
            "generated_on": today.isoformat(),
            "intro": f"Here is what is new in {topic} this week.",
            "items": items,
            "outro": "That is all for this edition.",
        }
    )

    llm = FakeLLM(
        model="fake-llm",
        scripted_replies={
            "researcher": [
                "Thought: I should search for recent news on this topic.\n"
                "Action: Search the internet with Serper\n"
                f'Action Input: {{"search_query": "{topic} news"}}',
                f"Thought: I have enough verified items.\nFinal Answer: {research_json}",
            ],
            "writer": [
                f"Thought: I will summarize the supplied items only.\nFinal Answer: {brief_json}",
            ],
        }
    )
    return llm, search_tool
