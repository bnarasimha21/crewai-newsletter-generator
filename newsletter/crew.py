"""The two-agent crew: researcher and writer. No LLM-driven PDF step.

Dropped from the original src/crew.py: the `pdf_generator` agent and
`html_to_pdf_task` (file conversion is deterministic, not an LLM's job -
see newsletter/render.py), and the unused `before_kickoff` hook that added
`inputs['extra_data']`, which no prompt ever referenced.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from newsletter.guardrails import make_research_guardrail
from newsletter.models import Brief, ResearchItems

DEFAULT_SEARCH_N_RESULTS = 10
DEFAULT_SEARCH_TYPE = "news"
DEFAULT_LOCALE = "en"
DEFAULT_COUNTRY = "us"


def build_search_tool(
    *,
    n_results: int = DEFAULT_SEARCH_N_RESULTS,
    search_type: str = DEFAULT_SEARCH_TYPE,
    locale: str = DEFAULT_LOCALE,
    country: str = DEFAULT_COUNTRY,
) -> SerperDevTool:
    """Configure the search tool explicitly instead of relying on defaults.

    Issue 3: the original code used `SerperDevTool()` with every parameter
    left at its default (10 results, generic "search" type, no locale/
    country), which also means no recency signal at all. `search_type="news"`
    biases results toward dated, recent items, which the freshness guardrail
    then checks for real.
    """
    return SerperDevTool(
        n_results=n_results,
        search_type=search_type,
        locale=locale,
        country=country,
    )


@CrewBase
class NewsletterCrew:
    """Researcher + writer crew producing a structured Brief."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        *,
        topic: str,
        today: date,
        llm: Any | None = None,
        search_tool: Any | None = None,
        window_days: int = 30,
        max_per_domain: int = 2,
        min_items: int = 3,
        guardrail_max_retries: int = 2,
    ) -> None:
        self.topic = topic
        self.today = today
        self.llm = llm
        self.search_tool = search_tool or build_search_tool()
        self.window_days = window_days
        self.max_per_domain = max_per_domain
        self.min_items = min_items
        self.guardrail_max_retries = guardrail_max_retries
        self.provenance_tracker: Any | None = None
        # Deliberately no `super().__init__()` here: `@CrewBase` applies
        # `CrewBaseMeta` as this class's metaclass via decorator syntax
        # (crewai/project/crew_base.py's `_CrewBaseType.__new__`), which
        # replaces the class object created by this `class NewsletterCrew:`
        # statement with a new one. That breaks the zero-arg `super()` form:
        # it resolves via the `__class__` closure cell captured at function-
        # definition time, which still points at the pre-replacement class,
        # not `self.__class__` - so `super()` raises "obj must be an
        # instance or subtype of type" (confirmed by reproducing it: this
        # exact line raised that error before being removed). NewsletterCrew
        # has no explicit base beyond `object`, whose `__init__` does
        # nothing we need, so skipping the call is safe.

    def inputs(self) -> dict[str, str]:
        """The `{topic}` / `{today}` values injected into every prompt."""
        return {"topic": self.topic, "today": self.today.isoformat()}

    @agent
    def researcher(self) -> Agent:
        kwargs: dict[str, Any] = {
            "config": self.agents_config["researcher"],
            "tools": [self.search_tool],
            "verbose": True,
        }
        if self.llm is not None:
            kwargs["llm"] = self.llm
        return Agent(**kwargs)

    @agent
    def writer(self) -> Agent:
        kwargs: dict[str, Any] = {
            "config": self.agents_config["writer"],
            "verbose": True,
        }
        if self.llm is not None:
            kwargs["llm"] = self.llm
        return Agent(**kwargs)

    @task
    def research_task(self) -> Task:
        from newsletter.provenance import ProvenanceTracker

        research_task_obj = Task(
            config=self.tasks_config["research_task"],
            output_pydantic=ResearchItems,
        )
        # Scope the provenance tracker to this task's own id so concurrent
        # pipeline runs never see each other's tool-returned URLs (see
        # newsletter/provenance.py docstring).
        self.provenance_tracker = ProvenanceTracker(task_ids={str(research_task_obj.id)})
        research_task_obj.guardrail = make_research_guardrail(
            returned_urls=self.provenance_tracker.returned_urls,
            today=self.today,
            window_days=self.window_days,
            max_per_domain=self.max_per_domain,
            min_items=self.min_items,
        )
        research_task_obj.guardrail_max_retries = self.guardrail_max_retries
        return research_task_obj

    @task
    def writer_task(self) -> Task:
        return Task(
            config=self.tasks_config["writer_task"],
            output_pydantic=Brief,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def detach_provenance(self) -> None:
        """Stop listening on the (process-wide) event bus. Call after kickoff."""
        if self.provenance_tracker is not None:
            self.provenance_tracker.detach()
