"""Capture which URLs the search tool actually returned during a crew run.

Issue 3's research guardrail needs a ground truth set of URLs it can check
every item's source_url against, otherwise a fabricated URL can't be told
apart from a real one. We get that ground truth by listening on CrewAI's
event bus for ToolUsageFinishedEvent, rather than trusting what the agent
claims to have found.

Confirmed on crewai==1.15.40 by reading
crewai/events/types/tool_usage_events.py: ToolUsageFinishedEvent carries
`tool_name: str` and `output: Any` (the tool's raw return value), plus
`failure: ToolFailure | None` set when the tool reported it did not
succeed. We only record URLs from finished, non-failed calls to the named
search tool.
"""

from __future__ import annotations

import re
from typing import Any

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.utilities.string_utils import sanitize_tool_name

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


def _extract_urls(output: Any) -> set[str]:
    """Pull every http(s) URL out of a tool's output, whatever shape it is in.

    Real SerperDevTool output is a dict of result lists (each item has a
    'link' key); our FakeSearchTool returns a list of dicts with the same
    shape. Walking generically and regex-scanning strings covers both
    without coupling to either tool's exact schema.
    """
    urls: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            urls.update(_URL_RE.findall(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple, set)):
            for value in node:
                walk(value)

    walk(output)
    return urls


class ProvenanceTracker(BaseEventListener):
    """Records the set of URLs returned by a named tool during one run.

    The event bus (`crewai_event_bus`) is a process-wide singleton, so two
    `generate()` calls running concurrently in different threads would both
    receive every ToolUsageFinishedEvent if we filtered only on tool name.
    To keep concurrent runs from leaking URLs into each other's guardrail
    check, callers must pass `task_ids`: the id(s) of the research task
    belonging to *this* run's Task objects (each Task gets a fresh uuid),
    and events are filtered on `event.task_id` first. Callers must also
    call `.detach()` (or use as a context manager) when the run finishes,
    which calls `crewai_event_bus.off(...)` so the handler stops receiving
    events for runs that start after it.
    """

    def __init__(
        self,
        task_ids: set[str],
        tool_names: set[str] | None = None,
    ) -> None:
        if not task_ids:
            raise ValueError("ProvenanceTracker requires at least one task id to scope to")
        self.task_ids = task_ids
        # Stored pre-sanitized: confirmed by reading crewai/tools/tool_usage.py
        # that every call site builds ToolUsageFinishedEvent with
        # `tool_name=sanitize_tool_name(tool.name)` (crewai/utilities/
        # string_utils.py: lowercases, splits camelCase, replaces anything
        # outside a-z0-9_ with "_"), never the tool's own human-readable
        # `.name`. So "Search the internet with Serper" is never what
        # arrives on the event; "search_the_internet_with_serper" is. This
        # was found by instrumenting a real kickoff() and printing the
        # event's actual `tool_name` - the human-readable name matched
        # nothing and `returned_urls` stayed empty, failing every item.
        self.tool_names = {
            sanitize_tool_name(name)
            for name in (tool_names or {"Search the internet with Serper", "search_news"})
        }
        self.returned_urls: set[str] = set()
        self.calls = 0
        self._handler: Any = None
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        def _on_tool_finished(source: Any, event: ToolUsageFinishedEvent) -> None:
            if event.task_id not in self.task_ids:
                return
            if event.tool_name not in self.tool_names:
                return
            if event.failure is not None:
                return
            self.calls += 1
            self.returned_urls.update(_extract_urls(event.output))

        crewai_event_bus.on(ToolUsageFinishedEvent)(_on_tool_finished)
        self._handler = _on_tool_finished
        self._bus = crewai_event_bus

    def detach(self) -> None:
        """Stop receiving events. Call this once the run is done."""
        if self._handler is not None:
            self._bus.off(ToolUsageFinishedEvent, self._handler)
            self._handler = None

    def __enter__(self) -> ProvenanceTracker:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.detach()
