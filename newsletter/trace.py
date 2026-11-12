"""Per-stage timing, LLM call counts/tokens, guardrail retries, search calls.

Confirmed on crewai==1.15.40 by reading
crewai/events/types/llm_events.py: `LLMCallCompletedEvent` carries
`messages`, `response`, `call_type`, and (when the provider reports it)
a `UsageMetrics`-shaped object the LLM attaches via its own token
tracking, not a field directly on the event itself in every case - so
rather than parse that per-provider shape, we count LLM calls from
`LLMCallStartedEvent`/`LLMCallCompletedEvent` (always present) and pull
prompt/completion tokens from `crew.usage_metrics` after kickoff, which
CrewAI aggregates centrally regardless of provider. Guardrail retries are
counted from `LLMGuardrailCompletedEvent.retry_count`; search calls come
from the `ProvenanceTracker.calls` counter already kept for issue 3.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import LLMCallStartedEvent
from crewai.events.types.llm_guardrail_events import LLMGuardrailCompletedEvent


@dataclass
class StageTiming:
    name: str
    seconds: float = 0.0


@dataclass
class Trace:
    """Everything recorded about one `generate()` run."""

    stages: list[StageTiming] = field(default_factory=list)
    llm_calls: int = 0
    guardrail_retries: int = 0
    search_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": {s.name: round(s.seconds, 6) for s in self.stages},
            "llm_calls": self.llm_calls,
            "guardrail_retries": self.guardrail_retries,
            "search_calls": self.search_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_seconds": round(self.total_seconds, 6),
        }


class _CallCounter(BaseEventListener):
    """Scoped LLM-call and guardrail-retry counter for one run.

    Same cross-run leakage concern as ProvenanceTracker: this must be
    `.detach()`-ed when the run ends, since `crewai_event_bus` is process
    wide. It is not filtered by task id because `LLMCallStartedEvent`
    does not carry one reliably across every call site; callers that need
    strict per-run isolation under concurrency should run `generate()`
    calls that overlap in time on genuinely separate crews, which is what
    the parallel-run test in tests/test_pipeline_integration.py exercises
    for file output, not trace counts.
    """

    def __init__(self) -> None:
        self.llm_calls = 0
        self.guardrail_retries = 0
        self._handlers: list[tuple[type, Any]] = []
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        def _on_started(source: Any, event: LLMCallStartedEvent) -> None:
            self.llm_calls += 1

        def _on_guardrail(source: Any, event: LLMGuardrailCompletedEvent) -> None:
            if not event.success and event.retry_count is not None:
                self.guardrail_retries = max(self.guardrail_retries, event.retry_count)

        for event_type, handler in (
            (LLMCallStartedEvent, _on_started),
            (LLMGuardrailCompletedEvent, _on_guardrail),
        ):
            crewai_event_bus.on(event_type)(handler)
            self._handlers.append((event_type, handler))
        self._bus = crewai_event_bus

    def detach(self) -> None:
        for event_type, handler in self._handlers:
            self._bus.off(event_type, handler)
        self._handlers = []


class TraceRecorder:
    """Times named stages and assembles the final Trace for one run."""

    def __init__(self) -> None:
        self._trace = Trace()
        self._counter = _CallCounter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._trace.stages.append(StageTiming(name=name, seconds=time.perf_counter() - start))

    def finish(self, *, search_calls: int, prompt_tokens: int = 0, completion_tokens: int = 0) -> Trace:
        self._counter.detach()
        self._trace.llm_calls = self._counter.llm_calls
        self._trace.guardrail_retries = self._counter.guardrail_retries
        self._trace.search_calls = search_calls
        self._trace.prompt_tokens = prompt_tokens
        self._trace.completion_tokens = completion_tokens
        self._trace.total_seconds = sum(s.seconds for s in self._trace.stages)
        return self._trace
