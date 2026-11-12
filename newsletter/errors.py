"""Typed errors surfaced by the pipeline, distinct from a generic exception
so the UI and CLI can show a specific message instead of "something failed".
"""


class PipelineError(Exception):
    """Base class for errors raised by newsletter.pipeline.generate."""


class SearchFailed(PipelineError):
    """The research stage could not produce enough verifiable items.

    Raised when the research guardrail rejects every attempt, or the final
    accepted brief has fewer than the configured minimum item count.
    """


class RenderFailed(PipelineError):
    """WeasyPrint could not turn the generated HTML into a PDF."""

    def __init__(self, message: str, original_message: str | None = None) -> None:
        super().__init__(message)
        self.original_message = original_message if original_message is not None else message


class Timeout(PipelineError):
    """The crew did not finish within the configured timeout."""
