"""Real-LLM backends for the eval harness's --llm do_serverless and
--llm claude_cli modes. --llm fake uses newsletter.fakes.FakeLLM directly
and needs nothing from this module.

Both backends build a crewai BaseLLM (or subclass) that newsletter.crew's
NewsletterCrew accepts via its existing `llm=` parameter, so no pipeline
code needs to know which mode produced it.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel


class MissingCredentialsError(RuntimeError):
    """Raised when a real-LLM eval mode is selected but its required
    environment variable is not set. Never read or print the value of an
    unset credential; just name which variable is missing."""


def build_do_serverless_llm(model: str | None = None) -> BaseLLM:
    """A crewai LLM pointed at DigitalOcean's serverless inference endpoint.

    Requires DIGITALOCEAN_INFERENCE_API_KEY and, since the exact base URL
    for the GradientAI/Inference Hub OpenAI-compatible endpoint is account-
    and region-specific and changes over time, DIGITALOCEAN_INFERENCE_BASE_URL
    as well: we do not guess or hardcode one here, to avoid silently sending
    requests to a wrong or stale host.

    `model` defaults to newsletter/pricing.yaml's default_model
    (gpt-4o-mini) so cost estimation lines up with what actually ran.
    """
    from crewai import LLM

    api_key = os.environ.get("DIGITALOCEAN_INFERENCE_API_KEY")
    base_url = os.environ.get("DIGITALOCEAN_INFERENCE_BASE_URL")
    if not api_key:
        raise MissingCredentialsError(
            "DIGITALOCEAN_INFERENCE_API_KEY is not set; required for --llm do_serverless"
        )
    if not base_url:
        raise MissingCredentialsError(
            "DIGITALOCEAN_INFERENCE_BASE_URL is not set; required for --llm do_serverless "
            "(the OpenAI-compatible base URL for your DigitalOcean GradientAI/Inference Hub endpoint)"
        )
    resolved_model = model or "gpt-4o-mini"
    return LLM(
        model=f"openai/{resolved_model}",
        base_url=base_url,
        api_key=api_key,
    )


# Every tool name the locally installed `claude` CLI could grant itself in
# print mode, explicitly disallowed below. Without this, a `claude -p`
# subprocess run as an eval "LLM" is a full agent with real Bash/Write/Edit
# access on this machine: the researcher/writer prompts CrewAI builds talk
# about "using tools" (its own ReAct-format search tool), and nothing stops
# Claude from reaching for its *own* Bash tool instead, or hanging the
# subprocess on a permission prompt that never arrives non-interactively.
# Confirmed by reproducing both failure modes directly: `claude -p "Try to
# run the bash command 'echo pwned' using your Bash tool..."` with no
# --disallowedTools actually runs it; passing --disallowedTools as separate
# argv tokens (not one space-joined string) blocks it and the CLI reports
# "this session has no Bash tool" instead.
_DISALLOWED_CLI_TOOLS = [
    "Bash", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
    "WebFetch", "WebSearch", "Agent",
]


class ClaudeCliLLM(BaseLLM):
    """Shells out to the locally installed, already-authenticated `claude`
    CLI instead of calling an API directly, so the eval harness can use a
    real Claude model with no API key: just `claude` logged in on this
    machine. Confirmed via `crewai/llms/base_llm.py`: BaseLLM only requires
    a `call()` method; no `supports_function_calling`, so CrewAI falls back
    to its text-based ReAct loop exactly like FakeLLM does, meaning this
    only ever needs to return plain text, never implement tool-calling.
    """

    model: str = "claude-cli"
    cli_path: str = "claude"
    timeout_seconds: float = 120.0

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
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = "\n\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
                if isinstance(m, dict)
            )
        try:
            result = subprocess.run(
                [
                    self.cli_path,
                    "-p",
                    prompt,
                    "--output-format",
                    "text",
                    "--disallowedTools",
                    *_DISALLOWED_CLI_TOOLS,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MissingCredentialsError(
                f"the '{self.cli_path}' CLI was not found on PATH; required for --llm claude_cli"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {result.returncode}: {result.stderr.strip()[:500]}"
            )
        return result.stdout.strip()


def build_claude_cli_llm() -> BaseLLM:
    return ClaudeCliLLM()
