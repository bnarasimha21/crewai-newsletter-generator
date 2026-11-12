# Newsletter Generator with CrewAI

Generates a researched, structured newsletter for any topic using a
two-agent CrewAI crew (researcher, writer), renders it to HTML and PDF, and
shows it in a Streamlit UI or a CLI. Research is gated by a guardrail that
rejects fabricated citations, stale items and over-concentrated sources
before the writer ever sees them; PDF rendering runs under a deny-all
network/filesystem policy, so a newsletter can never be used to make the
renderer fetch an internal URL or a local file.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bnarasimha21/crewai-newsletter-generator.git
cd crewai-newsletter-generator
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt       # runtime only
pip install -r requirements-dev.txt   # + pytest, ruff, pypdf (adds requirements.txt too)
```

4. WeasyPrint (used for PDF rendering) needs Pango on the system, not a
   separate PDF engine like wkhtmltopdf:

**macOS:**
```bash
brew install pango
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0
```

**Windows:** follow WeasyPrint's own install notes at
https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows

5. Set the API keys the crew needs (as environment variables), or use the
   offline fake mode below if you just want to try the pipeline:

```bash
export OPENAI_API_KEY=...   # default LLM used by the CrewAI agents
export SERPER_API_KEY=...   # web search tool used by the researcher agent
```

## Project Structure

```
crewai-newsletter-generator/
├── newsletter/
│   ├── pipeline.py        # generate(topic) -> NewsletterResult; the one entry point
│   ├── crew.py             # researcher + writer agents/tasks (no LLM-driven PDF step)
│   ├── models.py           # Item / ResearchItems / Brief (pydantic)
│   ├── guardrails.py       # research guardrail: citation validity, freshness, dedupe, domain cap
│   ├── provenance.py       # tracks which URLs the search tool actually returned, per run
│   ├── render.py           # WeasyPrint HTML -> PDF under a deny-all url_fetcher
│   ├── history.py          # SQLite run history + cost estimation
│   ├── trace.py            # per-run token/latency tracing
│   ├── pricing.yaml        # DigitalOcean serverless inference pricing used by history.py
│   ├── fakes.py            # FakeLLM / FakeSearchTool, shared by tests, CLI fake mode, evals
│   ├── __main__.py         # `python -m newsletter "topic" ...` CLI
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   └── templates/
│       ├── newsletter.html.j2
│       └── print.css
├── evals/
│   ├── run.py               # `python -m evals.run --llm fake|do_serverless|claude_cli`
│   ├── checks.py            # citation validity, freshness, source diversity, faithfulness, layout
│   ├── llms.py               # real-LLM backends for do_serverless / claude_cli eval modes
│   ├── fixtures/             # 6 fixed topics with fixed search results
│   └── results/              # one markdown report per eval run, <date>-<model>.md
├── tests/                    # pytest, see "Testing" below
├── app.py                    # Streamlit UI
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically http://localhost:8501), enter a
topic, pick a template, and generate. Past runs are listed from the
SQLite-backed history, not just the current browser session.

### CLI

```bash
python -m newsletter "quantum computing" --template compact --out runs/
```

Prints a JSON summary (run directory, HTML/PDF paths, item count, trace) to
stdout and exits 0 on success, 1 on a pipeline failure, 2 on timeout
(default 300s, override with `--timeout`).

To run the whole pipeline offline with no API key, set `LLM_PROVIDER=fake`:

```bash
LLM_PROVIDER=fake python -m newsletter "quantum computing"
```

## Testing

```bash
pytest -q          # all tests
ruff check .       # lint
```

Tests marked `integration` (most of them) exercise the real crew, render
pipeline, or CLI end to end, offline, via `newsletter.fakes.FakeLLM` and
`FakeSearchTool`: no network call and no API key needed to run the suite.

## Evals

```bash
python -m evals.run --llm fake                       # offline, what CI runs
python -m evals.run --llm do_serverless --model gpt-4o-mini  # needs DIGITALOCEAN_INFERENCE_API_KEY + DIGITALOCEAN_INFERENCE_BASE_URL
python -m evals.run --llm claude_cli                  # needs a locally logged-in `claude` CLI
```

Runs the real two-agent crew over 6 fixed topics (`evals/fixtures/`) with
fixed search results, scores the resulting brief on citation validity,
freshness, source diversity, a word-overlap faithfulness proxy, and PDF
layout, and writes a markdown report to `evals/results/<date>-<model>.md`.
`--llm do_serverless` and `--llm claude_cli` are for local use (they call a
real model); only `--llm fake` runs in CI.

## Configuration

- CrewAI agent and task prompts: `newsletter/config/agents.yaml`, `tasks.yaml`
- Research guardrail thresholds (freshness window, per-domain cap, minimum
  items): passed as keyword arguments to `newsletter.pipeline.generate()`,
  also exposed as CLI flags (`--window-days`, `--max-per-domain`, `--min-items`)
- Run history / cost pricing: `newsletter/pricing.yaml`

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes, with tests (`pytest -q`) and lint (`ruff check .`) passing
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## License

This project is licensed under the MIT License, see the LICENSE file for details.

## Roadmap

- [x] Isolated per-run output directories (no cwd collisions)
- [x] Structured research brief + template-rendered HTML
- [x] Research guardrails (citation validity, freshness, dedupe, domain cap)
- [x] PDF rendering under a deny-all network/filesystem policy
- [x] Offline evals harness with quality checks
- [x] Run history, cost/latency reporting, CLI
- [ ] Custom newsletter templates beyond classic/compact
- [ ] User authentication for the Streamlit UI
- [ ] Export formats beyond HTML/PDF
