# DeepResearch AI

An autonomous research agent. Given a question, it plans the research, searches the web, collects pages and PDFs,
indexes them for retrieval-augmented generation (RAG), fact-checks the important claims and writes a structured report
with traceable citations.

**Status: phase 1 of 11 (project setup).** Research is not available yet; see the [roadmap](#roadmap).

## Design principles

- **Zero cost by default.** The default language model runs locally with [Ollama](https://ollama.com/). Paid APIs
  (Anthropic, OpenAI) are refused unless `ALLOW_PAID_APIS=true`, an API key is set and a per-research spending cap
  (`MAX_COST_USD_PER_RESEARCH`) is configured.
- **Traceable answers.** Every important claim is linked to its sources and gets a verification status
  (`VERIFIED`, `PARTIALLY_VERIFIED`, `CONFLICTING`, `UNVERIFIED`). No source, URL, author or date is ever invented.
- **Web content is untrusted data.** Instructions found in pages are never executed; downloads are validated and
  private network addresses are blocked.

## Quick start

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv/Scripts/activate                 # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                   # then adjust .env (never commit it)

python -m app.main                     # check the configuration and create local storage
streamlit run frontend/dashboard.py    # open http://localhost:8501
pytest                                 # run the tests
```

`python -m app.main` prints the active configuration (secrets are only reported as "set" or "not set") and exits with
an error message if the configuration is invalid, for example a paid provider selected without `ALLOW_PAID_APIS=true`.

## Configuration

All settings are environment variables, documented in [.env.example](.env.example) and validated by
[app/config.py](app/config.py). Main groups:

| Group | Settings |
| --- | --- |
| Language model | `LLM_PROVIDER` (`ollama` / `anthropic` / `openai`), model names, API keys, timeout |
| Cost guardrails | `ALLOW_PAID_APIS` (default `false`), `MAX_COST_USD_PER_RESEARCH` |
| Web search | `SEARCH_PROVIDER` (`tavily` / `duckduckgo`), `TAVILY_API_KEY`, results per query |
| Research limits | up to 3 extra research cycles, sources per research, maximum duration |
| Downloads | HTTP timeout, maximum download size |
| RAG | chunk size 800 tokens with 120 overlap, multilingual embedding model, reranker, top-k |

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 1 | Setup: configuration, dependencies, starter interface, tests | done |
| 2 | Web search: isolated search interface, errors, quotas, timeouts, raw results saved | |
| 3 | Document collection: web pages and PDFs, cleaning, deduplication, metadata | |
| 4 | RAG: chunking, embeddings, Qdrant, BM25, hybrid fusion, reranking | |
| 5 | Agents: Planner, Researcher, Analyst, Fact Checker, Writer (Pydantic inputs/outputs) | |
| 6 | Orchestration with LangGraph: routing, iteration limits, checkpointing | |
| 7 | Citations and reports: claim traceability, Markdown and PDF export | |
| 8 | Streamlit dashboard: history, progress, sources | |
| 9 | Evaluation: 20+ questions, RAG quality, citation accuracy, coverage, latency, cost | |
| 10 | Security: secrets, URL validation, SSRF protection, rate limits, prompt injection | |
| 11 | Packaging: Docker, docker-compose, reproducible demo | |

## Project structure (current)

```text
app/
  config.py          settings and validation
  main.py            command-line configuration check
frontend/
  dashboard.py       Streamlit interface
storage/             local documents, vector index, database, reports (not committed)
tests/               configuration and startup tests
```
