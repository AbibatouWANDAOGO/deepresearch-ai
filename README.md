# DeepResearch AI

An autonomous research agent. Given a question, it plans the research, searches the web, collects pages and PDFs,
indexes them for retrieval-augmented generation (RAG), fact-checks the important claims and writes a structured report
with traceable citations.

**Status: phase 3 of 11 (document collection).** Full research is not available yet; see the [roadmap](#roadmap).

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
| Web search | `SEARCH_PROVIDER` (`tavily` / `duckduckgo`), `SEARCH_FALLBACK`, `TAVILY_API_KEY`, monthly credit limit, results per query, retries |
| Research limits | up to 3 extra research cycles, sources per research, maximum duration |
| Downloads | HTTP timeout, maximum download size, redirects, robots.txt, user agent |
| Documents | maximum PDF pages, minimum and maximum characters per document |
| RAG | chunk size 800 tokens with 120 overlap, multilingual embedding model, reranker, top-k |

## Web search

```bash
python -m app.tools.web_search "What is retrieval-augmented generation?" --max-results 3
```

[app/tools/web_search.py](app/tools/web_search.py) puts every search engine behind one interface returning normalized
results (rank, title, URL, snippet, score, publication date when known):

- **Providers**: [Tavily](https://www.tavily.com/) (basic depth, 1 credit per query; free tier of 1,000 credits a
  month) and DuckDuckGo through `ddgs` (no key, unofficial) as the fallback.
- **Typed errors**: invalid key (401) and exhausted quota (432/433) are never retried; rate limits (429), capacity
  errors (434), server errors, timeouts and network failures are retried with exponential backoff, honoring
  `Retry-After` (capped at 30 s). An invalid request (400/422) is not sent to the fallback, since it would fail there too.
- **Fallback**: if Tavily is unconfigured, rejected or unavailable, the same query goes to DuckDuckGo and the response
  records why (`fallback_used`, `errors`).
- **Local quota guard**: a monthly credit ledger (`storage/search/credit_usage.json`) refuses a Tavily search *before*
  it is sent once `TAVILY_MONTHLY_CREDIT_LIMIT` would be exceeded.
- **Traceability**: every raw provider response is saved as JSON under `storage/search/<research id>/`, next to the
  normalized results; duplicate URLs (case, trailing slash, fragment) and non-web URLs are removed.

DuckDuckGo sometimes answers "no results" for ordinary queries when its backends are blocked (observed intermittently
during development), so an empty DuckDuckGo answer is treated as a temporary failure and retried.

## Document collection

```bash
python -m app.tools.document_parser --query "How do lithium-ion battery prices evolve?" --max-results 5
python -m app.tools.document_parser --url https://arxiv.org/pdf/2005.11401
```

For each URL, [document_parser.py](app/tools/document_parser.py) downloads, extracts, cleans, de-duplicates and stores
the source:

- **Safe download** ([web_scraper.py](app/tools/web_scraper.py), [url_safety.py](app/tools/url_safety.py)): only
  http(s) URLs whose host resolves exclusively to public addresses (no localhost, private ranges, link-local or cloud
  metadata addresses, no credentials in the URL). Redirects are followed one hop at a time and each target is checked
  again, so a public page cannot redirect the downloader to a local service. robots.txt is respected (cached per
  site), the body is streamed and cut at `MAX_DOWNLOAD_MB`, and every failure becomes a status instead of an exception.
- **Extraction**: [Trafilatura](https://trafilatura.readthedocs.io/) keeps the main content of web pages (no menus,
  footers or scripts) with title, author, date and site name; [PyMuPDF](https://pymupdf.readthedocs.io/) reads PDFs
  page by page ([pdf_reader.py](app/tools/pdf_reader.py)). When a PDF has no metadata title, the title from the search
  result or the first lines of page 1 are used.
- **Cleaning**: Unicode normalization, control and zero-width characters removed, whitespace collapsed; for PDFs,
  page numbers and running headers/footers repeated on most pages are removed.
- **De-duplication**: the same URL (case, trailing slash, fragment) is fetched once per research; the same text at
  another address (hash insensitive to case, spacing and punctuation) is recorded as a `duplicate` of the first source.
- **Storage**: metadata in SQLite (`storage/documents.db`, [app/database/](app/database/)): `source_id`, title, URL,
  final URL, excerpt, publication and collection dates, type, access status, error, content hash, size, pages.
  Text in `storage/documents/<research_id>/<source_id>.txt`, PDF pages separated by form feeds so citations can
  point to a page.

Access statuses: `ok`, `duplicate`, `too_short`, `extraction_failed`, `unsafe_url`, `robots_disallowed`, `http_error`,
`timeout`, `network_error`, `too_large`, `unsupported_type`, `too_many_redirects`. Extracted text is stored and later
quoted as data: instructions found in a page are never executed.

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 1 | Setup: configuration, dependencies, starter interface, tests | done |
| 2 | Web search: isolated search interface, errors, quotas, timeouts, raw results saved | done |
| 3 | Document collection: web pages and PDFs, cleaning, deduplication, metadata | done |
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
  database/          SQLite connection and source records
  tools/
    web_search.py    search providers, retries, fallback, credit ledger, raw-result storage
    url_safety.py    URL checks before any download (SSRF protection)
    web_scraper.py   streamed downloads, redirects, robots.txt, size limits
    pdf_reader.py    PDF text and metadata, page by page
    document_parser.py  extraction, cleaning, de-duplication, storage
frontend/
  dashboard.py       Streamlit interface
storage/             local documents, vector index, database, reports (not committed)
tests/               122 tests, no network and no credits
```
