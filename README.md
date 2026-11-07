# DeepResearch AI

An autonomous research agent. Given a question, it plans the research, searches the web, collects pages and PDFs,
indexes them for retrieval-augmented generation (RAG), fact-checks the important claims and writes a structured report
with traceable citations.

**Status: phase 7 of 11 (cited reports).** A full research runs from the command line and produces a cited report in Markdown, PDF and JSON. The web interface comes in phase 8; see the [roadmap](#roadmap).

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
| Language model | `LLM_PROVIDER` (`ollama` / `anthropic`; `openai` is reserved but not implemented yet), model names, API keys, timeout, Ollama context size, output length |
| Cost guardrails | `ALLOW_PAID_APIS` (default `false`), `MAX_COST_USD_PER_RESEARCH`, Anthropic prices used for the estimate |
| Web search | `SEARCH_PROVIDER` (`tavily` / `duckduckgo`), `SEARCH_FALLBACK`, `TAVILY_API_KEY`, monthly credit limit, results per query, retries |
| Research limits | up to 3 extra research cycles, sources per research, maximum duration, sub-questions, claims checked per sub-question, passages per analysis, supported-claim ratio |
| Downloads | HTTP timeout, maximum download size, redirects, robots.txt, user agent |
| Documents | maximum PDF pages, minimum and maximum characters per document |
| RAG | chunk size 480 tokens with 72 overlap, multilingual embedding model, reranker, top-k, RRF constant, Qdrant collection |

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

## Retrieval (RAG)

```bash
python -m app.rag.retriever --research-id demo-batteries --query "Why have lithium-ion battery prices fallen?"
```

The first run downloads the embedding model and the reranker (about 1.5 GB in total, with CPU PyTorch); everything then
runs locally on the CPU.

1. **Chunking** ([chunking.py](app/rag/chunking.py)): whole sentences and paragraphs packed into chunks of up to
   `CHUNK_SIZE_TOKENS` (counted with the embedding model's own tokenizer), each repeating the last sentences of the
   previous one up to `CHUNK_OVERLAP_TOKENS`. Every chunk keeps its character offsets in the source text and, for PDFs,
   its pages, so a citation can point to the exact passage.
2. **Embeddings** ([embeddings.py](app/rag/embeddings.py)): `intfloat/multilingual-e5-base` with its `query:` /
   `passage:` prefixes and normalized vectors; a French question finds passages in English documents.
3. **Vector store** ([vector_store.py](app/rag/vector_store.py)): Qdrant in local embedded mode (no server), cosine
   distance, filtered by research. Chunk texts are also kept in SQLite, so BM25 and citations never depend on Qdrant.
4. **Hybrid retrieval** ([retriever.py](app/rag/retriever.py)): dense search and BM25 (accent-insensitive, French and
   English stop words) each return their top `RETRIEVAL_TOP_K`; their rankings are merged with **Reciprocal Rank Fusion**
   (score = sum of 1 / (k + rank), k = 60), which needs no score calibration between the two.
5. **Reranking** ([reranker.py](app/rag/reranker.py)): the multilingual cross-encoder
   `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reads question and passage together and keeps the best `RERANK_TOP_K`.

Each retrieved passage carries its source (title, URL, type, dates), its location (page or character range) and the
ranks and scores from every stage. Indexing is incremental: sources already chunked are skipped.

**Why 480 tokens and not the ~800 of the original brief:** multilingual-e5-base reads at most 512 tokens. Chunks of 800
would be cut silently when embedded, hiding their last third from semantic search. The application refuses any
`CHUNK_SIZE_TOKENS` the model cannot read in full.

Measured on a laptop CPU (Intel i7-1355U): 24 chunks from 3 web pages indexed and queried in under a minute; a 19-page
PDF gives 47 chunks.

## Agents

The language model is reached through [app/llm.py](app/llm.py), which always asks for **structured output**: the
JSON schema of a Pydantic model is sent with the request and the answer is validated against it (one corrective retry).

- **Ollama** (default, free): the schema is passed as Ollama's required output `format`; default model `qwen2.5:3b`
  with an 8,192-token context. Install [Ollama](https://ollama.com/), then `ollama pull qwen2.5:3b`.
- **Anthropic** (paid, opt-in): `messages.parse` with the Pydantic model as `output_format`. A cost tracker estimates the
  spend of every call from the token counts and refuses the next call once `MAX_COST_USD_PER_RESEARCH` is reached.

| Agent | Input | Output | Guardrails applied in code |
| --- | --- | --- | --- |
| [Planner](app/agents/planner.py) | question, language | objective, sub-questions with search queries, quality criteria | count capped, ids renumbered, duplicate/empty queries removed, never empty |
| [Researcher](app/agents/researcher.py) | sub-question | sources to collect, primary-source flags, follow-up queries | results merged and de-duplicated across queries, known URLs skipped, invented labels ignored (falls back to search ranking) |
| [Analyst](app/agents/analyst.py) | sub-question, retrieved passages | summary, claims with cited chunks, disagreements, gaps | cited passage ids checked; invented ids dropped and counted; uncited claims kept for the fact checker; prompt ids ("passage P1") removed from the text |
| [Fact Checker](app/agents/fact_checker.py) | claim, passages retrieved for it | `VERIFIED` / `PARTIALLY_VERIFIED` / `CONFLICTING` / `UNVERIFIED`, evidence, justification | a status is never stronger than its cited evidence (e.g. `VERIFIED` without a valid supporting passage becomes `UNVERIFIED`) |
| [Writer](app/agents/writer.py) | checked claims with ids and statuses | title and summary sentences, each naming the claims it states | sections, citations, limitations and bibliography built by code; summary sentences that name no reportable claim, or add numbers or words not in it, are dropped (see below) |

Web content reaches the models only inside `<document>` tags, with the rule that it is data and never instructions;
attempts to close the tag from inside a page are neutralized. The Writer never sees web text, only checked claims.

Measured with `qwen2.5:3b` on a laptop CPU (Intel i7-1355U, no GPU): about 50 to 95 seconds per agent call; one
plan, one analysis, three fact checks and a short report took 6 calls and 6.5 minutes, at no cost. A small local model
tends to be lenient when judging evidence: the code guardrails stop unsupported verdicts, and phase 9 measures the rest.

## Full research (LangGraph)

```bash
python -m app.graph.workflow "Quels sont les avantages et les limites des pompes à chaleur ?" --language fr
python -m app.graph.workflow --resume <research_id>     # continue an interrupted research
```

```text
plan -> research -> collect -> index -> analyze -> fact_check --(evidence sufficient, time or cycle limit)--> write
                                                       |
                                                       +--(a sub-question lacks support)--> next_cycle -> research ...
```

- **State** ([state.py](app/graph/state.py)): plain JSON only (plan, queries, analyses, checked claims, report and an
  append-only event log), so checkpoints never depend on Python classes.
- **Routing** ([routing.py](app/graph/routing.py)): a sub-question is answered when at least `MIN_SUPPORTED_RATIO` of
  its checked claims are `VERIFIED` or `PARTIALLY_VERIFIED`. Otherwise only the unanswered sub-questions get another
  cycle, using the complementary queries proposed by the Researcher, at most `MAX_EXTRA_RESEARCH_CYCLES` (3) times.
- **Limits**: every node checks `MAX_RESEARCH_MINUTES` before each expensive call and, once it has passed, the run goes
  straight to the report with what it has; `MAX_CLAIMS_PER_SUB_QUESTION` bounds fact checking. Claims skipped because
  of a limit are reported as unverified but do not count as missing evidence.
- **Graceful degradation**: a timed-out or invalid model answer does not stop the research. Source selection falls
  back to the search engine's ranking, a failed analysis becomes a gap, a failed check an unchecked claim. A missing
  model or an exhausted budget does stop it, and the research can be resumed afterwards.
- **Checkpointing** ([workflow.py](app/graph/workflow.py)): LangGraph's SQLite checkpointer saves the state after every
  node (`storage/checkpoints.db`, one thread per research id). `--resume` continues from the last completed step and
  restarts the duration limit for the new session.

**First full run, measured** (local `qwen2.5:3b`, laptop CPU, question on heat pumps in French): interrupted once by a
model timeout, resumed from its checkpoint without redoing the plan; 12 sources collected, 83 chunks, 18 model calls,
one extra research cycle for the sub-question on limitations (two claims found `CONFLICTING`), 34 minutes, $0. The
report was written, but the small model cited no source in its text and stated some unverified claims as facts:
phase 7 builds cited statements in code from the checked claims instead of relying on the model.

## Cited reports

```bash
python -m app.reports.export --research-id <research_id>   # re-render a saved report
```

At the end of a research the report is saved in `storage/reports/<research_id>/` as `report.json` (the full record),
`report.md` and `report.pdf`. Traceability does not depend on the model:

- **Statements are built by code** ([writer.py](app/agents/writer.py)) from the checked claims, one section per
  sub-question. Each carries its citations (`[S1]`, numbered in order of first appearance) and the wording of its
  status: a `PARTIALLY_VERIFIED` claim is introduced as partially supported, a `CONFLICTING` one as disputed with the
  sources that contradict it. `UNVERIFIED` claims never appear in the body: they are listed apart, with the reason
  (not supported, or not examined because of a limit).
- **Every citation points to a passage**: source, chunk, page for PDFs or character offsets otherwise, and a short
  quote (the sentence of the passage closest to the claim, kept on one page so the page number holds).
- **The model writes only the title and the summary.** Each summary sentence must name the claims it states. Code
  drops a sentence that names none of the reportable claims, contains a number that is not in them, or whose words
  mostly come from neither the claims nor the question; it then adds the citations itself. If nothing is left, or the
  call fails, the summary is built from the verified claims.
- **Limitations are stated by code** from what the research knows about itself: why it stopped, how many claims could
  not be verified or were not examined, parts without any verified claim, claims resting on a single source, and the
  gaps found by the Analyst.
- **Export**: Markdown and PDF are rendered from the same JSON ([export.py](app/reports/export.py)). The PDF is laid out
  by PyMuPDF with embedded Unicode fonts, so no extra dependency. Text from the web is escaped in both formats: a page
  cannot inject links, images or HTML into the report. Workflow reasons are shown in the report language.

**Measured on the heat-pump research** (same checked claims as the phase 6 run, only the writing step redone, one
model call of about a minute): 7 sources cited instead of 0, unverified claims kept out of the body, every statement
linked to its passages. The summary check dropped one sentence that attached an unverified figure ("80 to 120 % of
heat losses") to an unrelated verified claim. What the report also makes visible: the fact checker is lenient, citing
loosely related passages and accepting a figure ("75 % renewable energy") that no quoted passage states. Phase 9
measures this.

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 1 | Setup: configuration, dependencies, starter interface, tests | done |
| 2 | Web search: isolated search interface, errors, quotas, timeouts, raw results saved | done |
| 3 | Document collection: web pages and PDFs, cleaning, deduplication, metadata | done |
| 4 | RAG: chunking, embeddings, Qdrant, BM25, hybrid fusion, reranking | done |
| 5 | Agents: Planner, Researcher, Analyst, Fact Checker, Writer (Pydantic inputs/outputs) | done |
| 6 | Orchestration with LangGraph: routing, iteration limits, checkpointing | done |
| 7 | Citations and reports: claim traceability, Markdown and PDF export | done |
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
  rag/
    chunking.py      sentence-aware chunks with offsets and pages
    embeddings.py    multilingual E5 embeddings (local)
    vector_store.py  Qdrant, local embedded mode
    reranker.py      multilingual cross-encoder
    retriever.py     BM25, dense search, Reciprocal Rank Fusion, reranking, incremental indexing
  agents/
    base.py          untrusted-content framing, passage labels, citation checks
    planner.py, researcher.py, analyst.py, fact_checker.py, writer.py
  graph/
    state.py         typed JSON state and event log
    routing.py       evidence sufficiency, cycle and time limits
    nodes.py         one node per step, with graceful degradation
    workflow.py      LangGraph graph, SQLite checkpoints, run / resume / stream
  reports/
    export.py        Markdown, HTML and PDF rendering, escaping of web text
    i18n.py          workflow reasons in the report language
  llm.py             Ollama and Anthropic clients with structured output and cost tracking
frontend/
  dashboard.py       Streamlit interface
storage/             local documents, vector index, database, reports (not committed)
tests/               214 tests, no network, no credits and no model download
```
