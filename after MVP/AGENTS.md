# Repository Guidelines

## Project Structure & Module Organization
- `main_pipeline.py`: Primary entrypoint. Orchestrates token search, on-chain queries, and scoring.
- `config.py`: Runtime configuration (API keys, search keyword). Prefer environment variables in development.
- `find_tokens.py`: Dune Client helper for keyword-based token discovery.
- `raw_data_collector.py`: Standalone Alchemy data collection example.
- `gemini_api_client.py`: Rate-limited API client with background worker and logging.
- `*.sql`, `*.docx`, `*.png`: Supporting assets and queries. No `tests/` directory yet.

## Build, Test, and Development Commands
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install pandas requests dune-spice dune-client alchemy-sdk pytest`
- Run pipeline: `export DUNE_API_KEY=... && python main_pipeline.py`
- Run Dune search: `export DUNE_API_KEY=... && python find_tokens.py`
- Run Alchemy collector: `python raw_data_collector.py` (set API key first)
- Lint/format (recommended): `pip install black ruff && black . && ruff check .`

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation.
- Use Black defaults (line length 88) and Ruff for linting. Prefer type hints and `snake_case` for functions/variables.
- Modules: group network helpers, parsing, and scoring utilities into cohesive functions; avoid large monoliths.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` with `test_*.py` names.
- Mock network I/O (`requests`, `alchemy`, `spice`, `DuneClient`) via `pytest-mock` or `responses`.
- Minimum: unit tests for helpers (e.g., `http_post/http_get` error paths, data shaping) and scoring logic. Add smoke tests that run without real keys by mocking.

## Commit & Pull Request Guidelines
- Commits: imperative mood, scoped changes. Example: `feat(pipeline): add totalSupply fetch`
- PRs: include purpose, summary of changes, screenshots/log snippets when relevant, and linked issues. Note required env vars and how you validated locally.
- Keep PRs small and focused; add checklists for data/backwards compatibility.

## Security & Configuration Tips
- Do not commit real API keys. Prefer environment variables: `DUNE_API_KEY`, `GEMINI_API_KEY`, `ALCHEMY_API_KEYS`, `ETHERSCAN_MULTICHAIN_API_KEY`.
- Consider `.env` with `python-dotenv` locally; add `.env` to `.gitignore`.
- Refactor `config.py` to read from env with safe defaults. Example: `os.getenv("DUNE_API_KEY")` and fail fast with clear errors.
- Log responsibly: avoid printing secrets; redact addresses only when needed.
