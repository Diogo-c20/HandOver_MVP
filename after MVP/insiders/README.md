# Insiders Risk Scoring Engine

Config-driven, modular, and testable Python implementation of an insiders risk
scoring engine focused on market impact from a general investor perspective.

Highlights:
- Denominator fixed at t0 for all ratio metrics
- Cohort normalization using percentile + robust-z mixture with hierarchical backoff and time decay
- Event penalties and reliefs with synergy damping
- Label trust weighting to conservatively adjust scores
- Mock-friendly I/O connectors; FastAPI endpoint for scoring

Run tests:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -U pytest numpy pydantic fastapi uvicorn
pytest -q
```

