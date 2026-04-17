# Icarus

Python project scaffold managed with Poetry.

## Tooling

- `poetry` for dependency and environment management
- `ruff` for formatting and linting
- `pytest` for unit tests
- `scripts/pycheck` for the full local quality check
- `scripts/socket_sanity.py` for a live websocket sanity check against supported exchanges

## Getting started

```bash
poetry install
poetry run icarus
./scripts/pycheck
./scripts/socket_sanity.py hyperliquid BTC --limit 5
./scripts/socket_sanity.py coinbase BTC-USD --limit 5
```
