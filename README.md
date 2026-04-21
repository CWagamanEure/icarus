# Icarus

Cryptocurrency market-making research platform. Ingests L2 order book and trade
data from four venues over websockets, estimates a cross-venue fair value with
a state-space filter, and uses the output for alpha research and strategy
backtesting.

## System

```
websockets ──▶ capture ──▶ SQLite ──▶ Kalman filter ──▶ drift predictor ──▶ maker simulator
  (4 venues)                          (fair value +                          (backtest + paper trade)
                                       per-venue basis)
```

Venues: Coinbase, Kraken, OKX, Hyperliquid (spot; Hyperliquid perp optional).

## Key components

| Component | Location |
|---|---|
| Fair-value filter (multi-venue Kalman) | `src/icarus/strategy/fair_value/filters/venue_basis_kalman_filter.py` |
| Drift predictor (Ridge, walk-forward) | `scripts/walk_forward_lagged.py` |
| Maker simulator (queue + latency) | `scripts/simulate_basis_maker.py` |
| Live capture pipeline | `scripts/capture_filter_eval.py` |
| Filter accuracy eval | `scripts/filter_innovation_mae.py` |
| Stat-arb backtest | `scripts/basis_mean_reversion_backtest.py` |

## Findings

**1. Filter tracks each venue at sub-spread accuracy**
One-step-ahead MAE on fair-value prediction across 1M+ observations (12h BTC
capture, 4 venues):

| venue | MAE | bps |
|---|---|---|
| kraken | $4.04 | 0.53 |
| coinbase | $10.40 | 1.37 |
| okx | $10.69 | 1.41 |
| hyperliquid | $11.46 | 1.51 |

Typical BTC bid-ask spread on these venues is 1–2 bps, so the filter's
prediction error is at or below the spread.

**2. Identified alpha, fee-gated at retail**
Walk-forward Ridge regression on 40 cross-venue and lagged features identifies
Coinbase as the price leader (1–2s ahead of Kraken/OKX/Hyperliquid). Maker
simulation produces ~$7/fill of simulated edge on Coinbase. At 40 bps retail
maker fee this is deeply unprofitable; the strategy breaks even only under a
negotiated market-maker rebate (≤1 bps).

**3. Cross-venue basis mean-reversion has no edge**
Tested entry on z-score deviations of per-venue basis vs cross-venue common
price; gross P&L (pre-fee) was <0.5 bps per trade on $10K notional across 2000+
trades, indistinguishable from noise. Ruled out quantitatively.

## Getting started

```bash
poetry install
./scripts/pycheck

# Live fair-value plot (coinbase anchor, spot venues)
make basis ASSET=BTC

# Replay a capture to measure filter accuracy
PYTHONPATH=src python scripts/filter_innovation_mae.py \
    --db-path data/capture/YYYY-MM-DD.sqlite3

# Backtests
PYTHONPATH=src python scripts/simulate_basis_maker.py
PYTHONPATH=src python scripts/basis_mean_reversion_backtest.py
```

## Tooling

- `poetry` for dependency and environment management
- `ruff` for formatting and linting
- `pytest` for unit tests
- `scripts/pycheck` for the full local quality check
- `scripts/socket_sanity.py` for live websocket health checks
