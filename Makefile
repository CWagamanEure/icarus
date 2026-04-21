.PHONY: basis basisall basisperp captureeval evalcapture basisanchorplot papertrade

# Override with e.g. `make basis ASSET=ETH` or `make basisall ASSET=SOL`.
ASSET ?= BTC

basis:
	PYTHONPATH=src python scripts/multi_venue_basis_fair_value_plot.py \
		--asset $(ASSET) \
		--kraken-market $(ASSET)/USD \
		--disable-hyperliquid \
		--disable-okx \
		--basis-anchor-exchange coinbase \
		--basis-min-live-spot-venues 1 \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995

basisall:
	PYTHONPATH=src python scripts/multi_venue_basis_fair_value_plot.py \
		--asset $(ASSET) \
		--kraken-market $(ASSET)/USD \
		--basis-anchor-exchange coinbase \
		--basis-min-live-spot-venues 1 \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995

basisperp:
	PYTHONPATH=src python scripts/multi_venue_basis_fair_value_plot.py \
		--asset $(ASSET) \
		--kraken-market $(ASSET)/USD \
		--enable-hyperliquid-perp \
		--basis-anchor-exchange coinbase \
		--basis-min-live-spot-venues 1 \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995 \
		--basis-perp-process-var-per-sec 0.05 \
		--basis-perp-rho-per-second 0.997

captureeval:
	PYTHONPATH=src python scripts/capture_filter_eval.py \
		--asset $(ASSET) \
		--kraken-market $(ASSET)/USD \
		--enable-hyperliquid-perp \
		--basis-anchor-exchange coinbase \
		--basis-min-live-spot-venues 1 \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995 \
		--basis-perp-process-var-per-sec 0.05 \
		--basis-perp-rho-per-second 0.997

evalcapture:
	PYTHONPATH=src python scripts/evaluate_filter_capture.py --db-path data/filter_eval.sqlite3

basisanchorplot:
	PYTHONPATH=src python scripts/basis_vs_anchor_plot.py \
		--db-path data/filter_eval.sqlite3 \
		--anchor-exchange coinbase

papertrade:
	PYTHONPATH=src python scripts/live_paper_trade_plot.py \
		--asset $(ASSET) \
		--kraken-market $(ASSET)/USD \
		--basis-anchor-exchange coinbase \
		--basis-min-live-spot-venues 1 \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995
