.PHONY: basis basisall basisperp captureeval evalcapture

basis:
	PYTHONPATH=src python scripts/multi_venue_basis_fair_value_plot.py \
		--coinbase-market BTC-USD \
		--kraken-market BTC/USD \
		--disable-hyperliquid \
		--disable-okx \
		--basis-anchor-exchange coinbase \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995

basisall:
	PYTHONPATH=src python scripts/multi_venue_basis_fair_value_plot.py \
		--coinbase-market BTC-USD \
		--kraken-market BTC/USD \
		--okx-market BTC-USDT \
		--hyperliquid-market BTC/USDC \
		--basis-anchor-exchange coinbase \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995

basisperp:
	PYTHONPATH=src python scripts/multi_venue_basis_fair_value_plot.py \
		--coinbase-market BTC-USD \
		--kraken-market BTC/USD \
		--okx-market BTC-USDT \
		--hyperliquid-market BTC/USDC \
		--enable-hyperliquid-perp \
		--hyperliquid-perp-market BTC \
		--basis-anchor-exchange coinbase \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995 \
		--basis-perp-process-var-per-sec 0.05 \
		--basis-perp-rho-per-second 0.997

captureeval:
	PYTHONPATH=src python scripts/capture_filter_eval.py \
		--coinbase-market BTC-USD \
		--kraken-market BTC/USD \
		--okx-market BTC-USDT \
		--hyperliquid-market BTC/USDC \
		--enable-hyperliquid-perp \
		--hyperliquid-perp-market BTC \
		--basis-anchor-exchange coinbase \
		--basis-common-price-process-var-per-sec 20 \
		--basis-process-var-per-sec 0.005 \
		--basis-rho-per-second 0.995 \
		--basis-perp-process-var-per-sec 0.05 \
		--basis-perp-rho-per-second 0.997

evalcapture:
	PYTHONPATH=src python scripts/evaluate_filter_capture.py --db-path data/filter_eval.sqlite3
