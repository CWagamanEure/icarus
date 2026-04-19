.PHONY: basis basisall

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
