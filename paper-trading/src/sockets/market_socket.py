from .base_socket import BaseSocket
from typing import Any, Dict,  Callable, List
from ..models import JSONDict
from ..events.market_events import * 


class MarketSocket(BaseSocket[MarketEvent]):
    """
    Public market channel (L2 price data) subscribed by token IDs
    """

    def __init__(self, *, custom_feature_enabled: bool = True, assets_ids: List[str], on_event: Callable[[MarketEvent], None],
                 url_base: str = "wss://ws-subscriptions-clob.polymarket.com",
                 **kwargs: Any):
        self.assets_ids = assets_ids
        self.custom_feature_enabled = custom_feature_enabled
        url = f"{url_base.rstrip('/')}/ws/market"
        super().__init__(url, on_event=on_event, **kwargs)


    def _initial_subscribe_payload(self):
        if not self.assets_ids:
            raise ValueError("assets_ids is required for MarketSocket")

        return {
            "type": "market",
            "assets_ids": self.assets_ids,
            "custom_feature_enabled": self.custom_feature_enabled,
        }

    def subscribe(self, more_assets_ids: List[str]):
        """
        adding more to socket
        """
        self.send_json({"operation": "subscribe", "assets_ids":  more_assets_ids})

    def unsubscribe(self, assets_ids: List[str]):
        self.send_json({"operation": "unsubscribe", "assets_ids": assets_ids})



    def parse_event(self, raw: Dict[str, Any]):
        def _ms( x: Any):
            return int(x)


        et = raw.get("event_type")

        market = str(raw.get("market"))


        if et == "book":
            bids = [(lvl["price"], lvl["size"]) for lvl in raw.get("bids", [])]
            asks = [(lvl["price"], lvl["size"]) for lvl in raw.get("asks", [])]
            return BookSnapshot(
                market=market,
                asset_id=raw["asset_id"],
                ts_ms=_ms(raw["timestamp"]),
                book_hash=raw.get("hash", ""),
                bids=bids,
                asks=asks,
                last_trade_price=raw.get("last_trade_price"),
            )

        if et == "price_change":
            changes: List[PriceLevelUpdate] = []
            for pc in raw.get("price_changes", []):
                changes.append(
                    PriceLevelUpdate(
                        asset_id=pc["asset_id"],
                        price=pc["price"],
                        size=pc["size"],
                        side=pc["side"],
                        book_hash=pc["hash"],
                        best_bid=pc.get("best_bid"),
                        best_ask=pc.get("best_ask"),
                    )
                )
            return PriceChangeBatch(
                market=market,
                ts_ms=_ms(raw["timestamp"]),
                changes=changes,
            )

        if et == "last_trade_price":
            return LastTrade(
                market=market,
                asset_id=raw["asset_id"],
                ts_ms=_ms(raw["timestamp"]),
                price=raw["price"],
                size=raw["size"],
                side=raw["side"],
            )

        if et == "best_bid_ask":
            return BestBidAsk(
                market=market,
                asset_id=raw["asset_id"],
                ts_ms=_ms(raw["timestamp"]),
                best_bid=raw["best_bid"],
                best_ask=raw["best_ask"],
                spread=raw.get("spread"),
            )

        if et == "tick_size_change":
            return TickSizeChange(
                market=market,
                asset_id=raw["asset_id"],
                ts_ms=_ms(raw["timestamp"]),
                old_tick_size=raw["old_tick_size"],
                new_tick_size=raw["new_tick_size"],
            )

        return UnknownMarketEvent(market=market, event_type=et, raw=raw)
