from typing import List, Callable, Any

from base_socket import BaseSocket
from ..models import UserAuth, JSONDict 


class UserSocket(BaseSocket):

    def __init__(self, *, markets: List[str], auth: UserAuth, custom_feature_enabled: bool = False, on_event: Callable[[JSONDict], None], url_base: str = "wss://ws-subscriptions-clob.polymarket.com", **kwargs: Any,):

        self.markets = markets
        self.auth = auth 
        self.custom_feature_enabled = custom_feature_enabled
        url = f"{url_base.rstrip('/')}/ws/user"
        super().__init__(url, on_event=on_event, **kwargs)


    def _initial_subscribe_payload(self):
        if not self.markets:
            raise ValueError("markets is required for UserSocket")

        return {
            "type": "user",
            "markets": self.markets,
            "auth": self.auth,
            "custom_feature_enabled": self.custom_feature_enabled
        }


    def subscribe(self, more_markets):
        self.send_json({"operation": "subscribe", "markets": more_markets})

    def unsubscribe(self, markets):
        self.send_json({"operation": "unsubscribe", "markets": markets})




