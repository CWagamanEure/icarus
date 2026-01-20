
import json
import urllib.request
import urllib.parse
from decimal import Decimal

def fetch_tick_size(token_id: str, base_url: str = "https://clob.polymarket.com", timeout: int = 10) -> Decimal:
    params = urllib.parse.urlencode({"token_id": str(token_id)})
    url = f"{base_url}/book?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "mm-bot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    ts = data.get("tick_size")
    if ts is None:
        raise ValueError(f"CLOB /book response missing tick_size for token_id={token_id}")
    return Decimal(str(ts))

