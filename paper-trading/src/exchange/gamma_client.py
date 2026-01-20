import json
import urllib.request
import urllib.parse
from typing import Optional

def fetch_gamma_market_by_condition(condition_id: str) -> dict:
    """
    Returns the first Gamma market object for a condition_id.
    """
    params = urllib.parse.urlencode({"condition_ids": condition_id, "limit": 1})
    url = f"https://gamma-api.polymarket.com/markets?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "paper-trading/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"No Gamma market found for condition_id={condition_id}")

    return data[0]


def parse_json_field(field: Optional[str], field_name: str):
    """
    Gamma returns some fields as JSON-encoded strings (e.g. "clobTokenIds": "[\"...\",\"...\"]").
    """
    if field is None:
        raise ValueError(f"Gamma response missing field '{field_name}'")
    try:
        return json.loads(field)
    except Exception as e:
        raise ValueError(f"Failed to json-decode Gamma field '{field_name}': {field!r}") from e


