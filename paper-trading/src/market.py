from dataclasses import dataclass
import json
import urllib.request
import urllib.parse
from typing import Dict, Union, Optional

from .outcome import Outcome 
from .helpers import CTHelpers
from .exchange.gamma_client import fetch_gamma_market_by_condition, parse_json_field   

TokenId = str

@dataclass(frozen=True)
class Market:
    condition_id: str
    collateral_address: str
    ct_token_ids: Dict[Outcome, TokenId]
    clob_token_ids: Dict[Outcome, TokenId]

    gamma_market_id: Optional[str] = None
    slug: Optional[str] = None


    @classmethod
    def from_condition(cls, condition_id: str, collateral_address: str) -> "Market":
        """
        Builds market object using polymarket market condition ID and collateral address
        """

        yes = str(CTHelpers.get_token_id(condition_id, collateral_address, 0))
        no = str(CTHelpers.get_token_id(condition_id, collateral_address, 1))

        mkt = fetch_gamma_market_by_condition(condition_id)
        outcomes = parse_json_field(mkt.get("outcomes"), "outcomes")
        clob_ids = parse_json_field(mkt.get("clobTokenIds"), "clobTokenIds")

        if len(outcomes) != len(clob_ids):
            raise ValueError(f"Gamma outcomes/clobTokenIds length mismatch: {len(outcomes)} vs {len(clob_ids)}")

        clob_map: Dict[Outcome, TokenId] = {}
        for label, tid in zip(outcomes, clob_ids):
            lab = str(label).strip().lower()
            if lab == "yes":
                clob_map[Outcome.YES] = str(tid)
            elif lab == "no":
                clob_map[Outcome.NO] = str(tid)

        return cls(
            condition_id=condition_id,
            collateral_address=collateral_address,
            ct_token_ids = {Outcome.YES: yes, Outcome.NO: no},
            clob_token_ids = clob_map,
            gamma_market_id=str(mkt.get("id")) if mkt.get("id") is not None else None,
            slug=mkt.get("slug")
        )


    def get_ct_token_ids(self) -> list[TokenId]:
        """
        returns token_ids
        """
        return [self.ct_token_ids[Outcome.YES], self.ct_token_ids[Outcome.NO]]

    def outcome_from_asset_id(self, asset_id: Union[str, int]) -> Outcome:
        aid = str(asset_id)
        for out, tid in self.ct_token_ids.items():
            if tid == aid:
                return out
        raise ValueError(f"Unknown asset_id: {asset_id}")

    def get_clob_token_ids(self) -> list[TokenId]:
        return [self.clob_token_ids[Outcome.YES], self.clob_token_ids[Outcome.NO]]

    




