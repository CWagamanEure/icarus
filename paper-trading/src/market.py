from dataclasses import dataclass
from typing import Dict, Union

from .outcome import Outcome 
from .helpers import CTHelpers

TokenId = str

@dataclass(frozen=True)
class Market:
    condition_id: str
    collateral_address: str
    token_ids: Dict[Outcome, TokenId]


    @classmethod
    def from_condition(cls, condition_id: str, collateral_address: str) -> "Market":

        yes = str(CTHelpers.get_token_id(condition_id, collateral_address, 0))
        no = str(CTHelpers.get_token_id(condition_id, collateral_address, 1))
        return cls(
            condition_id=condition_id,
            collateral_address=collateral_address,
            token_ids = {Outcome.YES: yes, Outcome.NO: no}
        )


    def get_token_ids(self) -> list[TokenId]:
        return [self.token_ids[Outcome.YES], self.token_ids[Outcome.NO]]

    def outcome_from_asset_id(self, asset_id: Union[str, int]) -> Outcome:
        aid = str(asset_id)
        for out, tid in self.token_ids.items():
            if tid == aid:
                return out
        raise ValueError(f"Unknown asset_id: {asset_id}")
