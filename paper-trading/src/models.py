from typing import Any, Dict
from dataclasses import dataclass
import json

JSONDict = Dict[str, Any]


@dataclass
class UserAuth:
    apikey: str
    secret: str
    passphrase: str

    def as_dict(self):
        return {"apikey": self.apikey, "secret": self.secret, "passphrase": self.passphrase}




