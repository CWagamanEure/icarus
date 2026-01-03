from enum import Enum


class Outcome(Enum):
    YES = "YES"
    NO = "NO"

    def complement(self) -> "Outcome":
        return Outcome.NO if self is Outcome.YES else Outcome.YES
