from enum import Enum
from .outcome import Outcome 


class Side(Enum):
    BUY = "BUY" 
    SELL = "SELL" 



class Order:
    def __init__(self, size: float, price: float, side: Side, outcome: Outcome, id=None):

        self.size = size
        self.price = price
        self.side = side
        self.token = Outcome 
        self.id = id


    def __repr__(self):
        return f"Order[id={self.id}, price={self.price}, size={self.size}, side={self.side.value}, token={self.token.value}]"
