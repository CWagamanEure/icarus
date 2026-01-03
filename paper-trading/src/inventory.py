from typing import List

from order import Order

class Inventory:

    def __init__(self, orders: List[Order], balances: dict, orders_being_place: bool, orders_being_cancelled: bool):

        self.orders = orders
        self.balances = balances
        self.orders_being_place = orders_being_place
        self.orders_being_cancelled = orders_being_cancelled
