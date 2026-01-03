import queue
import time
from .market import Market
from .sockets.market_socket import MarketSocket
from .constants import COLLATERAL_ADDRESS
from .market_state import MarketState


if __name__ == "__main__":
    CONDITION_ID = "0xe601d5047ffb85c43758f1cbd5fe5faa5e646f0020f56fcc2534b17741ed9521"

    q =  queue.SimpleQueue()

    def on_market(evt):
        q.put(evt)



    market = Market.from_condition(condition_id=CONDITION_ID, collateral_address=COLLATERAL_ADDRESS)
    state = MarketState(market=market)
    socket = MarketSocket(assets_ids=market.get_token_ids(), on_event=on_market)
    socket.start()

    asset_id = market.get_token_ids()[0]

    try:
        while True:
            evt = q.get()
            state.apply(evt)
            book = state.book(asset_id)
            print(book.summary())
            print(book.pretty())

            
    except KeyboardInterrupt:
        socket.stop()
