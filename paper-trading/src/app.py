import queue
import time
from .market import Market
from .sockets.market_socket import MarketSocket
from .constants import COLLATERAL_ADDRESS
from .market_state import MarketState
from .quoting.fair_price_engine import FairPriceEngine
from .mm_controller import MMController, ControllerParams
from .quoting.models import NoiseModelParams
from .exchange.clob_http import fetch_tick_size

def main() -> None:
    CONDITION_ID = "0x0f8ef3cc906ba7ba94a44724738df44bdd5f73e59e40c9c8b4ff8569e349643c"

    #-----------------
    # Fair Engine Params
    #-------------------

    Q = 1e-4
    x0 = 0.0
    P0 = 0.5

    noise_params = NoiseModelParams(
        a0=1e-4,
        a_spread=5e-2,
        a_depth=1e-2,
        a_imb=1e-2,
    )

    q =  queue.SimpleQueue()

    def on_market(evt):
        q.put(evt)



    print("Building market")
    market = Market.from_condition(condition_id=CONDITION_ID, collateral_address=COLLATERAL_ADDRESS)

    asset_id = market.get_clob_token_ids()[0]

    print("Building state")
    state = MarketState(market=market)
    tick = fetch_tick_size(asset_id) 

    print("Building socket")
    socket = MarketSocket(assets_ids=market.get_clob_token_ids(), on_event=on_market)

    print("Building fair engine and controller")
    fair_engine = FairPriceEngine(Q=Q, x0=x0, P0=P0, noise_params=noise_params) 

    controller = MMController(
        fair_engine=fair_engine,
        params=ControllerParams(
            size=ControllerParams.size,
            tick=tick
        )
    )
    print("Connecting socket")
    socket.start()

    asset_id = market.get_clob_token_ids()[0]
    print("CT:", market.get_ct_token_ids())
    print("CLOB:", market.get_clob_token_ids())
    print("slug:", market.slug)


    last_seq = -1

    try:
        while True:
            evt = q.get()
            state.apply(evt)

            book = state.book(asset_id)

            print(book.summary())

            if book.update_seq == last_seq:
                continue
            last_seq = book.update_seq

            dq = controller.on_book(book)
            if dq:
                print(f"QUOTE bid={dq.bid} ask={dq.ask} size={dq.size} fair={dq.p_fair} ts={dq.ts_ms}")
            
    except KeyboardInterrupt:
        socket.stop()

if __name__ == "__main__":
    main()
