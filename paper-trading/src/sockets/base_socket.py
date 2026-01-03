from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Optional, Dict, TypeVar, Generic

import websocket 
from websocket import WebSocketApp

JSONDict = dict[str, Any]

TEvent = TypeVar("TEvent") 


class BaseSocket(Generic[TEvent]):
    def __init__(
        self,
        url: str,
        *,
        on_event: Callable[[TEvent], None],
        ping_interval_s: int = 5,    
        verbose: bool = False,
    ):
        self.url = url
        self.on_event = on_event
        self.ping_interval_s = ping_interval_s
        self.verbose = verbose

        self._stop = threading.Event()
        self._ws: Optional[WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None

        self._send_lock = threading.Lock()


    # -------------- Public API --------------

    def start(self, daemon: bool = True) -> threading.Thread:
        self._ws_thread = threading.Thread(target=self.run_forever, daemon=daemon)
        self._ws_thread.start()
        return self._ws_thread

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._ping_thread and self._ping_thread.is_alive():
            self._ping_thread.join(timeout=1)

    def send_json(self, payload: JSONDict) -> None:
        self._send_text(json.dumps(payload))

    # -------------- Hooks for subclasses --------------

    def _initial_subscribe_payload(self) -> JSONDict:
        raise NotImplementedError

    def parse_event(self, raw: Dict[str, Any]) -> Optional[TEvent]:
        """Subclasses: raw dict -> typed event (or None to ignore)."""
        raise NotImplementedError

    # -------------- websocket-client callbacks --------------

    def run_forever(self): 
        self._ws = WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        if self.verbose:
            print(f"[socket] connecting -> {self.url}")

        self._ws.run_forever()

    def _on_open(self, ws ) -> None:
        payload = self._initial_subscribe_payload()
        if self.verbose:
            print(f"[socket] on_open subscribe -> {payload}")
        self._send_text(json.dumps(payload))

        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

    def _on_message(self, ws , message: str) -> None:
        if message.strip() == "PONG":
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            if self.verbose:
                print(f"[socket] non-json message: {message!r}")
            return

        if isinstance(data, list):
            for d in data:
                self._handle_one(d)
        elif isinstance(data, dict):
            self._handle_one(data)
        else:
            return

    def _handle_one(self, raw: JSONDict) -> None:
        evt = self.parse_event(raw)
        if evt is not None:
            self.on_event(evt)

    def _on_error(self, ws , error: Any) -> None:
        print(f"[socket] error: {error!r}")

    def _on_close(self, ws , code: int, reason: str) -> None:
        if self.verbose:
            print(f"[socket] closed: code={code} reason={reason!r}")

    # -------------- internals --------------

    def _send_text(self, text: str) -> None:
        if not self._ws:
            raise RuntimeError("Socket not started yet")
        with self._send_lock:
            self._ws.send(text)

    def _ping_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._send_text("PING")
            except Exception:
                if self.verbose:
                    print("[socket] ping failed; stopping ping loop")
                return
            time.sleep(self.ping_interval_s)

