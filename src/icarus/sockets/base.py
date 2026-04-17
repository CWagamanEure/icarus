from __future__ import annotations

import abc
import asyncio
import importlib
import importlib.util
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

websockets = (
    importlib.import_module("websockets") if importlib.util.find_spec("websockets") else None
)


class BaseSocket(abc.ABC):
    """Reusable async websocket client with reconnect/backoff support."""

    def __init__(
        self,
        url: str,
        *,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        max_message_size: int | None = None,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.url = url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_message_size = max_message_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._ws: Any | None = None
        self._closed = False

    @abc.abstractmethod
    async def after_connect(self) -> None:
        """Send any auth or subscription messages after the connection opens."""

    async def connect(self) -> None:
        if websockets is None:
            raise RuntimeError(
                "The 'websockets' package is required to use BaseSocket. "
                "Install dependencies before opening a socket."
            )

        self._closed = False
        self._ws = await websockets.connect(
            self.url,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_size=self.max_message_size,
        )
        await self.after_connect()

    async def close(self) -> None:
        self._closed = True

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Socket is not connected.")

        await self._ws.send(json.dumps(payload))

    def parse_message(self, raw_message: str) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(raw_message))

    async def recv_json(self) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Socket is not connected.")

        raw_message = await self._ws.recv()
        return self.parse_message(raw_message)

    async def stream_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed messages indefinitely, reconnecting on disconnects."""
        backoff = self.reconnect_delay

        while not self._closed:
            try:
                await self.connect()
                backoff = self.reconnect_delay

                assert self._ws is not None
                async for raw_message in self._ws:
                    yield self.parse_message(raw_message)
            except asyncio.CancelledError:
                raise
            except RuntimeError:
                raise
            except Exception as exc:  # pragma: no cover - exercised in integration runtime.
                self.logger.warning(
                    "Socket error for %s: %s. Reconnecting in %.1fs.",
                    self.url,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_reconnect_delay)
            finally:
                if self._ws is not None:
                    await self._ws.close()
                    self._ws = None
