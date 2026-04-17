from __future__ import annotations

import pytest

from icarus.sockets.base import BaseSocket


class DummyWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.sent_messages: list[str] = []
        self.recv_message = '{"status":"ok"}'

    async def close(self) -> None:
        self.closed = True

    async def send(self, payload: str) -> None:
        self.sent_messages.append(payload)

    async def recv(self) -> str:
        return self.recv_message


class DummySocket(BaseSocket):
    def __init__(self) -> None:
        super().__init__("wss://example.test/socket")
        self.after_connect_calls = 0

    async def after_connect(self) -> None:
        self.after_connect_calls += 1


@pytest.mark.asyncio
async def test_close_closes_active_socket() -> None:
    dummy_ws = DummyWebSocket()
    socket = DummySocket()
    socket._ws = dummy_ws

    await socket.close()

    assert socket._closed is True
    assert dummy_ws.closed is True
    assert socket._ws is None


@pytest.mark.asyncio
async def test_connect_resets_closed_state(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_ws = DummyWebSocket()
    socket = DummySocket()
    socket._closed = True

    async def fake_connect(
        url: str, *, ping_interval: float, ping_timeout: float, max_size: int | None
    ) -> DummyWebSocket:
        assert url == "wss://example.test/socket"
        assert ping_interval == socket.ping_interval
        assert ping_timeout == socket.ping_timeout
        assert max_size is None
        return dummy_ws

    monkeypatch.setattr("icarus.sockets.base.websockets.connect", fake_connect)

    await socket.connect()

    assert socket._closed is False
    assert socket._ws is dummy_ws
    assert socket.after_connect_calls == 1


@pytest.mark.asyncio
async def test_send_and_recv_json_use_active_socket() -> None:
    dummy_ws = DummyWebSocket()
    socket = DummySocket()
    socket._ws = dummy_ws

    await socket.send_json({"hello": "world"})
    response = await socket.recv_json()

    assert dummy_ws.sent_messages == ['{"hello": "world"}']
    assert response == {"status": "ok"}


def test_parse_message_returns_json_object() -> None:
    socket = DummySocket()

    parsed = socket.parse_message('{"count": 3, "ok": true}')

    assert parsed == {"count": 3, "ok": True}
