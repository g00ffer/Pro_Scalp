"""
Market Data Gateway для Binance USDT-M Futures.

Поддерживаемые потоки:
- <symbol>@bookTicker
- <symbol>@aggTrade
- <symbol>@depth@100ms

Snapshot берётся через REST:
- GET /fapi/v1/depth
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import httpx
import msgspec
import websockets
from websockets.exceptions import ConnectionClosed

from proscalper.core.events import (
    BookDeltaBatchEvent,
    BookLevel,
    BookSnapshotEvent,
    BookTickerEvent,
    TradeEvent,
)


class MarketDataHandler(Protocol):
    """
    Обработчик рыночных событий.
    """

    async def on_trade(self, event: TradeEvent) -> None:
        ...

    async def on_book_ticker(self, event: BookTickerEvent) -> None:
        ...

    async def on_snapshot(self, event: BookSnapshotEvent) -> None:
        ...

    async def on_delta_batch(self, event: BookDeltaBatchEvent) -> None:
        ...

    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class BinanceFuturesWSConfig:
    """
    Конфигурация Binance Futures market data gateway.
    """
    symbols: List[str]
    base_ws_url: str = "wss://fstream.binance.com"
    rest_base_url: str = "https://fapi.binance.com"
    depth_update_speed: str = "100ms"
    snapshot_limit: int = 100
    request_timeout_sec: float = 10.0
    reconnect_base_delay_sec: float = 1.0
    reconnect_max_delay_sec: float = 30.0
    snapshot_request_interval_sec: float = 0.2


class BinanceFuturesMarketDataWS:
    """
    WebSocket-шлюз рыночных данных Binance Futures.
    """

    def __init__(
        self,
        config: BinanceFuturesWSConfig,
        handler: MarketDataHandler,
    ) -> None:
        self._config = config
        self._handler = handler

        self._symbols = [s.upper() for s in config.symbols]
        self._running = False
        self._ws: Optional[Any] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._reconnect_attempts = 0

    async def run(self) -> None:
        """
        Главный цикл работы.
        """
        self._running = True
        self._http = httpx.AsyncClient(timeout=self._config.request_timeout_sec)

        try:
            while self._running:
                try:
                    await self._notify_state("CONNECTING", {
                        "symbols": self._symbols,
                    })

                    await self._connect_and_consume()

                    self._reconnect_attempts = 0

                except asyncio.CancelledError:
                    raise

                except ConnectionClosed as exc:
                    await self._notify_state("WS_CONNECTION_CLOSED", {
                        "error": str(exc),
                    })

                except Exception as exc:
                    await self._notify_state("WS_ERROR", {
                        "error": str(exc),
                        "type": type(exc).__name__,
                    })

                if not self._running:
                    break

                delay = self._next_reconnect_delay()
                await self._notify_state("RECONNECT_SCHEDULED", {
                    "delay_sec": delay,
                    "attempt": self._reconnect_attempts,
                })

                await asyncio.sleep(delay)

        finally:
            if self._http is not None:
                await self._http.aclose()
                self._http = None

    async def stop(self) -> None:
        """
        Остановить шлюз.
        """
        self._running = False

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    # =========================
    # Internal
    # =========================

    async def _connect_and_consume(self) -> None:
        url = self._build_combined_stream_url()

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=2 ** 23,
            compression=None,
            open_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_attempts = 0

            await self._notify_state("CONNECTED", {
                "url": url,
            })

            consumer_task = asyncio.create_task(self._consume_messages(ws))

            try:
                await self._request_snapshots()
                await consumer_task
            finally:
                if not consumer_task.done():
                    consumer_task.cancel()
                    try:
                        await consumer_task
                    except asyncio.CancelledError:
                        pass

            self._ws = None

    async def _consume_messages(self, ws: Any) -> None:
        async for raw_message in ws:
            await self._handle_message(raw_message)

    async def _handle_message(self, raw_message: Any) -> None:
        if isinstance(raw_message, str):
            raw_bytes = raw_message.encode("utf-8")
        else:
            raw_bytes = raw_message

        payload_wrapper = msgspec.json.decode(raw_bytes)

        if not isinstance(payload_wrapper, dict):
            return

        stream_name = payload_wrapper.get("stream", "")
        payload = payload_wrapper.get("data", payload_wrapper)

        event_type = payload.get("e")

        if event_type is None:
            if stream_name.endswith("@bookTicker"):
                event_type = "bookTicker"
            elif stream_name.endswith("@aggTrade"):
                event_type = "aggTrade"
            elif "@depth" in stream_name:
                event_type = "depthUpdate"
            else:
                return

        local_ts_ns = time.time_ns()

        if event_type == "aggTrade":
            event = self._parse_agg_trade(payload, local_ts_ns)
            if event is not None:
                await self._handler.on_trade(event)

        elif event_type == "bookTicker":
            event = self._parse_book_ticker(payload, local_ts_ns)
            if event is not None:
                await self._handler.on_book_ticker(event)

        elif event_type == "depthUpdate":
            event = self._parse_depth_update(payload, local_ts_ns)
            if event is not None:
                await self._handler.on_delta_batch(event)

    def _parse_agg_trade(
        self,
        payload: Dict[str, Any],
        local_ts_ns: int,
    ) -> Optional[TradeEvent]:
        try:
            symbol = payload["s"].upper()

            if symbol not in self._symbols:
                return None

            event_time_ms = int(payload.get("E", 0))
            ts_exchange_ns = event_time_ms * 1_000_000 if event_time_ms else local_ts_ns

            return TradeEvent(
                ts_exchange_ns=ts_exchange_ns,
                ts_local_ns=local_ts_ns,
                symbol=symbol,
                price=float(payload["p"]),
                quantity=float(payload["q"]),
                is_buyer_maker=bool(payload["m"]),
                trade_id=int(payload["a"]) if payload.get("a") is not None else None,
            )

        except Exception:
            return None

    def _parse_book_ticker(
        self,
        payload: Dict[str, Any],
        local_ts_ns: int,
    ) -> Optional[BookTickerEvent]:
        try:
            symbol = payload["s"].upper()

            if symbol not in self._symbols:
                return None

            event_time_ms = int(payload.get("E", 0))
            ts_exchange_ns = event_time_ms * 1_000_000 if event_time_ms else local_ts_ns

            return BookTickerEvent(
                ts_exchange_ns=ts_exchange_ns,
                ts_local_ns=local_ts_ns,
                symbol=symbol,
                bid_price=float(payload["b"]),
                bid_qty=float(payload["B"]),
                ask_price=float(payload["a"]),
                ask_qty=float(payload["A"]),
                update_id=int(payload["u"]) if payload.get("u") is not None else None,
            )

        except Exception:
            return None

    def _parse_depth_update(
        self,
        payload: Dict[str, Any],
        local_ts_ns: int,
    ) -> Optional[BookDeltaBatchEvent]:
        try:
            symbol = payload["s"].upper()

            if symbol not in self._symbols:
                return None

            event_time_ms = int(payload.get("E", 0))
            ts_exchange_ns = event_time_ms * 1_000_000 if event_time_ms else local_ts_ns

            bids = [
                BookLevel(price=float(price), quantity=float(quantity))
                for price, quantity in payload.get("b", [])
            ]

            asks = [
                BookLevel(price=float(price), quantity=float(quantity))
                for price, quantity in payload.get("a", [])
            ]

            return BookDeltaBatchEvent(
                ts_exchange_ns=ts_exchange_ns,
                ts_local_ns=local_ts_ns,
                symbol=symbol,
                first_update_id=int(payload["U"]),
                last_update_id=int(payload["u"]),
                prev_update_id=int(payload.get("pu", 0)),
                bids=bids,
                asks=asks,
            )

        except Exception:
            return None

    async def _request_snapshots(self) -> None:
        for symbol in self._symbols:
            try:
                snapshot = await self._fetch_depth_snapshot(symbol)
                await self._handler.on_snapshot(snapshot)

                await self._notify_state("SNAPSHOT_RECEIVED", {
                    "symbol": symbol,
                    "last_update_id": snapshot.last_update_id,
                })

            except Exception as exc:
                await self._notify_state("SNAPSHOT_ERROR", {
                    "symbol": symbol,
                    "error": str(exc),
                })

            await asyncio.sleep(self._config.snapshot_request_interval_sec)

    async def _fetch_depth_snapshot(self, symbol: str) -> BookSnapshotEvent:
        if self._http is None:
            raise RuntimeError("HTTP client не инициализирован")

        url = f"{self._config.rest_base_url}/fapi/v1/depth"
        params = {
            "symbol": symbol.upper(),
            "limit": self._config.snapshot_limit,
        }

        response = await self._http.get(url, params=params)
        response.raise_for_status()

        data = msgspec.json.decode(response.content)

        local_ts_ns = time.time_ns()
        event_time_ms = int(data.get("E", 0))
        ts_exchange_ns = event_time_ms * 1_000_000 if event_time_ms else local_ts_ns

        bids = [
            BookLevel(price=float(price), quantity=float(quantity))
            for price, quantity in data.get("bids", [])
        ]

        asks = [
            BookLevel(price=float(price), quantity=float(quantity))
            for price, quantity in data.get("asks", [])
        ]

        return BookSnapshotEvent(
            ts_exchange_ns=ts_exchange_ns,
            ts_local_ns=local_ts_ns,
            symbol=symbol.upper(),
            bids=bids,
            asks=asks,
            first_update_id=int(data["lastUpdateId"]),
            last_update_id=int(data["lastUpdateId"]),
        )

    def _build_combined_stream_url(self) -> str:
        streams: List[str] = []

        for symbol in self._symbols:
            symbol_lower = symbol.lower()

            streams.append(f"{symbol_lower}@bookTicker")
            streams.append(f"{symbol_lower}@aggTrade")
            streams.append(
                f"{symbol_lower}@depth@{self._config.depth_update_speed}"
            )

        streams_query = "/".join(streams)
        return f"{self._config.base_ws_url}/stream?streams={streams_query}"

    def _next_reconnect_delay(self) -> float:
        delay = (
            self._config.reconnect_base_delay_sec
            * (2 ** self._reconnect_attempts)
        )

        delay = min(delay, self._config.reconnect_max_delay_sec)
        self._reconnect_attempts += 1

        return delay

    async def _notify_state(self, state: str, details: Dict[str, Any]) -> None:
        try:
            await self._handler.on_state(state, details)
        except Exception:
            # Ошибка в обработчике состояния не должна ронять gateway.
            pass