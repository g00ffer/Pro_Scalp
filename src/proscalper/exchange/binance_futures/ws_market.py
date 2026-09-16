"""
Market Data Gateway для Binance USDT-M Futures.

Используются ДВА отдельных WebSocket-соединения:
1. bookTicker + depth (для стакана)
2. aggTrade (для ленты сделок)

Snapshot берётся через REST ПОСЛЕ открытия WS, чтобы избежать
гонки между snapshot и первыми depth events.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import httpx
import msgspec
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from proscalper.core.events import (
    BookDeltaBatchEvent,
    BookLevel,
    BookSnapshotEvent,
    BookTickerEvent,
    TradeEvent,
)


class MarketDataHandler(Protocol):
    """Обработчик рыночных событий."""

    async def on_trade(self, event: TradeEvent) -> None: ...
    async def on_book_ticker(self, event: BookTickerEvent) -> None: ...
    async def on_snapshot(self, event: BookSnapshotEvent) -> None: ...
    async def on_delta_batch(self, event: BookDeltaBatchEvent) -> None: ...
    async def on_state(self, state: str, details: Dict[str, Any]) -> None: ...
    
    # НОВОЕ: сброс локальных стаканов перед reconnect
    async def reset_books(self) -> None: ...


@dataclass(frozen=True)
class BinanceFuturesWSConfig:
    """Конфигурация Binance Futures market data gateway."""
    symbols: List[str]
    base_ws_url: str = "wss://fstream.binance.com"
    rest_base_url: str = "https://fapi.binance.com"
    depth_update_speed: str = "100ms"
    snapshot_limit: int = 1000  # Увеличили с 100 до 1000 для надёжности
    request_timeout_sec: float = 10.0
    reconnect_base_delay_sec: float = 1.0
    reconnect_max_delay_sec: float = 30.0
    snapshot_request_interval_sec: float = 0.3


class BinanceFuturesMarketDataWS:
    """
    WebSocket-шлюз рыночных данных Binance Futures.
    
    Использует ДВА отдельных соединения для надёжности:
    - Соединение 1: bookTicker + depth
    - Соединение 2: aggTrade
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
        self._http: Optional[httpx.AsyncClient] = None
        self._reconnect_attempts = 0

    async def run(self) -> None:
        """Главный цикл работы."""
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

                except (ConnectionClosed, ConnectionClosedError) as exc:
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
        """Остановить шлюз."""
        self._running = False

    async def _connect_and_consume(self) -> None:
        """Подключение и запуск двух параллельных соединений."""
        
        # ВАЖНО: сбрасываем все локальные стаканы перед новым подключением.
        # Это необходимо, чтобы старые last_update_id не мешали новым events.
        await self._handler.reset_books()
        
        book_depth_task = asyncio.create_task(
            self._run_book_depth_stream()
        )
        trade_task = asyncio.create_task(
            self._run_trade_stream()
        )

        try:
            # Ждём, пока хотя бы одно соединение не упадёт
            done, pending = await asyncio.wait(
                [book_depth_task, trade_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Отменяем оставшиеся задачи, подавляя ошибки отмены
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, ConnectionClosed, ConnectionClosedError, Exception):
                    pass

        except asyncio.CancelledError:
            book_depth_task.cancel()
            trade_task.cancel()
            # Подавляем ошибки отмены
            for task in (book_depth_task, trade_task):
                try:
                    await task
                except Exception:
                    pass
            raise

    async def _run_book_depth_stream(self) -> None:
        """Отдельное соединение для bookTicker и depth."""
        streams = []
        for symbol in self._symbols:
            symbol_lower = symbol.lower()
            streams.append(f"{symbol_lower}@bookTicker")
            streams.append(f"{symbol_lower}@depth@{self._config.depth_update_speed}")

        url = f"{self._config.base_ws_url}/stream?streams={'/'.join(streams)}"

        async with websockets.connect(
            url,
            ping_interval=30,
            ping_timeout=30,
            close_timeout=5,
            max_size=2 ** 23,
            compression=None,
            open_timeout=10,
        ) as ws:
            await self._notify_state("BOOK_DEPTH_CONNECTED", {"url": url})

            snapshot_task = asyncio.create_task(self._request_snapshots())

            try:
                async for raw_message in ws:
                    await self._handle_message(raw_message)
            except (ConnectionClosed, ConnectionClosedError) as exc:
                # Нормальное закрытие соединения — не пробрасываем дальше
                await self._notify_state("BOOK_DEPTH_CLOSED", {"error": str(exc)})
            finally:
                if not snapshot_task.done():
                    snapshot_task.cancel()
                    try:
                        await snapshot_task
                    except (asyncio.CancelledError, Exception):
                        pass

    async def _run_trade_stream(self) -> None:
        """Отдельное соединение для сырых сделок (@trade)."""
        tasks = []
        
        for symbol in self._symbols:
            symbol_lower = symbol.lower()
            # ВАЖНО: используем @trade вместо @aggTrade
            url = f"{self._config.base_ws_url}/ws/{symbol_lower}@trade"
            
            task = asyncio.create_task(
                self._run_single_trade_stream(symbol, url)
            )
            tasks.append(task)

        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_single_trade_stream(self, symbol: str, url: str) -> None:
        """Отдельное соединение для одного символа trade."""
        await self._notify_state("TRADE_CONNECTED", {"url": url, "symbol": symbol})

        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_size=2 ** 23,
                compression=None,
                open_timeout=10,
            ) as ws:
                async for raw_message in ws:
                    await self._handle_message(raw_message)
        except (ConnectionClosed, ConnectionClosedError) as exc:
            await self._notify_state("TRADE_CLOSED", {
                "symbol": symbol,
                "error": str(exc),
            })

    async def _handle_message(self, raw_message: Any) -> None:
        if isinstance(raw_message, str):
            raw_bytes = raw_message.encode("utf-8")
        else:
            raw_bytes = raw_message

        try:
            payload_wrapper = msgspec.json.decode(raw_bytes)
        except Exception:
            return

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
            elif stream_name.endswith("@trade"):
                event_type = "trade"  # НОВОЕ: сырые сделки
            elif "@depth" in stream_name:
                event_type = "depthUpdate"
            else:
                return

        local_ts_ns = time.time_ns()

        if event_type in ("aggTrade", "trade"):  # Обрабатываем оба типа
            event = self._parse_trade(payload, local_ts_ns)
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

    def _parse_trade(
        self,
        payload: Dict[str, Any],
        local_ts_ns: int,
    ) -> Optional[TradeEvent]:
        """
        Парсит и @aggTrade, и @trade события.
        
        @aggTrade поля: s, p, q, a, T, m
        @trade поля: s, p, q, t, T, m
        """
        try:
            symbol = payload["s"].upper()

            if symbol not in self._symbols:
                return None

            event_time_ms = int(payload.get("E", 0))
            ts_exchange_ns = event_time_ms * 1_000_000 if event_time_ms else local_ts_ns

            # Trade ID: в @aggTrade это "a", в @trade это "t"
            trade_id = payload.get("a") or payload.get("t")

            return TradeEvent(
                ts_exchange_ns=ts_exchange_ns,
                ts_local_ns=local_ts_ns,
                symbol=symbol,
                price=float(payload["p"]),
                quantity=float(payload["q"]),
                is_buyer_maker=bool(payload["m"]),
                trade_id=int(trade_id) if trade_id is not None else None,
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
        """Запрашивает snapshot через REST для всех символов."""
        for symbol in self._symbols:
            try:
                # ВАЖНО: помечаем, что snapshot запрошен, ДО самого запроса
                await self._handler.on_snapshot_requested(symbol)
                
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
            pass