"""
Приватный WebSocket Binance Futures (User Data Stream).

Получает события аккаунта в реальном времени:
- ORDER_TRADE_UPDATE: исполнение/отмена ордеров
- ACCOUNT_UPDATE: изменения баланса, маржи, позиций
- ACCOUNT_CONFIG_UPDATE: изменения leverage и других настроек

Работает через listenKey:
1. POST /fapi/v1/listenKey → получаем ключ (действителен 60 мин)
2. Подключаемся к wss://fstream.binance.com/ws/<listenKey>
3. Каждые 30 минут продлеваем ключ через PUT /fapi/v1/listenKey
4. При закрытии — DELETE /fapi/v1/listenKey

Отличие от публичного ws_market.py:
- требует аутентификации (API key)
- работает для всего аккаунта, а не для конкретных символов
- события приходят реже, но критичнее

Используется в связке с:
- exchange/binance_futures/auth.py (подпись запросов)
- exchange/binance_futures/rest.py (управление listenKey)
- execution/order_manager.py (обновление состояния ордеров)
- risk/position_manager.py (обновление позиций из ACCOUNT_UPDATE)
- app/live_runner.py (живая торговля)

Принципы:
- listenKey продлевается в фоне каждые 30 минут
- при disconnect автоматический reconnect с новым listenKey
- все события маршрутизируются через UserStreamHandler
- минимальная нагрузка на event loop
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from proscalper.exchange.binance_futures.auth import BinanceAuth
from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient


logger = logging.getLogger(__name__)


# ============================================================
# Типы событий
# ============================================================

class UserEventType(str, Enum):
    """Типы событий user data stream."""
    ORDER_TRADE_UPDATE = "ORDER_TRADE_UPDATE"
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
    ACCOUNT_CONFIG_UPDATE = "ACCOUNT_CONFIG_UPDATE"
    MARGIN_CALL = "MARGIN_CALL"
    LISTEN_KEY_EXPIRED = "LISTEN_KEY_EXPIRED"
    TRADE_LITE = "TRADE_LITE"


@dataclass
class OrderTradeUpdate:
    """
    Событие ORDER_TRADE_UPDATE.
    
    Приходит при:
    - создании ордера
    - частичном/полном исполнении
    - отмене ордера
    - отклонении
    """
    event_type: str = "ORDER_TRADE_UPDATE"
    event_time_ms: int = 0
    transaction_time_ms: int = 0
    
    # Ордер
    symbol: str = ""
    client_order_id: str = ""
    side: str = ""          # BUY/SELL
    order_type: str = ""    # MARKET/LIMIT/STOP_MARKET/...
    time_in_force: str = ""
    original_qty: float = 0.0
    original_price: float = 0.0
    avg_price: float = 0.0
    stop_price: float = 0.0
    
    # Исполнение
    last_filled_qty: float = 0.0
    last_filled_price: float = 0.0
    accumulated_filled_qty: float = 0.0
    accumulated_commission: float = 0.0
    
    # Статус
    order_status: str = ""  # NEW/PARTIALLY_FILLED/FILLED/CANCELED/REJECTED
    order_id: int = 0
    trade_id: int = 0
    
    # Метаданные
    reduce_only: bool = False
    working_type: str = ""
    original_order_type: str = ""
    position_side: str = ""  # LONG/SHORT/BOTH
    close_position: bool = False
    
    # Сырой JSON
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> OrderTradeUpdate:
        """Парсинг из сырого события Binance."""
        o = event.get("o", {})
        return cls(
            event_time_ms=int(event.get("E", 0)),
            transaction_time_ms=int(event.get("T", 0)),
            symbol=str(o.get("s", "")),
            client_order_id=str(o.get("c", "")),
            side=str(o.get("S", "")),
            order_type=str(o.get("o", "")),
            time_in_force=str(o.get("f", "")),
            original_qty=float(o.get("q", 0.0)),
            original_price=float(o.get("p", 0.0)),
            avg_price=float(o.get("ap", 0.0)),
            stop_price=float(o.get("sp", 0.0)),
            last_filled_qty=float(o.get("l", 0.0)),
            last_filled_price=float(o.get("L", 0.0)),
            accumulated_filled_qty=float(o.get("z", 0.0)),
            accumulated_commission=float(o.get("n", 0.0)),
            order_status=str(o.get("X", "")),
            order_id=int(o.get("i", 0)),
            trade_id=int(o.get("t", 0)),
            reduce_only=bool(o.get("R", False)),
            working_type=str(o.get("wt", "")),
            original_order_type=str(o.get("ot", "")),
            position_side=str(o.get("ps", "")),
            close_position=bool(o.get("cp", False)),
            raw=event,
        )
    
    @property
    def is_fill(self) -> bool:
        """Это событие исполнения (частичного или полного)?"""
        return self.order_status in ("FILLED", "PARTIALLY_FILLED")
    
    @property
    def is_cancel(self) -> bool:
        """Это событие отмены?"""
        return self.order_status == "CANCELED"
    
    @property
    def is_reject(self) -> bool:
        """Это событие отклонения?"""
        return self.order_status in ("REJECTED", "EXPIRED")


@dataclass
class AccountUpdate:
    """
    Событие ACCOUNT_UPDATE.
    
    Приходит при изменениях:
    - баланса
    - доступной маржи
    - позиций (PnL, размер)
    """
    event_time_ms: int = 0
    transaction_id: int = 0
    event_reason: str = ""  # DEPOSIT/WITHDRAW/ORDER/FUNDING_FEE/...
    
    # Баланс по активам
    balances: List[Dict[str, Any]] = field(default_factory=list)
    
    # Позиции
    positions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Сырой JSON
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> AccountUpdate:
        """Парсинг из сырого события."""
        a = event.get("a", {})
        return cls(
            event_time_ms=int(event.get("E", 0)),
            transaction_id=int(a.get("T", 0)),
            event_reason=str(a.get("m", "")),
            balances=list(a.get("B", [])),
            positions=list(a.get("P", [])),
            raw=event,
        )
    
    def get_balance(self, asset: str = "USDT") -> Optional[Dict[str, Any]]:
        """Возвращает баланс указанного актива."""
        for b in self.balances:
            if b.get("a") == asset:
                return b
        return None
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Возвращает позицию по символу."""
        symbol = symbol.upper()
        for p in self.positions:
            if p.get("s", "").upper() == symbol:
                return p
        return None


@dataclass
class AccountConfigUpdate:
    """Событие изменения конфигурации аккаунта."""
    event_time_ms: int = 0
    symbol: str = ""
    leverage: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_event(cls, event: Dict[str, Any]) -> AccountConfigUpdate:
        ac = event.get("ac", {})
        return cls(
            event_time_ms=int(event.get("E", 0)),
            symbol=str(ac.get("s", "")),
            leverage=int(ac.get("l", 0)),
            raw=event,
        )


# ============================================================
# Интерфейс обработчика
# ============================================================

class UserStreamHandler(ABC):
    """
    Интерфейс обработчика событий user data stream.
    
    Реализации должны обрабатывать события, приходящие от биржи.
    """
    
    @abstractmethod
    async def on_order_update(self, update: OrderTradeUpdate) -> None:
        """Вызывается при ORDER_TRADE_UPDATE."""
        ...
    
    @abstractmethod
    async def on_account_update(self, update: AccountUpdate) -> None:
        """Вызывается при ACCOUNT_UPDATE."""
        ...
    
    async def on_account_config_update(
        self, update: AccountConfigUpdate
    ) -> None:
        """Вызывается при ACCOUNT_CONFIG_UPDATE (по умолчанию — no-op)."""
        pass
    
    async def on_margin_call(self, event: Dict[str, Any]) -> None:
        """Вызывается при MARGIN_CALL (по умолчанию — no-op)."""
        pass
    
    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        """Вызывается при изменениях состояния соединения."""
        pass


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class UserStreamWSConfig:
    """Конфигурация приватного WebSocket."""
    # Base URL
    base_url: str = "wss://fstream.binance.com"
    
    # Интервал продления listenKey (в секундах)
    # Binance рекомендует продлевать каждые 30 минут
    # listenKey живёт 60 минут
    keepalive_interval_sec: float = 30 * 60
    
    # Таймауты
    reconnect_delay_sec: float = 1.0
    max_reconnect_delay_sec: float = 60.0
    ping_interval_sec: float = 20.0
    ping_timeout_sec: float = 20.0
    
    # Лимит reconnect attempts
    max_reconnect_attempts: int = 100


# ============================================================
# Основной класс
# ============================================================

class BinanceFuturesUserStreamWS:
    """
    Приватный WebSocket для Binance Futures User Data Stream.
    
    Использование:
        handler = MyUserStreamHandler()
        ws = BinanceFuturesUserStreamWS(config, auth, handler)
        
        # Запуск (обычно через asyncio.create_task)
        task = asyncio.create_task(ws.run())
        
        # Остановка
        await ws.stop()
        task.cancel()
    
    Автоматически:
    - получает listenKey через REST
    - продлевает listenKey каждые 30 минут
    - переподключается при disconnect
    - маршрутизирует события в handler
    """
    
    def __init__(
        self,
        config: UserStreamWSConfig,
        auth: BinanceAuth,
        handler: UserStreamHandler,
    ) -> None:
        self._config = config
        self._auth = auth
        self._handler = handler
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._listen_key: Optional[str] = None
        self._listen_key_obtained_ts: float = 0.0
        
        self._running = False
        self._stop_event = asyncio.Event()
        self._keepalive_task: Optional[asyncio.Task] = None
        
        # Счётчики для диагностики
        self._reconnect_attempts = 0
        self._messages_received = 0
        self._errors_count = 0
    
    # ============================================
    # Публичный API
    # ============================================
    
    async def run(self) -> None:
        """
        Основной цикл: подключение, приём сообщений, reconnect.
        """
        self._running = True
        
        while self._running and not self._stop_event.is_set():
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._errors_count += 1
                logger.error(f"UserStreamWS error: {exc}")
                
                if self._reconnect_attempts >= self._config.max_reconnect_attempts:
                    logger.critical(
                        f"UserStreamWS: max reconnect attempts reached "
                        f"({self._config.max_reconnect_attempts})"
                    )
                    await self._handler.on_state(
                        "FATAL_ERROR",
                        {"reason": "max_reconnect_attempts"},
                    )
                    break
                
                # Exponential backoff
                delay = min(
                    self._config.reconnect_delay_sec
                    * (2 ** min(self._reconnect_attempts, 6)),
                    self._config.max_reconnect_delay_sec,
                )
                self._reconnect_attempts += 1
                
                logger.info(
                    f"UserStreamWS: reconnecting in {delay:.1f}s "
                    f"(attempt {self._reconnect_attempts})"
                )
                await asyncio.sleep(delay)
        
        self._running = False
    
    async def stop(self) -> None:
        """Останавливает работу."""
        self._running = False
        self._stop_event.set()
        
        # Останавливаем keepalive
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
        
        # Закрываем WebSocket
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        
        # Удаляем listenKey на сервере
        if self._listen_key is not None:
            await self._delete_listen_key()
            self._listen_key = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Диагностическая статистика."""
        return {
            "running": self._running,
            "connected": self._ws is not None,
            "reconnect_attempts": self._reconnect_attempts,
            "messages_received": self._messages_received,
            "errors_count": self._errors_count,
            "listen_key_age_sec": (
                time.time() - self._listen_key_obtained_ts
                if self._listen_key_obtained_ts > 0
                else 0.0
            ),
        }
    
    # ============================================
    # Внутренние методы: подключение
    # ============================================
    
    async def _connect_and_listen(self) -> None:
        """Подключается и слушает сообщения."""
        # Получаем listenKey
        self._listen_key = await self._create_listen_key()
        self._listen_key_obtained_ts = time.time()
        self._reconnect_attempts = 0
        
        # Запускаем keepalive в фоне
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        
        ws_url = f"{self._config.base_url}/ws/{self._listen_key}"
        
        logger.info(f"UserStreamWS: connecting to {ws_url}")
        await self._handler.on_state(
            "CONNECTING",
            {"url": ws_url},
        )
        
        try:
            async with websockets.connect(
                ws_url,
                ping_interval=self._config.ping_interval_sec,
                ping_timeout=self._config.ping_timeout_sec,
                close_timeout=5.0,
            ) as ws:
                self._ws = ws
                
                logger.info("UserStreamWS: connected")
                await self._handler.on_state("CONNECTED", {})
                
                # Слушаем сообщения
                await self._listen_loop()
        
        except (ConnectionClosed, ConnectionClosedError) as exc:
            logger.warning(f"UserStreamWS: connection closed: {exc}")
            await self._handler.on_state(
                "DISCONNECT",
                {"code": getattr(exc, "code", None), "reason": str(exc)},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"UserStreamWS: unexpected error: {exc}")
            await self._handler.on_state(
                "ERROR",
                {"error": str(exc)},
            )
            raise
        finally:
            self._ws = None
            
            # Останавливаем keepalive
            if self._keepalive_task is not None:
                self._keepalive_task.cancel()
                try:
                    await self._keepalive_task
                except asyncio.CancelledError:
                    pass
                self._keepalive_task = None
    
    async def _listen_loop(self) -> None:
        """Цикл приёма сообщений."""
        assert self._ws is not None
        
        async for message in self._ws:
            if self._stop_event.is_set():
                break
            
            self._messages_received += 1
            
            try:
                await self._dispatch_message(message)
            except Exception as exc:
                logger.error(f"UserStreamWS: error dispatching message: {exc}")
                self._errors_count += 1
    
    async def _dispatch_message(self, message: str) -> None:
        """Диспетчеризует входящее сообщение."""
        # Проверяем, что это не ping (websockets обрабатывает их автоматически)
        if not message or message == "":
            return
        
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"UserStreamWS: invalid JSON: {message[:100]}")
            return
        
        if not isinstance(event, dict):
            return
        
        event_type = event.get("e")
        
        if event_type == "ORDER_TRADE_UPDATE":
            update = OrderTradeUpdate.from_event(event)
            await self._handler.on_order_update(update)
        
        elif event_type == "ACCOUNT_UPDATE":
            update = AccountUpdate.from_event(event)
            await self._handler.on_account_update(update)
        
        elif event_type == "ACCOUNT_CONFIG_UPDATE":
            update = AccountConfigUpdate.from_event(event)
            await self._handler.on_account_config_update(update)
        
        elif event_type == "MARGIN_CALL":
            await self._handler.on_margin_call(event)
        
        else:
            logger.debug(f"UserStreamWS: unknown event type: {event_type}")
    
    # ============================================
    # Внутренние методы: управление listenKey
    # ============================================
    
    async def _create_listen_key(self) -> str:
        """Создаёт новый listenKey через REST."""
        async with BinanceFuturesRestClient() as rest:
            response = await rest._client.post(
                f"{self._auth.base_url}/fapi/v1/listenKey",
                headers=self._auth.get_headers(),
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to create listenKey: {response.status_code} "
                    f"{response.text}"
                )
            
            data = response.json()
            listen_key = data.get("listenKey")
            if not listen_key:
                raise RuntimeError(f"No listenKey in response: {data}")
            
            logger.info(f"UserStreamWS: obtained listenKey (60 min TTL)")
            return listen_key
    
    async def _keepalive_listen_key(self) -> None:
        """Продлевает listenKey."""
        if self._listen_key is None:
            return
        
        try:
            async with BinanceFuturesRestClient() as rest:
                response = await rest._client.put(
                    f"{self._auth.base_url}/fapi/v1/listenKey",
                    headers=self._auth.get_headers(),
                )
                
                if response.status_code == 200:
                    self._listen_key_obtained_ts = time.time()
                    logger.debug("UserStreamWS: listenKey extended")
                else:
                    logger.warning(
                        f"UserStreamWS: keepalive failed: "
                        f"{response.status_code}"
                    )
        except Exception as exc:
            logger.error(f"UserStreamWS: keepalive error: {exc}")
    
    async def _delete_listen_key(self) -> None:
        """Удаляет listenKey при закрытии."""
        if self._listen_key is None:
            return
        
        try:
            async with BinanceFuturesRestClient() as rest:
                response = await rest._client.delete(
                    f"{self._auth.base_url}/fapi/v1/listenKey",
                    headers=self._auth.get_headers(),
                )
                
                if response.status_code == 200:
                    logger.info("UserStreamWS: listenKey deleted")
                else:
                    logger.warning(
                        f"UserStreamWS: delete listenKey failed: "
                        f"{response.status_code}"
                    )
        except Exception as exc:
            logger.warning(f"UserStreamWS: error deleting listenKey: {exc}")
    
    async def _keepalive_loop(self) -> None:
        """Фоновый цикл продления listenKey."""
        try:
            while self._running and not self._stop_event.is_set():
                await asyncio.sleep(self._config.keepalive_interval_sec)
                
                if self._stop_event.is_set():
                    break
                
                await self._keepalive_listen_key()
        except asyncio.CancelledError:
            pass