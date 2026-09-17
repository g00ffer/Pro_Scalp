"""
Абстракция биржевого интерфейса.

Определяет единый протокол для работы с биржами. Сейчас
используется только Binance Futures, но структура позволяет
добавить Bybit или другие биржи без переписывания бизнес-логики.

Принципы:
- интерфейс минимальный (только то, что реально нужно)
- все методы асинхронные
- ошибки выбрасываются как исключения (не возвращаются)
- нет привязки к конкретной бирже в сигнатурах

Используется в связке с:
- exchange/binance_futures/* (реализация для Binance)
- exchange/bybit_v5/* (будущая реализация для Bybit)
- execution/execution_engine.py (вызывает через этот интерфейс)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


# ============================================================
# Общие типы
# ============================================================

class ExchangeName(str, Enum):
    """Имя биржи."""
    BINANCE_FUTURES = "binance_futures"
    BYBIT_V5 = "bybit_v5"


class ExchangeOrderType(str, Enum):
    """Унифицированный тип ордера."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class ExchangeOrderSide(str, Enum):
    """Унифицированная сторона ордера."""
    BUY = "BUY"
    SELL = "SELL"


class ExchangeOrderStatus(str, Enum):
    """Унифицированный статус ордера."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class ExchangeOrderRequest:
    """Унифицированный запрос на создание ордера."""
    symbol: str
    side: ExchangeOrderSide
    order_type: ExchangeOrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    reduce_only: bool = False
    time_in_force: Optional[str] = None


@dataclass(frozen=True)
class ExchangeOrderResponse:
    """Унифицированный ответ на создание ордера."""
    exchange: ExchangeName
    order_id: str
    client_order_id: str
    symbol: str
    side: ExchangeOrderSide
    order_type: ExchangeOrderType
    price: float
    quantity: float
    status: ExchangeOrderStatus
    raw: Dict[str, Any]


@dataclass(frozen=True)
class ExchangePosition:
    """Унифицированная позиция на бирже."""
    exchange: ExchangeName
    symbol: str
    side: str              # "LONG" / "SHORT"
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: float
    liquidation_price: float


@dataclass(frozen=True)
class ExchangeBalance:
    """Унифицированный баланс."""
    exchange: ExchangeName
    asset: str
    balance: float
    available_balance: float


# ============================================================
# Протокол публичных данных
# ============================================================

class PublicDataProtocol(Protocol):
    """Протокол для публичных рыночных данных."""

    @property
    def exchange_name(self) -> ExchangeName: ...

    async def get_exchange_info(self) -> List[Dict[str, Any]]:
        """Загрузка информации об инструментах."""
        ...

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 288,
    ) -> List[List]:
        """Загрузка исторических свечей."""
        ...


# ============================================================
# Протокол приватных данных
# ============================================================

class PrivateDataProtocol(Protocol):
    """Протокол для приватных данных аккаунта."""

    async def get_balances(self) -> List[ExchangeBalance]:
        """Балансы по активам."""
        ...

    async def get_positions(self) -> List[ExchangePosition]:
        """Открытые позиции."""
        ...

    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Открытые ордера."""
        ...


# ============================================================
# Протокол торговли
# ============================================================

class TradingProtocol(Protocol):
    """Протокол для отправки ордеров."""

    async def create_order(
        self, request: ExchangeOrderRequest
    ) -> ExchangeOrderResponse:
        """Отправка ордера."""
        ...

    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> ExchangeOrderResponse:
        """Отмена ордера."""
        ...

    async def query_order(
        self, symbol: str, order_id: str
    ) -> ExchangeOrderResponse:
        """Запрос статуса ордера."""
        ...


# ============================================================
# Абстрактный базовый класс
# ============================================================

class ExchangeBase(ABC):
    """
    Абстрактная биржа.

    Объединяет все протоколы в единый класс-наследник.
    Реализации (BinanceFuturesExchange, BybitV5Exchange)
    наследуются от этого класса.

    Пример реализации:
        class BinanceFuturesExchange(ExchangeBase):
            @property
            def exchange_name(self):
                return ExchangeName.BINANCE_FUTURES

            async def get_exchange_info(self):
                ...

            async def create_order(self, request):
                ...
    """

    @property
    @abstractmethod
    def exchange_name(self) -> ExchangeName:
        """Имя биржи."""
        ...

    # --- Публичные данные ---

    @abstractmethod
    async def get_exchange_info(self) -> List[Dict[str, Any]]:
        """Загрузка информации об инструментах."""
        ...

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 288,
    ) -> List[List]:
        """Загрузка исторических свечей."""
        ...

    # --- Приватные данные ---

    @abstractmethod
    async def get_balances(self) -> List[ExchangeBalance]:
        """Балансы по активам."""
        ...

    @abstractmethod
    async def get_positions(self) -> List[ExchangePosition]:
        """Открытые позиции."""
        ...

    @abstractmethod
    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Открытые ордера."""
        ...

    # --- Торговля ---

    @abstractmethod
    async def create_order(
        self, request: ExchangeOrderRequest
    ) -> ExchangeOrderResponse:
        """Отправка ордера."""
        ...

    @abstractmethod
    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> ExchangeOrderResponse:
        """Отмена ордера."""
        ...

    @abstractmethod
    async def query_order(
        self, symbol: str, order_id: str
    ) -> ExchangeOrderResponse:
        """Запрос статуса ордера."""
        ...