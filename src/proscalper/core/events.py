"""
События системы. Используют msgspec.Struct для быстрой сериализации.
Все события имеют timestamps для синхронизации и аудита.
"""
from typing import Optional
import msgspec

from proscalper.core.types import (
    Symbol, OrderSide, OrderType, TimeInForce, OrderStatus,
    BookSide, EntryType, SignalRejectReason, ExecutionStatus
)


# === События рыночных данных ===

class TradeEvent(msgspec.Struct):
    """Сделка из ленты (aggTrade / publicTrade)."""
    ts_exchange_ns: int  # время на бирже (наносекунды)
    ts_local_ns: int     # время получения локально
    symbol: str
    price: float
    quantity: float
    is_buyer_maker: bool  # True = продавец агрессор, False = покупатель агрессор
    trade_id: Optional[int] = None
    
    @property
    def aggressor_side(self) -> OrderSide:
        """Определяет агрессора сделки."""
        return OrderSide.SELL if self.is_buyer_maker else OrderSide.BUY
    
    @property
    def notional(self) -> float:
        """Номинал сделки."""
        return self.price * self.quantity


class BookTickerEvent(msgspec.Struct):
    """Быстрый best bid/ask (bookTicker)."""
    ts_exchange_ns: int
    ts_local_ns: int
    symbol: str
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    update_id: Optional[int] = None
    
    @property
    def spread(self) -> float:
        """Спред в цене."""
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        """Средняя цена."""
        return (self.bid_price + self.ask_price) / 2


class BookLevel(msgspec.Struct):
    """Один уровень стакана."""
    price: float
    quantity: float


class BookSnapshotEvent(msgspec.Struct):
    """Полный срез стакана (snapshot)."""
    ts_exchange_ns: int
    ts_local_ns: int
    symbol: str
    bids: list[BookLevel]
    asks: list[BookLevel]
    first_update_id: int
    last_update_id: int


class BookDeltaEvent(msgspec.Struct):
    """Изменение в стакане (delta)."""
    ts_exchange_ns: int
    ts_local_ns: int
    symbol: str
    side: BookSide
    price: float
    quantity: float  # 0 = удаление уровня
    first_update_id: int
    last_update_id: int
    prev_update_id: int  # для проверки последовательности
    
    @property
    def is_delete(self) -> bool:
        """Удаление уровня."""
        return self.quantity == 0
    
    @property
    def is_update(self) -> bool:
        """Обновление уровня."""
        return self.quantity > 0


class MarketSnapshot(msgspec.Struct):
    """Агрегированный снимок рынка (для сигналов)."""
    ts_ns: int
    symbol: str
    
    # Текущие цены
    last_price: float
    bid_price: float
    ask_price: float
    mid_price: float
    spread_ticks: int
    
    # Статистика ленты (rolling window)
    trades_per_sec: float
    buy_volume_1s: float
    sell_volume_1s: float
    net_delta_1s: float
    aggressor_volume_1s: float
    
    # Статистика стакана
    book_imbalance: float
    top_bid_depth: float
    top_ask_depth: float
    
    # Задержки
    book_lag_ms: int
    bookticker_age_ms: int
    
    # Мин/макс за последние N мс
    min_price_since_cross: float
    max_price_since_cross: float


# === События сигналов ===

class SignalFeatures(msgspec.Struct):
    """Снимок фичей в момент генерации сигнала."""
    tape_acceleration: float = 0.0
    delta_strength: float = 0.0
    volume_strength: float = 0.0
    book_strength: float = 0.0
    impulse_score: float = 0.0
    compression_percentile: float = 0.0
    wall_consumed_score: float = 0.0
    spoof_score: float = 0.0
    iceberg_score: float = 0.0


class Signal(msgspec.Struct):
    """Торговый сигнал."""
    signal_id: str
    symbol: str
    side: OrderSide
    entry_type: EntryType
    level_id: str
    level_price: float
    
    # Расчётные цены
    entry_price: Optional[float] = None  # для LIMIT
    stop_price: float = 0.0
    take_profit_price: Optional[float] = None
    
    # Объём
    quantity: float = 0.0
    notional: float = 0.0
    
    # Причины и фичи
    reasons: list[str] = msgspec.field(default_factory=list)
    features: Optional[SignalFeatures] = None
    
    # Timestamps
    created_ts_ns: int = 0
    decision_ts_ns: int = 0
    
    # Метаданные
    strategy_version: str = "1.0.0"
    config_hash: str = ""


class SignalRejection(msgspec.Struct):
    """Отклонение сигнала."""
    signal_id: str
    symbol: str
    reason: SignalRejectReason
    details: dict = msgspec.field(default_factory=dict)
    ts_ns: int = 0


# === События исполнения ===

class OrderEvent(msgspec.Struct):
    """Событие ордера (создание, обновление, отмена)."""
    # Обязательные поля (без дефолтных значений)
    ts_ns: int
    symbol: str
    client_order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    
    # Опциональные поля (с дефолтными значениями)
    exchange_order_id: Optional[str] = None
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    signal_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    leg_type: str = ""


class FillEvent(msgspec.Struct):
    """Исполнение ордера (fill)."""
    ts_ns: int
    symbol: str
    client_order_id: str
    exchange_order_id: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    commission_asset: str
    trade_id: Optional[int] = None
    is_maker: bool = False
    
    @property
    def notional(self) -> float:
        """Номинал исполнения."""
        return self.price * self.quantity


class PositionUpdate(msgspec.Struct):
    """Обновление позиции."""
    ts_ns: int
    symbol: str
    side: OrderSide
    position_qty: float
    entry_price: float
    unrealized_pnl: float
    mark_price: float
    leverage: float
    is_protected: bool = False
    stop_order_id: Optional[str] = None

class BookDeltaBatchEvent(msgspec.Struct):
    """
    Пакетное изменение стакана.

    Один WebSocket depthUpdate может содержать сразу несколько изменений
    по bid/ask, поэтому для корректной синхронизации последовательности
    лучше обрабатывать его как один батч.
    """
    ts_exchange_ns: int
    ts_local_ns: int
    symbol: str
    first_update_id: int
    last_update_id: int
    prev_update_id: int
    bids: list[BookLevel]
    asks: list[BookLevel]