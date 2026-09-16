"""
Базовые перечисления и типы для всей системы.
Используем enum.Enum для бизнес-логики и msgspec.Struct для сериализации.
"""
from enum import Enum, auto
from typing import NewType
import msgspec


# === Базовые типы для type safety ===
Symbol = NewType('Symbol', str)
Price = NewType('Price', float)
Quantity = NewType('Quantity', float)
Notional = NewType('Notional', float)
Timestamp = NewType('Timestamp', int)  # nanoseconds since epoch
PriceTick = NewType('PriceTick', int)  # цена в тиках (integer)


class OrderSide(str, Enum):
    """Сторона ордера."""
    BUY = "BUY"
    SELL = "SELL"
    
    def opposite(self) -> 'OrderSide':
        """Возвращает противоположную сторону."""
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class PositionSide(str, Enum):
    """Сторона позиции."""
    LONG = "LONG"
    SHORT = "SHORT"
    
    def to_order_side(self) -> OrderSide:
        """Возвращает сторону ордера для закрытия позиции."""
        return OrderSide.SELL if self == PositionSide.LONG else OrderSide.BUY


class OrderType(str, Enum):
    """Тип ордера."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class TimeInForce(str, Enum):
    """Время жизни ордера."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Post Only


class OrderStatus(str, Enum):
    """Статус ордера."""
    PENDING = "PENDING"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class BookSide(str, Enum):
    """Сторона стакана."""
    BID = "BID"
    ASK = "ASK"


class LevelState(str, Enum):
    """Состояние уровня."""
    FORMING = "FORMING"
    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    BROKEN = "BROKEN"
    FAILED_BREAK = "FAILED_BREAK"
    INVALID = "INVALID"
    RETESTABLE = "RETESTABLE"


class LevelSide(str, Enum):
    """Тип уровня."""
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class EntryType(str, Enum):
    """Тип входа."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    MARKETABLE_LIMIT_IOC = "MARKETABLE_LIMIT_IOC"


class ExecutionStatus(str, Enum):
    """Статус исполнения."""
    PENDING = "PENDING"
    PENDING_PROTECTION = "PENDING_PROTECTION"
    PROTECTED = "PROTECTED"
    PROTECTION_FAILED = "PROTECTION_FAILED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


class SignalRejectReason(str, Enum):
    """Причины отклонения сигнала."""
    FILTERS_FAILED = "FILTERS_FAILED"
    NO_COMPRESSION = "NO_COMPRESSION"
    PRICE_NOT_CROSSED = "PRICE_NOT_CROSSED"
    WEAK_VOLUME = "WEAK_VOLUME"
    WEAK_DELTA = "WEAK_DELTA"
    NEGATIVE_IMBALANCE = "NEGATIVE_IMBALANCE"
    WALL_NOT_CONSUMED = "WALL_NOT_CONSUMED"
    OPPOSING_WALL_TOO_STRONG = "OPPOSING_WALL_TOO_STRONG"
    STOP_DISTANCE_INVALID = "STOP_DISTANCE_INVALID"
    BOOKTICKER_STALE = "BOOKTICKER_STALE"
    PRICE_ALREADY_TOO_FAR = "PRICE_ALREADY_TOO_FAR"
    FAST_REVERSAL = "FAST_REVERSAL"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    BOOK_LAG_TOO_HIGH = "BOOK_LAG_TOO_HIGH"
    RISK_LIMIT_REACHED = "RISK_LIMIT_REACHED"
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    KILL_SWITCH = "KILL_SWITCH"


class IncidentType(str, Enum):
    """Типы инцидентов."""
    WS_DISCONNECT = "WS_DISCONNECT"
    BOOK_OUT_OF_SYNC = "BOOK_OUT_OF_SYNC"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    STOP_PLACE_FAILED = "STOP_PLACE_FAILED"
    PROTECTION_TIMEOUT = "PROTECTION_TIMEOUT"
    ENTRY_NO_FILL_TIMEOUT = "ENTRY_NO_FILL_TIMEOUT"
    PARTIAL_FILL_UNPROTECTED = "PARTIAL_FILL_UNPROTECTED"
    EXCHANGE_RATE_LIMIT = "EXCHANGE_RATE_LIMIT"
    MARGIN_REJECT = "MARGIN_REJECT"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"


class Priority(str, Enum):
    """Приоритет задачи."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# === Базовые структуры данных ===

class InstrumentInfo(msgspec.Struct, frozen=True):
    """Информация об инструменте с биржи."""
    symbol: Symbol
    base_asset: str
    quote_asset: str
    tick_size: float  # минимальный шаг цены
    step_size: float  # минимальный шаг объёма
    min_notional: float  # минимальный номинал
    price_precision: int
    quantity_precision: int
    max_leverage: float = 1.0
    is_trading: bool = True


class LevelZone(msgspec.Struct, frozen=True):
    """Зона уровня (не одна цена, а диапазон)."""
    lower: float
    upper: float
    center: float
    
    def contains(self, price: float) -> bool:
        """Проверяет, находится ли цена в зоне."""
        return self.lower <= price <= self.upper
    
    def distance_to(self, price: float) -> float:
        """Расстояние от цены до зоны (0 если внутри)."""
        if self.contains(price):
            return 0.0
        return min(abs(price - self.lower), abs(price - self.upper))


class Wall(msgspec.Struct):
    """Крупная плотность в стакане."""
    price: float
    current_notional: float
    max_notional: float
    age_ms: int
    executed_notional: float
    cancelled_notional: float
    replenish_count: int
    spoof_score: float = 0.0
    iceberg_score: float = 0.0


class LevelLiquidity(msgspec.Struct):
    """Ликвидность вокруг уровня."""
    level_id: str
    ask_wall: Wall | None = None
    bid_wall: Wall | None = None
    ask_wall_consumed: bool = False
    bid_support_score: float = 0.0
    imbalance: float = 0.0
    spoof_score_ask: float = 0.0
    spoof_score_bid: float = 0.0
    iceberg_score_ask: float = 0.0
    iceberg_score_bid: float = 0.0
    spread_ticks: int = 0
    top_depth_notional: float = 0.0