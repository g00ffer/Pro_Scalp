"""
Построитель ордеров для исполнения сигналов.

Отвечает за:
- Построение входных ордеров (market, marketable limit IOC)
- Построение защитных стопов (STOP_MARKET, reduceOnly)
- Построение тейк-профитов
- Нормализацию цен и объёмов до валидных значений
- Генерацию уникальных clientOrderId для идемпотентности

Принципы (из формализации системы):
- Вход: marketable LIMIT IOC (защита от проскальзывания)
- Стоп: STOP_MARKET + reduceOnly (гарантированное исполнение)
- Каждая нога имеет уникальный clientOrderId

Используется в связке с:
- BracketExecutor (bracket orders)
- ProtectionWatchdog (защита позиций)
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from proscalper.core.types import (
    EntryType,
    OrderSide,
    OrderType,
    TimeInForce,
)


@dataclass(frozen=True)
class OrderBuilderConfig:
    """
    Конфигурация построителя ордеров.
    
    Параметры из формализации системы (раздел 1.14):
    - Вход: marketable LIMIT IOC, max_slippage_ticks = 4
    - Стоп: буфер за уровнем/плотностью
    - Минимальный номинал ордера
    """
    # Входные ордера
    entry_order_type: EntryType = EntryType.MARKETABLE_LIMIT_IOC
    max_slippage_ticks: int = 4
    allow_market_on_ultra_liquidity: bool = True
    
    # Защитные стопы
    stop_buffer_ticks: int = 2
    stop_buffer_pct: float = 0.0002  # 0.02% от цены
    stop_working_type: str = "CONTRACT_PRICE"
    
    # Тейк-профиты
    use_take_profit: bool = False  # в скальпинге TP обычно не нужен
    
    # Лимиты
    min_order_notional: float = 5.0


@dataclass
class OrderRequest:
    """
    Запрос на создание ордера.
    
    Содержит все параметры для отправки на биржу.
    Используется в BracketExecutor и ProtectionWatchdog.
    """
    # Идентификация
    client_order_id: str
    signal_id: str
    
    # Параметры ордера
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    close_position: bool = False
    working_type: str = "CONTRACT_PRICE"
    
    # Тип ноги (для логирования)
    leg_type: str = ""  # "entry", "stop", "take_profit", "close"
    
    # Время создания
    created_ts_ns: int = 0
    
    @property
    def is_entry(self) -> bool:
        return self.leg_type == "entry"
    
    @property
    def is_stop(self) -> bool:
        return self.leg_type == "stop"
    
    @property
    def is_close(self) -> bool:
        return self.leg_type == "close"
    
    def to_exchange_dict(self) -> Dict:
        """
        Конвертирует в формат для отправки на биржу.
        
        Формат зависит от биржи:
        - Binance: camelCase параметры
        - Bybit: другие имена параметров
        """
        result = {
            "symbol": self.symbol,
            "side": self.side.value.upper(),
            "type": self.order_type.value,
            "quantity": self.quantity,
            "newClientOrderId": self.client_order_id,
        }
        
        if self.price is not None:
            result["price"] = self.price
        
        if self.stop_price is not None:
            result["stopPrice"] = self.stop_price
            result["workingType"] = self.working_type
        
        if self.time_in_force != TimeInForce.GTC:
            result["timeInForce"] = self.time_in_force.value.upper()
        
        if self.reduce_only:
            result["reduceOnly"] = True
        
        if self.close_position:
            result["closePosition"] = True
        
        return result


class OrderBuilder:
    """
    Построитель ордеров для исполнения сигналов.
    
    Чистые функции без побочных эффектов.
    Не знает про сеть и асинхронность.
    
    Использование:
        builder = OrderBuilder(
            config=OrderBuilderConfig(),
            tick_sizes={"BTCUSDT": 0.1},
            step_sizes={"BTCUSDT": 0.001},
        )
        
        # Входной ордер
        entry = builder.build_entry(
            signal_id="sig_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.001,
            best_bid=76000.3,
            best_ask=76000.5,
        )
        
        # Защитный стоп
        stop = builder.build_protective_stop(
            signal_id="sig_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.001,
            stop_price=75990.0,
        )
    """
    
    def __init__(
        self,
        config: Optional[OrderBuilderConfig] = None,
        tick_sizes: Optional[Dict[str, float]] = None,
        step_sizes: Optional[Dict[str, float]] = None,
        min_notionals: Optional[Dict[str, float]] = None,
    ):
        self.config = config or OrderBuilderConfig()
        self._tick_sizes = {k.upper(): v for k, v in (tick_sizes or {}).items()}
        self._step_sizes = {k.upper(): v for k, v in (step_sizes or {}).items()}
        self._min_notionals = {k.upper(): v for k, v in (min_notionals or {}).items()}
    
    # ============================================
    # Входные ордера
    # ============================================
    
    def build_entry(
        self,
        signal_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        best_bid: float,
        best_ask: float,
        spread_ticks: int = 0,
        top_depth_notional: float = 0.0,
    ) -> Optional[OrderRequest]:
        """
        Строит входной ордер для сигнала.
        
        Логика:
        1. По умолчанию: marketable LIMIT IOC
        2. Для сверхликвидных: чистый MARKET (если разрешено)
        
        Для покупки: цена = best_ask + max_slippage_ticks * tick_size
        Для продажи: цена = best_bid - max_slippage_ticks * tick_size
        """
        symbol = symbol.upper()
        tick_size = self._tick_sizes.get(symbol)
        step_size = self._step_sizes.get(symbol, 0.001)
        
        if tick_size is None or tick_size <= 0:
            return None
        
        # Нормализуем объём
        quantity = self._normalize_quantity(quantity, step_size)
        if quantity <= 0:
            return None
        
        # Проверяем минимальный номинал
        reference_price = best_ask if side == OrderSide.BUY else best_bid
        notional = quantity * reference_price
        min_notional = self._min_notionals.get(symbol, self.config.min_order_notional)
        
        if notional < min_notional:
            return None
        
        # Генерируем client_order_id
        client_order_id = self._generate_client_order_id(signal_id, "entry")
        
        # Определяем тип ордера
        use_market = (
            self.config.allow_market_on_ultra_liquidity
            and spread_ticks <= 1
            and top_depth_notional >= 500_000
        )
        
        if use_market:
            return OrderRequest(
                client_order_id=client_order_id,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                time_in_force=TimeInForce.GTC,
                reduce_only=False,
                leg_type="entry",
                created_ts_ns=time.time_ns(),
            )
        else:
            # Marketable LIMIT IOC
            if side == OrderSide.BUY:
                price = best_ask + self.config.max_slippage_ticks * tick_size
            else:
                price = best_bid - self.config.max_slippage_ticks * tick_size
            
            price = self._normalize_price(price, tick_size)
            
            return OrderRequest(
                client_order_id=client_order_id,
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                time_in_force=TimeInForce.IOC,
                reduce_only=False,
                leg_type="entry",
                created_ts_ns=time.time_ns(),
            )
    
    # ============================================
    # Защитные стопы
    # ============================================
    
    def build_protective_stop(
        self,
        signal_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
    ) -> Optional[OrderRequest]:
        """
        Строит защитный стоп-ордер.
        
        Использует:
        - STOP_MARKET для гарантированного исполнения
        - reduceOnly = True (закрытие позиции)
        - workingType = CONTRACT_PRICE
        """
        symbol = symbol.upper()
        tick_size = self._tick_sizes.get(symbol)
        step_size = self._step_sizes.get(symbol, 0.001)
        
        if tick_size is None or tick_size <= 0:
            return None
        
        # Нормализуем объём
        quantity = self._normalize_quantity(quantity, step_size)
        if quantity <= 0:
            return None
        
        if stop_price <= 0:
            return None
        
        # Нормализуем стоп-цену
        stop_price = self._normalize_price(stop_price, tick_size)
        
        # Генерируем client_order_id
        client_order_id = self._generate_client_order_id(signal_id, "stop")
        
        # Сторона ордера противоположна позиции
        close_side = side.opposite()
        
        return OrderRequest(
            client_order_id=client_order_id,
            signal_id=signal_id,
            symbol=symbol,
            side=close_side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=True,
            working_type=self.config.stop_working_type,
            leg_type="stop",
            created_ts_ns=time.time_ns(),
        )
    
    # ============================================
    # Аварийное закрытие
    # ============================================
    
    def build_emergency_close(
        self,
        signal_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> Optional[OrderRequest]:
        """
        Строит ордер аварийного закрытия позиции.
        
        Используется в ProtectionWatchdog при отказе защиты.
        Всегда MARKET для гарантированного исполнения.
        """
        symbol = symbol.upper()
        step_size = self._step_sizes.get(symbol, 0.001)
        
        # Нормализуем объём
        quantity = self._normalize_quantity(quantity, step_size)
        if quantity <= 0:
            return None
        
        client_order_id = self._generate_client_order_id(signal_id, "close")
        close_side = side.opposite()
        
        return OrderRequest(
            client_order_id=client_order_id,
            signal_id=signal_id,
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=True,
            leg_type="close",
            created_ts_ns=time.time_ns(),
        )
    
    # ============================================
    # Расчёт цен
    # ============================================
    
    def calculate_stop_price(
        self,
        symbol: str,
        side: OrderSide,
        level_price: float,
        density_price: Optional[float] = None,
        spread: float = 0.0,
    ) -> float:
        """
        Рассчитывает цену стоп-лосса.
        
        Логика (из формализации системы, раздел 1.14):
        1. Базовый стоп: за уровнем
        2. Если есть плотность за уровнем: за плотностью
        3. Буфер: макс(буфер_тики, буфер_процент, спред)
        """
        tick_size = self._tick_sizes.get(symbol, 0.01)
        
        # Определяем базовую цену
        base_price = density_price if density_price is not None else level_price
        
        # Рассчитываем буфер
        buffer_ticks = self.config.stop_buffer_ticks * tick_size
        buffer_pct = base_price * self.config.stop_buffer_pct
        buffer = max(buffer_ticks, buffer_pct, spread)
        
        if side == OrderSide.BUY:
            return base_price - buffer
        else:
            return base_price + buffer
    
    # ============================================
    # Утилиты
    # ============================================
    
    def set_tick_size(self, symbol: str, tick_size: float) -> None:
        """Устанавливает тик-сайз для символа."""
        self._tick_sizes[symbol.upper()] = tick_size
    
    def set_step_size(self, symbol: str, step_size: float) -> None:
        """Устанавливает степ-сайз для символа."""
        self._step_sizes[symbol.upper()] = step_size
    
    def set_min_notional(self, symbol: str, min_notional: float) -> None:
        """Устанавливает минимальный номинал для символа."""
        self._min_notionals[symbol.upper()] = min_notional
    
    def get_tick_size(self, symbol: str) -> Optional[float]:
        """Возвращает тик-сайз для символа."""
        return self._tick_sizes.get(symbol.upper())
    
    def get_step_size(self, symbol: str) -> Optional[float]:
        """Возвращает степ-сайз для символа."""
        return self._step_sizes.get(symbol.upper())
    
    def _normalize_price(self, price: float, tick_size: float) -> float:
        """Нормализует цену до валидного тика."""
        if tick_size <= 0:
            return price
        return round(round(price / tick_size) * tick_size, 10)
    
    def _normalize_quantity(self, quantity: float, step_size: float) -> float:
        """Нормализует объём до валидного шага (округление вниз)."""
        if step_size <= 0:
            return quantity
        return math.floor(quantity / step_size) * step_size
    
    def _generate_client_order_id(self, signal_id: str, leg: str) -> str:
        """
        Генерирует уникальный client_order_id.
        
        Формат: {signal_id}:{leg}:{unique_suffix}
        Обеспечивает идемпотентность при ретраях.
        """
        suffix = uuid.uuid4().hex[:8]
        return f"{signal_id}:{leg}:{suffix}"