"""
Управление позициями после входа.

Реализует логику стратегии:
- Вход в начале импульса
- Выход в точке экстремума
- Цели — следующие уровни
- Частичная фиксация при сильном движении
- Безубыток — когда цена отбивает комиссии биржи

Логика управления:
1. Вход в позицию
2. Стоп за уровнем
3. Тейк 1 на ближайшем уровне (закрываем 50%)
4. Тейк 2 на следующем уровне (закрываем 30%)
5. Оставшиеся 20% тянем с трейлингом до экстремума
6. Безубыток при покрытии комиссий

Используется в связке с:
- SignalGenerator (сигналы на вход)
- PositionSizer (размеры позиций)
- ExecutionEngine (исполнение сделок)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.core.types import OrderSide
from proscalper.signals.signal_generator import TradingSignal
from proscalper.risk.sizing import PositionSize


class PositionState(Enum):
    """
    Состояние управляемой позиции.
    
    Жизненный цикл:
        OPENING → ACTIVE → PARTIALLY_CLOSED → TRAILING → CLOSING → CLOSED
    """
    OPENING = auto()            # Позиция открывается
    ACTIVE = auto()             # Позиция активна, цели ещё не достигнуты
    PARTIALLY_CLOSED = auto()   # Частично закрыта (первая цель достигнута)
    TRAILING = auto()           # Трейлинг до экстремума (вторая цель достигнута)
    CLOSING = auto()            # Позиция закрывается
    CLOSED = auto()             # Позиция полностью закрыта
    STOPPED = auto()            # Позиция закрыта по стоп-лоссу


@dataclass
class TakeProfitLevel:
    """
    Цель (тейк-профит) позиции.
    
    Каждая цель имеет:
    - Цену (на уровне)
    - Процент позиции для закрытия
    - Статус достижения
    """
    price: float
    quantity_pct: float         # % от исходной позиции (0.5, 0.3, 0.2)
    is_reached: bool = False
    reached_ts_ns: int = 0
    level_id: Optional[str] = None  # связанный уровень
    
    @property
    def is_pending(self) -> bool:
        """Цель ещё не достигнута."""
        return not self.is_reached


@dataclass
class PositionConfig:
    """
    Конфигурация управления позицией.
    
    Все параметры подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Цели (тейк-профиты)
    num_take_profit_levels: int = 2           # количество целей
    take_profit_split: List[float] = field(
        default_factory=lambda: [0.5, 0.3, 0.2]
    )  # деление позиции: 50% на первой цели, 30% на второй, 20% трейлинг
    
    # Безубыток
    taker_fee_pct: float = 0.04               # комиссия тейкера (%)
    maker_fee_pct: float = 0.02               # комиссия мейкера (%)
    breakeven_buffer_ticks: int = 0           # буфер безубытка (0 = только комиссии)
    
    # Трейлинг
    trailing_stop_ticks: int = 10             # трейлинг-стоп в тиках
    trailing_activation_ticks: int = 5        # активация трейлинга после цели
    
    # Стоп-лосс
    stop_loss_buffer_ticks: int = 3           # буфер стопа за уровнем
    
    # Ограничения
    max_position_age_sec: float = 3600.0      # максимальное время позиции (1 час)


@dataclass
class ManagedPosition:
    """
    Управляемая позиция.
    
    Содержит всю информацию о позиции и её состоянии.
    """
    # Идентификация
    position_id: str
    symbol: str
    direction: OrderSide
    
    # Вход
    entry_price: float
    entry_ts_ns: int
    initial_quantity: float
    current_quantity: float
    
    # Стоп-лосс
    stop_loss_price: float
    is_stop_at_breakeven: bool = False
    
    # Цели (тейк-профиты)
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)
    
    # Безубыток
    breakeven_price: float = 0.0
    is_breakeven_reached: bool = False
    
    # Трейлинг
    trailing_stop_price: float = 0.0
    is_trailing_active: bool = False
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = float('inf')
    
    # Состояние
    state: PositionState = PositionState.OPENING
    
    # Результаты
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Комиссии
    total_fees: float = 0.0
    
    @property
    def is_long(self) -> bool:
        """Позиция в лонг."""
        return self.direction == OrderSide.BUY
    
    @property
    def is_short(self) -> bool:
        """Позиция в шорт."""
        return self.direction == OrderSide.SELL
    
    @property
    def remaining_quantity_pct(self) -> float:
        """Оставшийся размер позиции в % от исходного."""
        if self.initial_quantity <= 0:
            return 0.0
        return self.current_quantity / self.initial_quantity
    
    @property
    def is_fully_closed(self) -> bool:
        """Позиция полностью закрыта."""
        return self.current_quantity <= 0 or self.state == PositionState.CLOSED
    
    @property
    def num_reached_targets(self) -> int:
        """Количество достигнутых целей."""
        return sum(1 for tp in self.take_profit_levels if tp.is_reached)


class PositionManager:
    """
    Менеджер управления позициями.
    
    Отвечает за:
    - Открытие позиций
    - Отслеживание целей (тейк-профитов)
    - Частичную фиксацию прибыли
    - Перемещение стопа в безубыток
    - Трейлинг до точки экстремума
    - Закрытие позиций
    
    Использование:
        manager = PositionManager(config)
        
        # Открываем позицию
        position = manager.open_position(signal, sizing)
        
        # На каждом обновлении цены
        manager.on_price_update(price, ts_ns)
        
        # Проверяем действия
        actions = manager.check_actions()
        
        for action in actions:
            if action.type == "CLOSE_PARTIAL":
                execution_engine.close_partial(action)
            elif action.type == "MOVE_STOP":
                execution_engine.move_stop(action)
    """
    
    def __init__(
        self,
        config: Optional[PositionConfig] = None,
    ):
        self.config = config or PositionConfig()
        
        # Активные позиции
        self._positions: Dict[str, ManagedPosition] = {}
        
        # Текущая цена (для расчётов)
        self._last_price: float = 0.0
        self._last_price_ts_ns: int = 0
    
    def open_position(
        self,
        signal: TradingSignal,
        sizing: PositionSize,
        next_levels: List[float],
    ) -> Optional[ManagedPosition]:
        """
        Открывает новую позицию.
        
        Вызывается при получении сигнала на вход.
        
        Args:
            signal: Торговый сигнал
            sizing: Рассчитанный размер позиции
            next_levels: Список следующих уровней (для тейков)
            
        Returns:
            Управляемая позиция или None, если открыть нельзя
        """
        if not sizing.is_valid:
            return None
        
        # Проверяем, что у нас достаточно уровней для тейков
        if len(next_levels) < self.config.num_take_profit_levels:
            return None
        
        now_ns = time.time_ns()
        
        # Рассчитываем цену безубытка (с учётом комиссий)
        breakeven_price = self._calculate_breakeven(
            entry_price=sizing.entry_price,
            quantity=sizing.quantity,
            direction=signal.direction,
        )
        
        # Создаём цели (тейк-профиты)
        take_profit_levels = []
        for i in range(self.config.num_take_profit_levels):
            tp_level = TakeProfitLevel(
                price=next_levels[i],
                quantity_pct=self.config.take_profit_split[i],
                level_id=None,
            )
            take_profit_levels.append(tp_level)
        
        # Создаём позицию
        position = ManagedPosition(
            position_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=sizing.entry_price,
            entry_ts_ns=now_ns,
            initial_quantity=sizing.quantity,
            current_quantity=sizing.quantity,
            stop_loss_price=sizing.stop_loss_price,
            take_profit_levels=take_profit_levels,
            breakeven_price=breakeven_price,
            state=PositionState.ACTIVE,
            highest_price_since_entry=sizing.entry_price,
            lowest_price_since_entry=sizing.entry_price,
        )
        
        self._positions[position.position_id] = position
        
        return position
    
    def on_price_update(self, price: float, ts_ns: int) -> None:
        """
        Обработка обновления цены.
        
        Вызывается на каждом тике цены.
        """
        if price <= 0:
            return
        
        self._last_price = price
        self._last_price_ts_ns = ts_ns
        
        # Обновляем экстремумы для всех позиций
        for position in self._positions.values():
            if position.state in (
                PositionState.ACTIVE,
                PositionState.PARTIALLY_CLOSED,
                PositionState.TRAILING,
            ):
                position.highest_price_since_entry = max(
                    position.highest_price_since_entry, price
                )
                position.lowest_price_since_entry = min(
                    position.lowest_price_since_entry, price
                )
    
    def check_actions(self) -> List[Dict]:
        """
        Проверяет необходимые действия для всех позиций.
        
        Возвращает список действий:
        - CLOSE_PARTIAL: закрыть часть позиции (цель достигнута)
        - MOVE_STOP: переместить стоп-лосс
        - CLOSE_ALL: закрыть всю позицию (стоп или трейлинг)
        """
        actions = []
        
        for position_id, position in list(self._positions.items()):
            position_actions = self._check_position_actions(position)
            actions.extend(position_actions)
        
        return actions
    
    def get_position(self, position_id: str) -> Optional[ManagedPosition]:
        """Возвращает позицию по ID."""
        return self._positions.get(position_id)
    
    def get_active_positions(self) -> List[ManagedPosition]:
        """Возвращает все активные позиции."""
        return [
            p for p in self._positions.values()
            if p.state in (
                PositionState.ACTIVE,
                PositionState.PARTIALLY_CLOSED,
                PositionState.TRAILING,
            )
        ]
    
    def close_position(
        self,
        position_id: str,
        reason: str,
        close_price: Optional[float] = None,
    ) -> None:
        """
        Закрывает позицию полностью.
        
        Вызывается при срабатывании стоп-лосса или трейлинг-стопа.
        """
        position = self._positions.get(position_id)
        if position is None:
            return
        
        price = close_price or self._last_price
        
        # Рассчитываем PnL
        position.unrealized_pnl = self._calculate_pnl(position, price)
        
        # Обновляем состояние
        if reason == "stop_loss":
            position.state = PositionState.STOPPED
        else:
            position.state = PositionState.CLOSED
        
        position.current_quantity = 0
    
    def _check_position_actions(
        self,
        position: ManagedPosition,
    ) -> List[Dict]:
        """Проверяет действия для одной позиции."""
        actions = []
        
        # Проверяем стоп-лосс
        if self._is_stop_loss_hit(position):
            actions.append({
                "type": "CLOSE_ALL",
                "position_id": position.position_id,
                "reason": "stop_loss",
                "price": position.stop_loss_price,
            })
            return actions
        
        # Проверяем безубыток
        if not position.is_breakeven_reached:
            if self._is_breakeven_reached(position):
                position.is_breakeven_reached = True
                
                # Перемещаем стоп в безубыток
                actions.append({
                    "type": "MOVE_STOP",
                    "position_id": position.position_id,
                    "new_stop_price": position.breakeven_price,
                    "reason": "breakeven",
                })
        
        # Проверяем цели (тейк-профиты)
        for i, tp_level in enumerate(position.take_profit_levels):
            if tp_level.is_pending and self._is_target_reached(position, tp_level):
                tp_level.is_reached = True
                tp_level.reached_ts_ns = self._last_price_ts_ns
                
                # Рассчитываем количество для закрытия
                close_quantity = position.initial_quantity * tp_level.quantity_pct
                
                actions.append({
                    "type": "CLOSE_PARTIAL",
                    "position_id": position.position_id,
                    "target_index": i,
                    "target_price": tp_level.price,
                    "close_quantity": close_quantity,
                    "reason": "take_profit",
                })
                
                # Обновляем состояние позиции
                position.current_quantity -= close_quantity
                
                # Если это первая цель — меняем состояние
                if i == 0:
                    position.state = PositionState.PARTIALLY_CLOSED
                
                # Если это последняя цель — активируем трейлинг
                if i == len(position.take_profit_levels) - 1:
                    position.state = PositionState.TRAILING
                    position.is_trailing_active = True
        
        # Проверяем трейлинг-стоп
        if position.is_trailing_active:
            if self._is_trailing_stop_hit(position):
                actions.append({
                    "type": "CLOSE_ALL",
                    "position_id": position.position_id,
                    "reason": "trailing_stop",
                    "price": position.trailing_stop_price,
                })
        
        # Проверяем максимальное время позиции
        position_age_sec = (self._last_price_ts_ns - position.entry_ts_ns) / 1_000_000_000
        if position_age_sec > self.config.max_position_age_sec:
            actions.append({
                "type": "CLOSE_ALL",
                "position_id": position.position_id,
                "reason": "max_age",
                "price": self._last_price,
            })
        
        return actions
    
    def _calculate_breakeven(
        self,
        entry_price: float,
        quantity: float,
        direction: OrderSide,
    ) -> float:
        """
        Рассчитывает цену безубытка с учётом комиссий.
        
        Безубыток = цена входа + комиссии / количество
        
        Для лонга: безубыток выше входа
        Для шорта: безубыток ниже входа
        """
        # Комиссия на вход и выход (тейкер)
        notional = entry_price * quantity
        entry_fee = notional * self.config.taker_fee_pct / 100
        exit_fee = notional * self.config.taker_fee_pct / 100
        total_fees = entry_fee + exit_fee
        
        # Буфер безубытка (если задан)
        buffer = self.config.breakeven_buffer_ticks  # в тиках, но нужен тик_сайз
        
        # Расстояние до безубытка в цене
        if quantity > 0:
            breakeven_distance = total_fees / quantity
        else:
            breakeven_distance = 0.0
        
        if direction == OrderSide.BUY:
            return entry_price + breakeven_distance
        else:
            return entry_price - breakeven_distance
    
    def _is_breakeven_reached(self, position: ManagedPosition) -> bool:
        """Проверяет, достигнута ли цена безубытка."""
        if position.is_long:
            return self._last_price >= position.breakeven_price
        else:
            return self._last_price <= position.breakeven_price
    
    def _is_stop_loss_hit(self, position: ManagedPosition) -> bool:
        """Проверяет, сработал ли стоп-лосс."""
        if position.is_long:
            return self._last_price <= position.stop_loss_price
        else:
            return self._last_price >= position.stop_loss_price
    
    def _is_target_reached(
        self,
        position: ManagedPosition,
        target: TakeProfitLevel,
    ) -> bool:
        """Проверяет, достигнута ли цель."""
        if position.is_long:
            return self._last_price >= target.price
        else:
            return self._last_price <= target.price
    
    def _is_trailing_stop_hit(self, position: ManagedPosition) -> bool:
        """Проверяет, сработал ли трейлинг-стоп."""
        # Обновляем трейлинг-стоп
        if position.is_long:
            # Для лонга трейлинг-стоп двигается вверх за ценой
            new_trailing = (
                position.highest_price_since_entry
                - self.config.trailing_stop_ticks * self._get_tick_size(position.symbol)
            )
            position.trailing_stop_price = max(
                position.trailing_stop_price, new_trailing
            )
            return self._last_price <= position.trailing_stop_price
        else:
            # Для шорта трейлинг-стоп двигается вниз за ценой
            new_trailing = (
                position.lowest_price_since_entry
                + self.config.trailing_stop_ticks * self._get_tick_size(position.symbol)
            )
            if position.trailing_stop_price == 0:
                position.trailing_stop_price = new_trailing
            else:
                position.trailing_stop_price = min(
                    position.trailing_stop_price, new_trailing
                )
            return self._last_price >= position.trailing_stop_price
    
    def _calculate_pnl(
        self,
        position: ManagedPosition,
        close_price: float,
    ) -> float:
        """Рассчитывает PnL позиции."""
        if position.is_long:
            pnl = (close_price - position.entry_price) * position.current_quantity
        else:
            pnl = (position.entry_price - close_price) * position.current_quantity
        
        # Вычитаем комиссии
        notional = close_price * position.current_quantity
        fees = notional * self.config.taker_fee_pct / 100
        
        return pnl - fees
    
    def _get_tick_size(self, symbol: str) -> float:
        """Возвращает тик-сайз для символа."""
        # В реальной реализации берём из реестра инструментов
        # Для сейчас возвращаем значение по умолчанию
        tick_sizes = {
            "BTCUSDT": 0.1,
            "ETHUSDT": 0.01,
        }
        return tick_sizes.get(symbol.upper(), 0.01)
    
    def reset(self) -> None:
        """Сбрасывает состояние менеджера."""
        self._positions.clear()
        self._last_price = 0.0
        self._last_price_ts_ns = 0