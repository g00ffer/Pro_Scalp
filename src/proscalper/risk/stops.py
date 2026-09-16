"""
Логика стоп-лоссов и тейк-профитов для стратегии пробоя.

Ключевой принцип: уровень определяется массой ордеров.
Если плотность проедена — уровень де-факто перестал существовать.
Стоп ставится не за уровнем, а за ЗОНОЙ ПРОБОЯ (проеденной плотностью).

Логика стопов:
1. Начальный стоп: за проеденной плотностью + буфер 5 тиков
2. Безубыток: когда цена отбивает комиссии биржи
3. Трейлинг: после достижения первой цели, тянем за ценой
4. Тейки: на следующих уровнях в направлении пробоя

Если плотности не было — уровень определён неверно,
такой пробой не торгуем (сигнал отклоняется раньше).

Используется в связке с:
- PositionManager (управление позициями)
- ExecutionEngine (исполнение ордеров)
- BreakoutDetector (информация о проеденной плотности)
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


class StopType(Enum):
    """
    Тип стопа.
    
    Жизненный цикл стопов для позиции:
        INITIAL → BREAKEVEN → TRAILING
    """
    INITIAL = auto()        # Начальный стоп (за проеденной плотностью)
    BREAKEVEN = auto()      # Стоп в безубытке (покрытие комиссий)
    TRAILING = auto()       # Трейлинг-стоп (тянется за ценой)


@dataclass
class StopConfig:
    """
    Конфигурация стопов.
    
    Все параметры подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Начальный стоп
    stop_buffer_ticks: int = 5              # буфер за проеденной плотностью
    
    # Безубыток
    taker_fee_pct: float = 0.04             # комиссия тейкера (%)
    maker_fee_pct: float = 0.02             # комиссия мейкера (%)
    breakeven_buffer_ticks: int = 0         # буфер безубытка (0 = только комиссии)
    
    # Трейлинг
    trailing_stop_ticks: int = 10           # расстояние трейлинг-стопа
    trailing_activation_ticks: int = 5      # активация трейлинга после цели
    
    # Тейк-профиты
    num_take_profit_levels: int = 2         # количество целей
    take_profit_split: List[float] = field(
        default_factory=lambda: [0.5, 0.3, 0.2]
    )  # деление позиции


@dataclass
class StopLevel:
    """
    Конкретный стоп или тейк-профит.
    
    Хранит информацию о цене, типе и состоянии.
    """
    stop_id: str
    stop_type: StopType
    price: float
    
    # Состояние
    is_active: bool = True
    created_ts_ns: int = 0
    triggered_ts_ns: int = 0
    
    # Для тейк-профитов
    is_take_profit: bool = False
    quantity_pct: float = 0.0               # % позиции для закрытия
    target_index: int = 0                   # индекс цели
    
    @property
    def is_triggered(self) -> bool:
        """Стоп сработал."""
        return self.triggered_ts_ns > 0
    
    @property
    def is_pending(self) -> bool:
        """Стоп активен и ещё не сработал."""
        return self.is_active and not self.is_triggered


@dataclass
class PositionStops:
    """
    Все стопы и тейки для одной позиции.
    
    Содержит полный набор ордеров для управления позицией.
    """
    position_id: str
    symbol: str
    direction: OrderSide
    
    # Стоп-лосс
    stop_loss: StopLevel
    
    # Тейк-профиты
    take_profits: List[StopLevel] = field(default_factory=list)
    
    # Текущий тип активного стопа
    active_stop_type: StopType = StopType.INITIAL
    
    # Цена проеденной плотности (для расчётов)
    wall_consumed_price: float = 0.0
    
    # Безубыток
    breakeven_price: float = 0.0
    is_breakeven_active: bool = False
    
    # Трейлинг
    trailing_stop_price: float = 0.0
    is_trailing_active: bool = False
    
    # Экстремумы для трейлинга
    highest_price: float = 0.0
    lowest_price: float = float('inf')


class StopsCalculator:
    """
    Калькулятор стопов и тейк-профитов.
    
    Реализует чистую математику расчёта:
    - Начального стопа (за проеденной плотностью)
    - Безубытка (покрытие комиссий)
    - Трейлинг-стопа
    - Тейк-профитов (на уровнях)
    
    Использование:
        calculator = StopsCalculator(config)
        
        # Начальный стоп
        stop_price = calculator.calculate_initial_stop(
            entry_price=76050,
            wall_consumed_price=76000,
            direction=OrderSide.BUY,
            tick_size=0.1,
        )
        
        # Безубыток
        breakeven = calculator.calculate_breakeven(
            entry_price=76050,
            quantity=0.1,
            direction=OrderSide.BUY,
        )
    """
    
    def __init__(self, config: Optional[StopConfig] = None):
        self.config = config or StopConfig()
    
    def calculate_initial_stop(
        self,
        entry_price: float,
        wall_consumed_price: float,
        direction: OrderSide,
        tick_size: float,
    ) -> float:
        """
        Рассчитывает начальный стоп-лосс.
        
        Стоп ставится за проеденной плотностью с буфером.
        
        Для лонга (пробой сопротивления вверх):
            стоп = цена_проеденной_плотности - буфер
        
        Для шорта (пробой поддержки вниз):
            стоп = цена_проеденной_плотности + буфер
        
        Если цена входа и проеденной плотности совпадают
        (что не должно происходить при правильном пробое),
        используем фиксированный стоп от входа.
        """
        buffer = self.config.stop_buffer_ticks * tick_size
        
        if direction == OrderSide.BUY:
            # Лонг: стоп ниже проеденной плотности
            stop_price = wall_consumed_price - buffer
            
            # Проверяем, что стоп ниже входа
            if stop_price >= entry_price:
                # Если стоп получился выше входа, используем фиксированный
                stop_price = entry_price - buffer
            
            return stop_price
        
        else:
            # Шорт: стоп выше проеденной плотности
            stop_price = wall_consumed_price + buffer
            
            # Проверяем, что стоп выше входа
            if stop_price <= entry_price:
                # Если стоп получился ниже входа, используем фиксированный
                stop_price = entry_price + buffer
            
            return stop_price
    
    def calculate_breakeven(
        self,
        entry_price: float,
        quantity: float,
        direction: OrderSide,
        fee_pct: Optional[float] = None,
    ) -> float:
        """
        Рассчитывает цену безубытка.
        
        Безубыток = цена входа + комиссии / количество
        
        Комиссии включают:
        - Комиссия на вход (тейкер или мейкер)
        - Комиссия на выход (тейкер или мейкер)
        """
        if fee_pct is None:
            fee_pct = self.config.taker_fee_pct
        
        # Комиссия на вход и выход
        notional = entry_price * quantity
        entry_fee = notional * fee_pct / 100
        exit_fee = notional * fee_pct / 100
        total_fees = entry_fee + exit_fee
        
        # Буфер безубытка
        buffer = self.config.breakeven_buffer_ticks  # в тиках, но нужен тик_сайз
        
        # Расстояние до безубытка
        if quantity > 0:
            breakeven_distance = total_fees / quantity
        else:
            breakeven_distance = 0.0
        
        if direction == OrderSide.BUY:
            return entry_price + breakeven_distance
        else:
            return entry_price - breakeven_distance
    
    def calculate_trailing_stop(
        self,
        current_extreme: float,
        direction: OrderSide,
        tick_size: float,
    ) -> float:
        """
        Рассчитывает трейлинг-стоп.
        
        Для лонга: трейлинг-стоп = максимум - расстояние
        Для шорта: трейлинг-стоп = минимум + расстояние
        """
        distance = self.config.trailing_stop_ticks * tick_size
        
        if direction == OrderSide.BUY:
            return current_extreme - distance
        else:
            return current_extreme + distance
    
    def calculate_take_profits(
        self,
        entry_price: float,
        next_levels: List[float],
        direction: OrderSide,
    ) -> List[StopLevel]:
        """
        Рассчитывает тейк-профиты на следующих уровнях.
        
        Тейки ставятся на уровнях в направлении пробоя.
        Каждый тейк закрывает часть позиции согласно конфигурации.
        """
        take_profits = []
        
        num_targets = min(
            self.config.num_take_profit_levels,
            len(next_levels),
        )
        
        for i in range(num_targets):
            target_price = next_levels[i]
            quantity_pct = self.config.take_profit_split[i]
            
            # Проверяем, что цель в правильном направлении
            if direction == OrderSide.BUY:
                # Для лонга цель должна быть выше входа
                if target_price <= entry_price:
                    continue
            else:
                # Для шорта цель должна быть ниже входа
                if target_price >= entry_price:
                    continue
            
            tp = StopLevel(
                stop_id=str(uuid.uuid4()),
                stop_type=StopType.INITIAL,
                price=target_price,
                is_take_profit=True,
                quantity_pct=quantity_pct,
                target_index=i,
                created_ts_ns=time.time_ns(),
            )
            take_profits.append(tp)
        
        return take_profits
    
    def should_use_breakeven(
        self,
        current_price: float,
        breakeven_price: float,
        direction: OrderSide,
    ) -> bool:
        """
        Проверяет, нужно ли переместить стоп в безубыток.
        
        Возвращает True, если цена достигла безубытка.
        """
        if direction == OrderSide.BUY:
            return current_price >= breakeven_price
        else:
            return current_price <= breakeven_price
    
    def should_activate_trailing(
        self,
        current_price: float,
        first_target_price: float,
        direction: OrderSide,
        tick_size: float,
    ) -> bool:
        """
        Проверяет, нужно ли активировать трейлинг-стоп.
        
        Трейлинг активируется после достижения первой цели.
        """
        activation_buffer = self.config.trailing_activation_ticks * tick_size
        
        if direction == OrderSide.BUY:
            return current_price >= first_target_price + activation_buffer
        else:
            return current_price <= first_target_price - activation_buffer


class StopsManager:
    """
    Менеджер стопов для позиций.
    
    Управляет жизненным циклом стопов:
    1. Создание стопов при открытии позиции
    2. Проверка срабатывания
    3. Перемещение в безубыток
    4. Активация трейлинга
    
    Использование:
        manager = StopsManager(config)
        
        # Создаём стопы для позиции
        stops = manager.create_stops_for_position(
            signal=signal,
            sizing=sizing,
            wall_consumed_price=76000,
            next_levels=[76200, 76500],
            tick_size=0.1,
        )
        
        # На каждом обновлении цены
        triggered = manager.check_stops_triggered(stops, price)
        
        for stop in triggered:
            if stop.is_take_profit:
                # Закрываем часть позиции
                pass
            else:
                # Закрываем всю позицию
                pass
    """
    
    def __init__(self, config: Optional[StopConfig] = None):
        self.config = config or StopConfig()
        self.calculator = StopsCalculator(self.config)
        
        # Активные позиции со стопами
        self._position_stops: Dict[str, PositionStops] = {}
    
    def create_stops_for_position(
        self,
        position_id: str,
        symbol: str,
        direction: OrderSide,
        entry_price: float,
        quantity: float,
        wall_consumed_price: float,
        next_levels: List[float],
        tick_size: float,
    ) -> Optional[PositionStops]:
        """
        Создаёт стопы для новой позиции.
        
        Вызывается при открытии позиции.
        
        Возвращает PositionStops или None, если создать нельзя
        (например, нет проеденной плотности).
        """
        # Проверяем, что есть проеденная плотность
        if wall_consumed_price <= 0:
            # Плотности не было — уровень определён неверно
            return None
        
        now_ns = time.time_ns()
        
        # Рассчитываем начальный стоп
        stop_price = self.calculator.calculate_initial_stop(
            entry_price=entry_price,
            wall_consumed_price=wall_consumed_price,
            direction=direction,
            tick_size=tick_size,
        )
        
        # Рассчитываем безубыток
        breakeven_price = self.calculator.calculate_breakeven(
            entry_price=entry_price,
            quantity=quantity,
            direction=direction,
        )
        
        # Рассчитываем тейк-профиты
        take_profits = self.calculator.calculate_take_profits(
            entry_price=entry_price,
            next_levels=next_levels,
            direction=direction,
        )
        
        # Создаём начальный стоп
        stop_loss = StopLevel(
            stop_id=str(uuid.uuid4()),
            stop_type=StopType.INITIAL,
            price=stop_price,
            created_ts_ns=now_ns,
        )
        
        # Создаём PositionStops
        position_stops = PositionStops(
            position_id=position_id,
            symbol=symbol,
            direction=direction,
            stop_loss=stop_loss,
            take_profits=take_profits,
            wall_consumed_price=wall_consumed_price,
            breakeven_price=breakeven_price,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        
        self._position_stops[position_id] = position_stops
        
        return position_stops
    
    def on_price_update(
        self,
        position_id: str,
        price: float,
        tick_size: float,
    ) -> List[StopLevel]:
        """
        Обработка обновления цены для позиции.
        
        Проверяет срабатывание стопов и тейков.
        Возвращает список сработавших стопов.
        """
        position_stops = self._position_stops.get(position_id)
        if position_stops is None:
            return []
        
        triggered = []
        
        # Обновляем экстремумы
        position_stops.highest_price = max(
            position_stops.highest_price, price
        )
        position_stops.lowest_price = min(
            position_stops.lowest_price, price
        )
        
        # Проверяем стоп-лосс
        if self._is_stop_triggered(position_stops, price):
            position_stops.stop_loss.triggered_ts_ns = time.time_ns()
            position_stops.stop_loss.is_active = False
            triggered.append(position_stops.stop_loss)
            return triggered  # Стоп сработал, дальше не проверяем
        
        # Проверяем безубыток
        if not position_stops.is_breakeven_active:
            if self.calculator.should_use_breakeven(
                price, position_stops.breakeven_price, position_stops.direction
            ):
                self._move_stop_to_breakeven(position_stops)
        
        # Проверяем тейк-профиты
        for tp in position_stops.take_profits:
            if tp.is_pending and self._is_target_reached(position_stops, tp, price):
                tp.triggered_ts_ns = time.time_ns()
                tp.is_active = False
                triggered.append(tp)
                
                # После первой цели активируем трейлинг
                if tp.target_index == 0:
                    self._activate_trailing(position_stops, tick_size)
        
        # Проверяем трейлинг-стоп
        if position_stops.is_trailing_active:
            if self._is_trailing_triggered(position_stops, price):
                # Трейлинг сработал — создаём фиктивный стоп для возврата
                trailing_stop = StopLevel(
                    stop_id=str(uuid.uuid4()),
                    stop_type=StopType.TRAILING,
                    price=position_stops.trailing_stop_price,
                    triggered_ts_ns=time.time_ns(),
                )
                triggered.append(trailing_stop)
        
        return triggered
    
    def get_position_stops(self, position_id: str) -> Optional[PositionStops]:
        """Возвращает стопы для позиции."""
        return self._position_stops.get(position_id)
    
    def remove_position(self, position_id: str) -> None:
        """Удаляет стопы для закрытой позиции."""
        if position_id in self._position_stops:
            del self._position_stops[position_id]
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику менеджера стопов."""
        return {
            "active_positions": len(self._position_stops),
        }
    
    def _is_stop_triggered(
        self,
        position_stops: PositionStops,
        price: float,
    ) -> bool:
        """Проверяет, сработал ли стоп-лосс."""
        if not position_stops.stop_loss.is_pending:
            return False
        
        if position_stops.direction == OrderSide.BUY:
            return price <= position_stops.stop_loss.price
        else:
            return price >= position_stops.stop_loss.price
    
    def _is_target_reached(
        self,
        position_stops: PositionStops,
        target: StopLevel,
        price: float,
    ) -> bool:
        """Проверяет, достигнута ли цель."""
        if position_stops.direction == OrderSide.BUY:
            return price >= target.price
        else:
            return price <= target.price
    
    def _is_trailing_triggered(
        self,
        position_stops: PositionStops,
        price: float,
    ) -> bool:
        """Проверяет, сработал ли трейлинг-стоп."""
        if position_stops.direction == OrderSide.BUY:
            return price <= position_stops.trailing_stop_price
        else:
            return price >= position_stops.trailing_stop_price
    
    def _move_stop_to_breakeven(
        self,
        position_stops: PositionStops,
    ) -> None:
        """Перемещает стоп в безубыток."""
        position_stops.stop_loss.price = position_stops.breakeven_price
        position_stops.stop_loss.stop_type = StopType.BREAKEVEN
        position_stops.is_breakeven_active = True
        position_stops.active_stop_type = StopType.BREAKEVEN
    
    def _activate_trailing(
        self,
        position_stops: PositionStops,
        tick_size: float,
    ) -> None:
        """Активирует трейлинг-стоп."""
        position_stops.is_trailing_active = True
        position_stops.active_stop_type = StopType.TRAILING
        
        # Рассчитываем начальный трейлинг-стоп
        if position_stops.direction == OrderSide.BUY:
            position_stops.trailing_stop_price = (
                position_stops.highest_price
                - self.config.trailing_stop_ticks * tick_size
            )
        else:
            position_stops.trailing_stop_price = (
                position_stops.lowest_price
                + self.config.trailing_stop_ticks * tick_size
            )