"""
Расчёт размеров позиций.

Критичный модуль риск-менеджмента. Отвечает за:
- Расчёт размера позиции по риску на сделку (% от депозита)
- Учёт плеча и маржи
- Валидацию минимального/максимального размера
- Округление до шага количества (step_size)

Модели расчёта:
- FIXED_RISK_PCT: фиксированный % риска от депозита (основная)
- FIXED_NOTIONAL: фиксированный номинал (для тестов)
- KELLY: критерий Келли (будет добавлен позже)

Используется в связке с:
- RiskManager (фильтры рисков)
- PositionManager (управление позициями)
- ExecutionEngine (исполнение сделок)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.core.types import OrderSide


class SizingModel(Enum):
    """
    Модель расчёта размера позиции.
    
    В первой версии реализованы:
    - FIXED_RISK_PCT (основная)
    - FIXED_NOTIONAL (для тестов)
    
    Позже будет добавлен:
    - KELLY (критерий Келли для оптимального размера)
    """
    FIXED_RISK_PCT = auto()     # Фиксированный % риска от депозита
    FIXED_NOTIONAL = auto()     # Фиксированный номинал
    KELLY = auto()              # Критерий Келли (не реализован)


@dataclass
class SizingConfig:
    """
    Конфигурация расчёта размеров позиций.
    
    Все пороги подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    
    Пример:
        Депозит: 10 000 USDT
        risk_per_trade_pct: 0.25
        Риск на сделку: 25 USDT
        
        Если стоп-лосс в 10 тиках (100 пунктов для BTC с тиком 0.1):
        Размер позиции: 25 / 100 = 0.25 BTC
    """
    # Модель расчёта
    model: SizingModel = SizingModel.FIXED_RISK_PCT
    
    # Риск на сделку (% от депозита)
    risk_per_trade_pct: float = 0.25          # 0.25% от депозита
    max_risk_per_trade_pct: float = 1.0       # максимум 1% от депозита
    
    # Плечо
    default_leverage: int = 5                 # плечо по умолчанию
    max_leverage: int = 10                    # максимальное плечо
    
    # Ограничения номинала
    min_notional: float = 10.0                # минимальный номинал (требование биржи)
    max_position_notional: float = 10000.0    # максимальный номинал позиции
    
    # Округление
    round_down: bool = True                   # округление вниз (консервативно)
    
    # Фиксированный номинал (для модели FIXED_NOTIONAL)
    fixed_notional: float = 100.0
    
    # Безопасность
    max_position_pct_of_deposit: float = 50.0 # максимум 50% депозита в одной позиции


@dataclass
class PositionSize:
    """
    Результат расчёта размера позиции.
    
    Содержит всю информацию для:
    - RiskManager (валидация рисков)
    - ExecutionEngine (исполнение сделки)
    - PositionManager (управление позицией)
    """
    # Идентификация
    symbol: str
    direction: OrderSide
    
    # Цены
    entry_price: float
    stop_loss_price: float
    
    # Размеры
    quantity: float             # размер в контрактах/токенах
    notional: float             # номинал (quantity * entry_price)
    
    # Риск
    risk_amount: float          # сумма риска в деньгах
    risk_pct: float             # риск в % от депозита
    
    # Плечо и маржа
    leverage: int
    margin_required: float      # требуемая маржа
    
    # Валидация
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    # Время расчёта
    calculated_ts_ns: int = 0
    
    @property
    def risk_reward_distance(self) -> float:
        """Расстояние до стоп-лосса в цене."""
        return abs(self.entry_price - self.stop_loss_price)
    
    @property
    def is_long(self) -> bool:
        """Позиция в лонг."""
        return self.direction == OrderSide.BUY
    
    @property
    def is_short(self) -> bool:
        """Позиция в шорт."""
        return self.direction == OrderSide.SELL
    
    def to_dict(self) -> Dict:
        """Преобразует в словарь для логирования."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "quantity": self.quantity,
            "notional": self.notional,
            "risk_amount": self.risk_amount,
            "risk_pct": self.risk_pct,
            "leverage": self.leverage,
            "margin_required": self.margin_required,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class PositionSizer:
    """
    Калькулятор размера позиции для одного символа.
    
    Основная логика:
    1. Определяем сумму риска: депозит × risk_per_trade_pct
    2. Определяем расстояние до стоп-лосса
    3. Рассчитываем размер: риск / расстояние до стопа
    4. Проверяем ограничения (плечо, маржа, минимум/максимум)
    5. Округляем до шага количества
    
    Использование:
        sizer = PositionSizer(
            symbol="BTCUSDT",
            tick_size=0.1,
            step_size=0.001,
            min_quantity=0.001,
            max_quantity=1000,
        )
        
        size = sizer.calculate_size(
            deposit=10000,
            entry_price=76000,
            stop_loss_price=75900,
            direction=OrderSide.BUY,
        )
        
        if size.is_valid:
            print(f"Размер: {size.quantity} BTC, номинал: {size.notional}")
        else:
            print(f"Ошибки: {size.validation_errors}")
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        step_size: float,
        min_quantity: float = 0.0,
        max_quantity: float = float('inf'),
        config: Optional[SizingConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.step_size = step_size
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.config = config or SizingConfig()
    
    def calculate_size(
        self,
        deposit: float,
        entry_price: float,
        stop_loss_price: float,
        direction: OrderSide,
    ) -> PositionSize:
        """
        Рассчитывает размер позиции.
        
        Основная формула для FIXED_RISK_PCT:
            risk_amount = deposit × risk_per_trade_pct / 100
            risk_distance = |entry_price - stop_loss_price|
            quantity = risk_amount / risk_distance
        
        Args:
            deposit: Текущий депозит в деньгах
            entry_price: Цена входа
            stop_loss_price: Цена стоп-лосса
            direction: Направление позиции
            
        Returns:
            Рассчитанный размер позиции
        """
        now_ns = time.time_ns()
        validation_errors: List[str] = []
        
        # Проверяем входные данные
        if deposit <= 0:
            validation_errors.append("Депозит должен быть положительным")
        
        if entry_price <= 0:
            validation_errors.append("Цена входа должна быть положительной")
        
        if stop_loss_price <= 0:
            validation_errors.append("Цена стоп-лосса должна быть положительной")
        
        # Проверяем направление стоп-лосса
        if direction == OrderSide.BUY and stop_loss_price >= entry_price:
            validation_errors.append(
                "Для лонга стоп-лосс должен быть ниже цены входа"
            )
        elif direction == OrderSide.SELL and stop_loss_price <= entry_price:
            validation_errors.append(
                "Для шорта стоп-лосс должен быть выше цены входа"
            )
        
        # Если есть критические ошибки - возвращаем невалидный размер
        if validation_errors:
            return PositionSize(
                symbol=self.symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                quantity=0.0,
                notional=0.0,
                risk_amount=0.0,
                risk_pct=0.0,
                leverage=0,
                margin_required=0.0,
                is_valid=False,
                validation_errors=validation_errors,
                calculated_ts_ns=now_ns,
            )
        
        # Рассчитываем размер в зависимости от модели
        if self.config.model == SizingModel.FIXED_RISK_PCT:
            quantity, risk_amount = self._calculate_fixed_risk(
                deposit, entry_price, stop_loss_price
            )
        elif self.config.model == SizingModel.FIXED_NOTIONAL:
            quantity, risk_amount = self._calculate_fixed_notional(
                entry_price, stop_loss_price
            )
        else:
            validation_errors.append(f"Модель {self.config.model} не реализована")
            quantity = 0.0
            risk_amount = 0.0
        
        # Округляем до шага количества
        quantity = self._round_to_step(quantity)
        
        # Рассчитываем номинал
        notional = quantity * entry_price
        
        # Рассчитываем риск в % от депозита
        risk_pct = (risk_amount / deposit * 100) if deposit > 0 else 0.0
        
        # Рассчитываем требуемую маржу
        leverage = self.config.default_leverage
        margin_required = notional / leverage
        
        # Валидация ограничений
        self._validate_constraints(
            quantity=quantity,
            notional=notional,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            margin_required=margin_required,
            deposit=deposit,
            validation_errors=validation_errors,
        )
        
        return PositionSize(
            symbol=self.symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            quantity=quantity,
            notional=notional,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            leverage=leverage,
            margin_required=margin_required,
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            calculated_ts_ns=now_ns,
        )
    
    def calculate_size_for_risk(
        self,
        risk_amount: float,
        entry_price: float,
        stop_loss_price: float,
        direction: OrderSide,
    ) -> float:
        """
        Рассчитывает размер позиции для заданного риска.
        
        Используется для перерасчёта позиции при изменении стопа.
        
        Формула:
            quantity = risk_amount / |entry_price - stop_loss_price|
        """
        risk_distance = abs(entry_price - stop_loss_price)
        
        if risk_distance <= 0:
            return 0.0
        
        quantity = risk_amount / risk_distance
        return self._round_to_step(quantity)
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: OrderSide,
        distance_ticks: int,
    ) -> float:
        """
        Рассчитывает цену стоп-лосса на расстоянии в тиках.
        
        Используется для быстрого расчёта стопа при входе.
        """
        distance = distance_ticks * self.tick_size
        
        if direction == OrderSide.BUY:
            return entry_price - distance
        else:
            return entry_price + distance
    
    def _calculate_fixed_risk(
        self,
        deposit: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> tuple:
        """
        Расчёт размера по модели фиксированного % риска.
        
        Возвращает (quantity, risk_amount).
        """
        # Сумма риска в деньгах
        risk_amount = deposit * self.config.risk_per_trade_pct / 100
        
        # Расстояние до стоп-лосса
        risk_distance = abs(entry_price - stop_loss_price)
        
        if risk_distance <= 0:
            return 0.0, 0.0
        
        # Размер позиции
        quantity = risk_amount / risk_distance
        
        return quantity, risk_amount
    
    def _calculate_fixed_notional(
        self,
        entry_price: float,
        stop_loss_price: float,
    ) -> tuple:
        """
        Расчёт размера по модели фиксированного номинала.
        
        Возвращает (quantity, risk_amount).
        """
        # Размер позиции из номинала
        quantity = self.config.fixed_notional / entry_price
        
        # Расстояние до стоп-лосса
        risk_distance = abs(entry_price - stop_loss_price)
        
        # Риск в деньгах
        risk_amount = quantity * risk_distance
        
        return quantity, risk_amount
    
    def _round_to_step(self, quantity: float) -> float:
        """
        Округляет количество до шага (step_size).
        
        Если round_down=True - округляем вниз (консервативно).
        Иначе - округляем к ближайшему.
        """
        if self.step_size <= 0:
            return quantity
        
        steps = quantity / self.step_size
        
        if self.config.round_down:
            # Округление вниз
            rounded_steps = int(steps)
        else:
            # Округление к ближайшему
            rounded_steps = round(steps)
        
        return rounded_steps * self.step_size
    
    def _validate_constraints(
        self,
        quantity: float,
        notional: float,
        risk_amount: float,
        risk_pct: float,
        margin_required: float,
        deposit: float,
        validation_errors: List[str],
    ) -> None:
        """Проверяет ограничения и добавляет ошибки в список."""
        # Проверка минимального количества
        if quantity < self.min_quantity:
            validation_errors.append(
                f"Количество {quantity} меньше минимального {self.min_quantity}"
            )
        
        # Проверка максимального количества
        if quantity > self.max_quantity:
            validation_errors.append(
                f"Количество {quantity} больше максимального {self.max_quantity}"
            )
        
        # Проверка минимального номинала
        if notional < self.config.min_notional:
            validation_errors.append(
                f"Номинал {notional:.2f} меньше минимального {self.config.min_notional}"
            )
        
        # Проверка максимального номинала
        if notional > self.config.max_position_notional:
            validation_errors.append(
                f"Номинал {notional:.2f} больше максимального {self.config.max_position_notional}"
            )
        
        # Проверка максимального риска
        if risk_pct > self.config.max_risk_per_trade_pct:
            validation_errors.append(
                f"Риск {risk_pct:.2f}% больше максимального {self.config.max_risk_per_trade_pct}%"
            )
        
        # Проверка маржи
        if margin_required > deposit:
            validation_errors.append(
                f"Требуемая маржа {margin_required:.2f} больше депозита {deposit:.2f}"
            )
        
        # Проверка максимального размера позиции относительно депозита
        max_position_notional = deposit * self.config.max_position_pct_of_deposit / 100
        if notional > max_position_notional:
            validation_errors.append(
                f"Номинал {notional:.2f} больше {self.config.max_position_pct_of_deposit}% депозита"
            )


class PositionSizerManager:
    """
    Менеджер PositionSizer'ов для множества символов.
    
    Использование:
        manager = PositionSizerManager()
        
        # Создаём сайзер для символа
        manager.create_sizer(
            symbol="BTCUSDT",
            tick_size=0.1,
            step_size=0.001,
            min_quantity=0.001,
        )
        
        # Рассчитываем размер позиции
        size = manager.calculate_size(
            symbol="BTCUSDT",
            deposit=10000,
            entry_price=76000,
            stop_loss_price=75900,
            direction=OrderSide.BUY,
        )
    """
    
    def __init__(self):
        self._sizers: Dict[str, PositionSizer] = {}
    
    def create_sizer(
        self,
        symbol: str,
        tick_size: float,
        step_size: float,
        min_quantity: float = 0.0,
        max_quantity: float = float('inf'),
        config: Optional[SizingConfig] = None,
    ) -> PositionSizer:
        """Создаёт сайзер для символа."""
        symbol = symbol.upper()
        
        self._sizers[symbol] = PositionSizer(
            symbol=symbol,
            tick_size=tick_size,
            step_size=step_size,
            min_quantity=min_quantity,
            max_quantity=max_quantity,
            config=config,
        )
        
        return self._sizers[symbol]
    
    def get_sizer(self, symbol: str) -> Optional[PositionSizer]:
        """Возвращает сайзер для символа."""
        return self._sizers.get(symbol.upper())
    
    def calculate_size(
        self,
        symbol: str,
        deposit: float,
        entry_price: float,
        stop_loss_price: float,
        direction: OrderSide,
    ) -> Optional[PositionSize]:
        """Рассчитывает размер позиции для символа."""
        sizer = self._sizers.get(symbol.upper())
        if sizer is None:
            return None
        return sizer.calculate_size(
            deposit, entry_price, stop_loss_price, direction
        )
    
    def reset_all(self) -> None:
        """Сбрасывает все сайзеры."""
        self._sizers.clear()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._sizers.keys())