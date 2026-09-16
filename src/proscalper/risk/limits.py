"""
Ограничения риск-менеджмента.

Чистая математика без субъективных "человеческих" нюансов:
- Максимальная просадка (защита от банкротства)
- Режим рынка (если нет условий для пробоя — не торгуем)
- Ограничения на позиции
- Ограничения на размер сделки

Все пороги конфигурируемые. Если порог не работает — меняем число, а не логику.

Используется в связке с:
- PositionSizer (расчёт размеров позиций)
- PositionManager (управление позициями)
- RiskManager (общий риск-менеджмент)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class MarketRegime(Enum):
    """
    Режим рынка.
    
    Определяет, торгуем ли мы вообще.
    Если режим не подходит для стратегии пробоя — не торгуем.
    """
    TRADING = auto()        # Условия для пробоя есть, торгуем
    RANGING = auto()        # Рынок в диапазоне без уровней, не торгуем
    TRENDING = auto()       # Трендовый рынок без консолидации, не торгуем
    VOLATILE = auto()       # Слишком волатильно, не торгуем
    LOW_VOLUME = auto()     # Низкий объём, не торгуем
    UNKNOWN = auto()        # Неизвестный режим


@dataclass
class LimitConfig:
    """
    Конфигурация ограничений риск-менеджмента.
    
    Все пороги подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # === Просадка ===
    max_drawdown_pct: float = 10.0            # максимальная просадка в % от депозита
    drawdown_check_interval_sec: float = 60.0 # интервал проверки просадки
    
    # === Режим рынка ===
    min_volume_per_hour: float = 5_000_000.0  # минимальный объём за час ($)
    min_volatility_pct: float = 0.3           # минимальная волатильность за час (%)
    max_volatility_pct: float = 5.0           # максимальная волатильность за час (%)
    
    # === Позиции ===
    max_open_positions: int = 1               # максимум открытых позиций (для скальпинга)
    max_position_notional: float = 10_000.0   # максимальный номинал позиции ($)
    max_position_pct_of_deposit: float = 50.0 # максимум % депозита в одной позиции
    
    # === Сделки ===
    min_position_notional: float = 5.0        # минимальный размер позиции ($)
    max_daily_loss_pct: float = 5.0           # максимальный дневной убыток в %
    
    # === Технические ограничения ===
    max_leverage: int = 10                    # максимальное плечо
    min_margin_buffer_pct: float = 20.0       # минимальный буфер маржи в %


@dataclass
class LimitCheckResult:
    """
    Результат проверки ограничений.
    
    Содержит информацию о том, прошла ли проверка,
    и какие лимиты были нарушены.
    """
    is_allowed: bool
    violated_limits: List[str] = field(default_factory=list)
    current_regime: MarketRegime = MarketRegime.UNKNOWN
    
    # Детали для логирования
    details: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_blocked(self) -> bool:
        """Торговля заблокирована."""
        return not self.is_allowed
    
    def add_violation(self, limit_name: str, detail: float = 0.0) -> None:
        """Добавляет нарушенный лимит."""
        self.is_allowed = False
        if limit_name not in self.violated_limits:
            self.violated_limits.append(limit_name)
        if detail != 0.0:
            self.details[limit_name] = detail


class DrawdownTracker:
    """
    Трекер просадки капитала.
    
    Отслеживает максимальную просадку от пика капитала.
    Если просадка достигает лимита — торговля останавливается.
    
    Использование:
        tracker = DrawdownTracker(initial_deposit=10000, max_drawdown_pct=10.0)
        
        tracker.update_equity(9800)  # просадка 2%
        tracker.update_equity(9500)  # просадка 5%
        tracker.update_equity(9000)  # просадка 10% → СТОП
        
        if tracker.is_limit_reached():
            print("Максимальная просадка достигнута, останавливаем торговлю")
    """
    
    def __init__(
        self,
        initial_deposit: float,
        max_drawdown_pct: float = 10.0,
    ):
        self.initial_deposit = initial_deposit
        self.max_drawdown_pct = max_drawdown_pct
        
        # Пик капитала (максимальное значение за всё время)
        self.peak_equity = initial_deposit
        
        # Текущий капитал
        self.current_equity = initial_deposit
        
        # Текущая просадка в %
        self.current_drawdown_pct = 0.0
        
        # Время последнего обновления
        self._last_update_ts_ns = 0
    
    def update_equity(self, equity: float) -> None:
        """
        Обновляет текущий капитал.
        
        Вызывается после каждой сделки или периодически.
        """
        if equity <= 0:
            return
        
        self.current_equity = equity
        self._last_update_ts_ns = time.time_ns()
        
        # Обновляем пик
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Рассчитываем просадку
        if self.peak_equity > 0:
            self.current_drawdown_pct = (
                (self.peak_equity - equity) / self.peak_equity * 100
            )
    
    def is_limit_reached(self) -> bool:
        """Проверяет, достигнута ли максимальная просадка."""
        return self.current_drawdown_pct >= self.max_drawdown_pct
    
    def get_remaining_drawdown_pct(self) -> float:
        """Возвращает оставшийся запас просадки в %."""
        return max(0.0, self.max_drawdown_pct - self.current_drawdown_pct)
    
    def reset_peak(self) -> None:
        """
        Сбрасывает пик на текущий капитал.
        
        Используется после периода остановки торговли,
        чтобы начать отсчёт просадки заново.
        """
        self.peak_equity = self.current_equity
        self.current_drawdown_pct = 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Возвращает статистику просадки."""
        return {
            "initial_deposit": self.initial_deposit,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "remaining_drawdown_pct": self.get_remaining_drawdown_pct(),
            "is_limit_reached": self.is_limit_reached(),
        }


class PositionLimitsChecker:
    """
    Проверка ограничений на позиции.
    
    Проверяет:
    - Максимум открытых позиций
    - Максимальный номинал позиции
    - Максимальный % депозита в позиции
    - Минимальный размер позиции
    
    Использование:
        checker = PositionLimitsChecker(config)
        
        result = checker.check_position(
            current_open_positions=0,
            position_notional=1000,
            deposit=10000,
        )
        
        if result.is_allowed:
            print("Позиция разрешена")
        else:
            print(f"Нарушены лимиты: {result.violated_limits}")
    """
    
    def __init__(self, config: LimitConfig):
        self.config = config
    
    def check_position(
        self,
        current_open_positions: int,
        position_notional: float,
        deposit: float,
    ) -> LimitCheckResult:
        """
        Проверяет, разрешена ли новая позиция.
        
        Args:
            current_open_positions: Текущее количество открытых позиций
            position_notional: Номинал новой позиции
            deposit: Текущий депозит
            
        Returns:
            Результат проверки
        """
        result = LimitCheckResult(is_allowed=True)
        
        # Проверка 1: Максимум открытых позиций
        if current_open_positions >= self.config.max_open_positions:
            result.add_violation(
                "max_open_positions",
                float(current_open_positions),
            )
        
        # Проверка 2: Минимальный размер позиции
        if position_notional < self.config.min_position_notional:
            result.add_violation(
                "min_position_notional",
                position_notional,
            )
        
        # Проверка 3: Максимальный номинал позиции
        if position_notional > self.config.max_position_notional:
            result.add_violation(
                "max_position_notional",
                position_notional,
            )
        
        # Проверка 4: Максимальный % депозита в позиции
        if deposit > 0:
            position_pct = position_notional / deposit * 100
            if position_pct > self.config.max_position_pct_of_deposit:
                result.add_violation(
                    "max_position_pct_of_deposit",
                    position_pct,
                )
        
        return result


class DailyLossTracker:
    """
    Трекер дневного убытка.
    
    Отслеживает убыток за текущий день.
    Если убыток достигает лимита — торговля останавливается до следующего дня.
    
    Использование:
        tracker = DailyLossTracker(max_daily_loss_pct=5.0)
        
        tracker.start_new_day(deposit=10000)
        
        tracker.record_loss(100)   # убыток 100
        tracker.record_profit(50)  # прибыль 50
        
        if tracker.is_limit_reached():
            print("Дневной лимит убытков достигнут")
    """
    
    def __init__(self, max_daily_loss_pct: float = 5.0):
        self.max_daily_loss_pct = max_daily_loss_pct
        
        self.day_start_deposit: float = 0.0
        self.current_day_pnl: float = 0.0
        self.current_day_loss_pct: float = 0.0
        
        self._current_day: int = 0
    
    def start_new_day(self, deposit: float) -> None:
        """
        Начинает новый день.
        
        Вызывается при старте торговли или в полночь.
        """
        self.day_start_deposit = deposit
        self.current_day_pnl = 0.0
        self.current_day_loss_pct = 0.0
        self._current_day = time.time() // 86400
    
    def record_trade_pnl(self, pnl: float) -> None:
        """
        Записывает результат сделки.
        
        Вызывается после каждой закрытой сделки.
        """
        self.current_day_pnl += pnl
        
        if self.day_start_deposit > 0 and self.current_day_pnl < 0:
            self.current_day_loss_pct = (
                abs(self.current_day_pnl) / self.day_start_deposit * 100
            )
    
    def is_limit_reached(self) -> bool:
        """Проверяет, достигнут ли дневной лимит убытков."""
        return self.current_day_loss_pct >= self.max_daily_loss_pct
    
    def get_remaining_loss_pct(self) -> float:
        """Возвращает оставшийся запас убытка в %."""
        return max(0.0, self.max_daily_loss_pct - self.current_day_loss_pct)
    
    def check_day_rollover(self, current_deposit: float) -> None:
        """
        Проверяет, наступил ли новый день.
        
        Если да — начинает новый день.
        """
        new_day = time.time() // 86400
        if new_day != self._current_day:
            self.start_new_day(current_deposit)
    
    def get_stats(self) -> Dict[str, float]:
        """Возвращает статистику дневного убытка."""
        return {
            "day_start_deposit": self.day_start_deposit,
            "current_day_pnl": self.current_day_pnl,
            "current_day_loss_pct": self.current_day_loss_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "is_limit_reached": self.is_limit_reached(),
        }


class RiskLimitsManager:
    """
    Общий менеджер ограничений риск-менеджмента.
    
    Объединяет все лимиты:
    - Просадка (DrawdownTracker)
    - Дневной убыток (DailyLossTracker)
    - Позиции (PositionLimitsChecker)
    
    Использование:
        manager = RiskLimitsManager(
            initial_deposit=10000,
            config=LimitConfig(),
        )
        
        # Проверяем, можно ли торговать
        check = manager.check_trading_allowed()
        
        if check.is_allowed:
            # Проверяем конкретную позицию
            position_check = manager.check_position(
                open_positions=0,
                position_notional=1000,
            )
            
            if position_check.is_allowed:
                # Открываем позицию
                pass
        
        # После сделки обновляем капитал
        manager.update_equity(9950)
    """
    
    def __init__(
        self,
        initial_deposit: float,
        config: Optional[LimitConfig] = None,
    ):
        self.config = config or LimitConfig()
        
        # Трекер просадки
        self.drawdown_tracker = DrawdownTracker(
            initial_deposit=initial_deposit,
            max_drawdown_pct=self.config.max_drawdown_pct,
        )
        
        # Трекер дневного убытка
        self.daily_loss_tracker = DailyLossTracker(
            max_daily_loss_pct=self.config.max_daily_loss_pct,
        )
        
        # Проверка позиций
        self.position_checker = PositionLimitsChecker(self.config)
        
        # Текущий депозит
        self.current_deposit = initial_deposit
        
        # Количество открытых позиций
        self.open_positions_count = 0
    
    def check_trading_allowed(self) -> LimitCheckResult:
        """
        Проверяет, разрешена ли торговля вообще.
        
        Возвращает LimitCheckResult с информацией о нарушенных лимитах.
        """
        result = LimitCheckResult(is_allowed=True)
        
        # Проверка 1: Просадка
        if self.drawdown_tracker.is_limit_reached():
            result.add_violation(
                "max_drawdown",
                self.drawdown_tracker.current_drawdown_pct,
            )
        
        # Проверка 2: Дневной убыток
        if self.daily_loss_tracker.is_limit_reached():
            result.add_violation(
                "max_daily_loss",
                self.daily_loss_tracker.current_day_loss_pct,
            )
        
        # Проверка 3: Открытые позиции
        if self.open_positions_count >= self.config.max_open_positions:
            result.add_violation(
                "max_open_positions",
                float(self.open_positions_count),
            )
        
        return result
    
    def check_position(
        self,
        position_notional: float,
    ) -> LimitCheckResult:
        """
        Проверяет, разрешена ли конкретная позиция.
        
        Вызывается перед открытием новой позиции.
        """
        # Сначала проверяем общие лимиты
        general_check = self.check_trading_allowed()
        if general_check.is_blocked:
            return general_check
        
        # Проверяем лимиты позиции
        position_check = self.position_checker.check_position(
            current_open_positions=self.open_positions_count,
            position_notional=position_notional,
            deposit=self.current_deposit,
        )
        
        return position_check
    
    def update_equity(self, equity: float) -> None:
        """
        Обновляет текущий капитал.
        
        Вызывается после каждой сделки или периодически.
        """
        self.current_deposit = equity
        self.drawdown_tracker.update_equity(equity)
        
        # Проверяем смену дня
        self.daily_loss_tracker.check_day_rollover(equity)
    
    def record_trade_pnl(self, pnl: float) -> None:
        """
        Записывает результат закрытой сделки.
        
        Вызывается после закрытия позиции.
        """
        self.daily_loss_tracker.record_trade_pnl(pnl)
        
        # Обновляем капитал
        self.current_deposit += pnl
        self.drawdown_tracker.update_equity(self.current_deposit)
    
    def open_position(self) -> None:
        """Помечает открытие позиции."""
        self.open_positions_count += 1
    
    def close_position(self) -> None:
        """Помечает закрытие позиции."""
        if self.open_positions_count > 0:
            self.open_positions_count -= 1
    
    def get_stats(self) -> Dict[str, Dict]:
        """Возвращает полную статистику риск-менеджмента."""
        return {
            "drawdown": self.drawdown_tracker.get_stats(),
            "daily_loss": self.daily_loss_tracker.get_stats(),
            "positions": {
                "open_positions_count": self.open_positions_count,
                "max_open_positions": self.config.max_open_positions,
            },
            "deposit": {
                "current": self.current_deposit,
            },
        }
    
    def reset_daily(self) -> None:
        """
        Сбрасывает дневные счётчики.
        
        Вызывается в полночь или при старте нового торгового дня.
        """
        self.daily_loss_tracker.start_new_day(self.current_deposit)
    
    def reset_after_stop(self) -> None:
        """
        Сбрасывает состояние после остановки торговли.
        
        Вызывается при возобновлении торговли после просадки.
        """
        self.drawdown_tracker.reset_peak()
        self.daily_loss_tracker.start_new_day(self.current_deposit)