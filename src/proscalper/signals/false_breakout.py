"""
Детектор ложного пробоя (закола).

Ложный пробой — это ситуация, когда цена вышла за уровень,
но не смогла удержаться и вернулась обратно. Это сигнал:
- НЕ входить в пробой
- Отменить отложенные заявки на пробой
- Закрыть позицию, если вход уже был выполнен по раннему подтверждению
- Перевести уровень в состояние FAILED_BREAK с cooldown

Признаки ложного пробоя (из формализации системы):
1. Цена вышла за уровень, но быстро вернулась внутрь
2. Удержание за уровнем не подтверждено
3. Агрессивный объём слабый
4. Цена не прошла минимальное продолжение
5. Крупная плотность в стакане не съедена
6. Поглощение: объём есть, а цена не идёт (absorption)

Архитектура:
- Работает параллельно с BreakoutEngine
- Если BreakoutEngine подтверждает пробой → ложный пробой отменяется
- Если ложный пробой подтверждается → пробой отменяется

Используется в связке с:
- BreakoutEngine (параллельная работа)
- LevelTracker (перевод уровня в FAILED_BREAK)
- SignalGenerator (отмена сигналов пробоя)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.core.types import OrderSide, SignalRejectReason
from proscalper.signals.base import (
    BaseSignalEngine,
    BaseEngineConfig,
    EngineResult,
    MarketContext,
    SetupType,
)
from proscalper.features.levels import Level, LevelState


# ============================================================
# Конфигурация
# ============================================================

@dataclass
class FalseBreakoutConfig(BaseEngineConfig):
    """
    Конфигурация детектора ложного пробоя.
    
    Все параметры подобраны как разумные значения по умолчанию
    согласно формализации системы. В дальнейшем будут
    подкручены на живых данных.
    """
    # Окно детекции ложного пробоя
    window_ms: int = 1500               # окно для возврата цены
    
    # Минимальное продолжение пробоя
    min_extension_ticks: int = 3        # мин. расширение за уровень в тиках
    
    # Поглощение (absorption)
    max_absorption_score: float = 2.0   # макс. отношение объём/смещение
    
    # Кулдаун после ложного пробоя
    cooldown_after_false_sec: float = 60.0
    
    # Буфер возврата внутрь
    return_inside_buffer_ticks: int = 1  # буфер для возврата за уровень
    
    # Требования к объёму
    min_volume_burst_ratio: float = 1.5  # мин. всплеск объёма для истинного пробоя
    
    # Требования к удержанию
    min_hold_ms: int = 200               # мин. время удержания за уровнем


# ============================================================
# Состояния детекции
# ============================================================

class FalseBreakoutState(Enum):
    """
    Состояние детекции ложного пробоя.
    
    Жизненный цикл:
        IDLE → MONITORING → CONFIRMED / CANCELLED
    """
    IDLE = auto()           # Нет активного мониторинга
    MONITORING = auto()     # Мониторинг потенциального ложного пробоя
    CONFIRMED = auto()      # Ложный пробой подтверждён
    CANCELLED = auto()      # Пробой оказался истинным, мониторинг отменён


# ============================================================
# Попытка детекции
# ============================================================

@dataclass
class FalseBreakoutAttempt:
    """
    Попытка детекции ложного пробоя для конкретного уровня.
    
    Хранит всю информацию о текущей попытке мониторинга.
    """
    # Идентификация
    attempt_id: str
    level_id: str
    symbol: str
    
    # Направление пробоя, который проверяем
    # Для лонга: проверяем ложный пробой сопротивления вверх
    # Для шорта: проверяем ложный пробой поддержки вниз
    breakout_side: OrderSide
    
    # Цены
    level_price: float
    cross_price: float              # цена пересечения уровня
    max_extension_price: float      # максимальное расширение за уровень
    
    # Время
    cross_ts_ns: int                # время пересечения уровня
    state_entered_ts_ns: int = 0
    
    # Состояние
    state: FalseBreakoutState = FalseBreakoutState.IDLE
    
    # Метрики для детекции
    max_extension_ticks: float = 0.0    # макс. расширение за уровень в тиках
    hold_duration_ms: float = 0.0       # время удержания за уровнем
    absorption_score: float = 0.0       # скор поглощения
    
    # Причины (для журнала)
    detection_reasons: List[str] = field(default_factory=list)
    
    @property
    def is_long_breakout(self) -> bool:
        """Проверяем ложный пробой в лонг (сопротивление вверх)."""
        return self.breakout_side == OrderSide.BUY
    
    @property
    def is_short_breakout(self) -> bool:
        """Проверяем ложный пробой в шорт (поддержка вниз)."""
        return self.breakout_side == OrderSide.SELL
    
    @property
    def elapsed_ms(self) -> float:
        """Время с момента пересечения уровня."""
        now_ns = time.time_ns()
        return (now_ns - self.cross_ts_ns) / 1_000_000
    
    @property
    def is_window_expired(self) -> bool:
        """Окно детекции истекло."""
        return False  # определяется в движке


# ============================================================
# Движок детекции ложного пробоя
# ============================================================

class FalseBreakoutEngine(BaseSignalEngine):
    """
    Движок детекции ложного пробоя (закола).
    
    Работает параллельно с BreakoutEngine:
    - Когда BreakoutEngine обнаруживает пересечение уровня,
      вызывается start_monitoring()
    - Движок отслеживает, вернётся ли цена за уровень
    - Если цена вернулась → ложный пробой подтверждён
    - Если пробой оказался истинным → вызывается cancel_monitoring()
    
    Использование:
        engine = FalseBreakoutEngine(config, tick_size)
        
        # При пересечении уровня (из BreakoutEngine)
        engine.start_monitoring(level, cross_price, side, ts_ns)
        
        # На каждом обновлении рынка
        result = engine.evaluate(context)
        
        if result.has_signal:
            # Ложный пробой подтверждён
            level_tracker.mark_failed_breakout(level_id)
        
        # При подтверждении истинного пробоя
        engine.cancel_monitoring(level_id)
    """
    
    def __init__(
        self,
        config: Optional[FalseBreakoutConfig] = None,
        tick_size: float = 0.01,
    ):
        super().__init__(
            config=config or FalseBreakoutConfig(),
            tick_size=tick_size,
        )
        self._config = config or FalseBreakoutConfig()
        
        # Активные попытки мониторинга
        self._attempts: Dict[str, FalseBreakoutAttempt] = {}
        
        # Счётчик попыток
        self._attempt_counter: int = 0
        
        # Подтверждённые ложные пробои (ожидают обработки)
        self._confirmed: List[FalseBreakoutAttempt] = []
    
    @property
    def setup_type(self) -> SetupType:
        return SetupType.FALSE_BREAKOUT
    
    @property
    def name(self) -> str:
        return "FalseBreakoutEngine"
    
    # ============================================
    # Публичный интерфейс
    # ============================================
    
    def start_monitoring(
        self,
        level: Level,
        cross_price: float,
        breakout_side: OrderSide,
        ts_ns: int,
    ) -> Optional[FalseBreakoutAttempt]:
        """
        Начинает мониторинг потенциального ложного пробоя.
        
        Вызывается из BreakoutEngine при обнаружении пересечения уровня.
        
        Args:
            level: уровень, который пересекается
            cross_price: цена пересечения уровня
            breakout_side: направление пробоя (BUY для лонга, SELL для шорта)
            ts_ns: время пересечения
            
        Returns:
            Созданная попытка мониторинга или None
        """
        # Проверяем, нет ли уже мониторинга для этого уровня
        if level.id in self._attempts:
            return None
        
        # Проверяем лимит одновременных попыток
        if len(self._attempts) >= self._config.max_concurrent_attempts:
            return None
        
        # Создаём попытку
        self._attempt_counter += 1
        attempt_id = f"fb_{self._attempt_counter}"
        
        attempt = FalseBreakoutAttempt(
            attempt_id=attempt_id,
            level_id=level.id,
            symbol=level.symbol,
            breakout_side=breakout_side,
            level_price=level.center,
            cross_price=cross_price,
            max_extension_price=cross_price,
            cross_ts_ns=ts_ns,
            state=FalseBreakoutState.MONITORING,
            state_entered_ts_ns=ts_ns,
        )
        
        self._attempts[level.id] = attempt
        
        return attempt
    
    def cancel_monitoring(self, level_id: str) -> None:
        """
        Отменяет мониторинг для уровня.
        
        Вызывается из BreakoutEngine при подтверждении истинного пробоя.
        """
        attempt = self._attempts.get(level_id)
        if attempt is not None:
            attempt.state = FalseBreakoutState.CANCELLED
            del self._attempts[level_id]
    
    def get_confirmed_false_breakouts(self) -> List[FalseBreakoutAttempt]:
        """
        Возвращает подтверждённые ложные пробои.
        
        Вызывается из SignalGenerator для обработки.
        """
        confirmed = self._confirmed.copy()
        self._confirmed.clear()
        return confirmed
    
    def get_active_attempts(self) -> List[FalseBreakoutAttempt]:
        """Возвращает все активные попытки мониторинга."""
        return [
            attempt for attempt in self._attempts.values()
            if attempt.state == FalseBreakoutState.MONITORING
        ]
    
    # ============================================
    # Реализация BaseSignalEngine
    # ============================================
    
    def evaluate(self, context: MarketContext) -> EngineResult:
        """
        Основная точка входа для движка.
        
        Проверяет все активные попытки мониторинга и
        определяет, есть ли подтверждённые ложные пробои.
        """
        self._mark_evaluation()
        
        if not self.is_enabled:
            return EngineResult.no_setup()
        
        now_ns = context.ts_ns
        
        # Проверяем каждую активную попытку
        confirmed_attempts: List[FalseBreakoutAttempt] = []
        attempts_to_remove: List[str] = []
        
        for level_id, attempt in list(self._attempts.items()):
            if attempt.state != FalseBreakoutState.MONITORING:
                continue
            
            # Проверяем, истекло ли окно детекции
            elapsed_ms = (now_ns - attempt.cross_ts_ns) / 1_000_000
            
            if elapsed_ms > self._config.window_ms:
                # Окно истекло без возврата — пробой истинный
                attempt.state = FalseBreakoutState.CANCELLED
                attempts_to_remove.append(level_id)
                continue
            
            # Обновляем метрики
            self._update_attempt_metrics(attempt, context)
            
            # Проверяем признаки ложного пробоя
            is_false, reasons = self._check_false_breakout(attempt, context)
            
            if is_false:
                # Ложный пробой подтверждён!
                attempt.state = FalseBreakoutState.CONFIRMED
                attempt.detection_reasons = reasons
                confirmed_attempts.append(attempt)
                attempts_to_remove.append(level_id)
        
        # Удаляем завершённые попытки
        for level_id in attempts_to_remove:
            self._attempts.pop(level_id, None)
        
        # Сохраняем подтверждённые ложные пробои
        for attempt in confirmed_attempts:
            self._confirmed.append(attempt)
            self._mark_rejection()
        
        # Возвращаем результат
        if confirmed_attempts:
            # Берём первый подтверждённый ложный пробой
            attempt = confirmed_attempts[0]
            
            # Определяем сторону сигнала (противоположная пробою)
            # Если был ложный пробой вверх → сигнал на шорт
            # Если был ложный пробой вниз → сигнал на лонг
            signal_side = (
                OrderSide.SELL if attempt.is_long_breakout
                else OrderSide.BUY
            )
            
            # Находим уровень
            level = None
            for lvl in context.active_levels:
                if lvl.id == attempt.level_id:
                    level = lvl
                    break
            
            if level is None:
                return EngineResult.no_setup()
            
            # Рассчитываем стоп-лосс
            stop_price = self._calculate_stop_price(attempt, signal_side)
            
            # Рассчитываем тейк-профит (возврат к цене пробоя)
            take_profit_price = attempt.cross_price
            
            return EngineResult.signal(
                setup_type=SetupType.FALSE_BREAKOUT,
                side=signal_side,
                level=level,
                entry_price=context.last_price,
                stop_price=stop_price,
                reasons=attempt.detection_reasons,
                confidence=self._calculate_confidence(attempt),
                take_profit_price=take_profit_price,
                features_snapshot={
                    "max_extension_ticks": attempt.max_extension_ticks,
                    "hold_duration_ms": attempt.hold_duration_ms,
                    "absorption_score": attempt.absorption_score,
                },
                created_ts_ns=now_ns,
            )
        
        return EngineResult.no_setup()
    
    def reset(self) -> None:
        """Сбрасывает состояние движка."""
        self._attempts.clear()
        self._confirmed.clear()
        self._attempt_counter = 0
        self._state = EngineState.IDLE
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    def _update_attempt_metrics(
        self,
        attempt: FalseBreakoutAttempt,
        context: MarketContext,
    ) -> None:
        """Обновляет метрики попытки мониторинга."""
        current_price = context.last_price
        
        # Обновляем максимальное расширение за уровень
        if attempt.is_long_breakout:
            # Для лонга: расширение вверх
            extension = current_price - attempt.level_price
            if extension > 0:
                attempt.max_extension_price = max(
                    attempt.max_extension_price, current_price
                )
                attempt.max_extension_ticks = extension / self._tick_size
            
            # Обновляем время удержания за уровнем
            if current_price > attempt.level_price:
                attempt.hold_duration_ms = (
                    (context.ts_ns - attempt.cross_ts_ns) / 1_000_000
                )
        
        else:
            # Для шорта: расширение вниз
            extension = attempt.level_price - current_price
            if extension > 0:
                attempt.max_extension_price = min(
                    attempt.max_extension_price, current_price
                )
                attempt.max_extension_ticks = extension / self._tick_size
            
            # Обновляем время удержания за уровнем
            if current_price < attempt.level_price:
                attempt.hold_duration_ms = (
                    (context.ts_ns - attempt.cross_ts_ns) / 1_000_000
                )
        
        # Рассчитываем скор поглощения
        if context.tape_metrics is not None:
            tape = context.tape_metrics
            price_displacement = abs(current_price - attempt.level_price)
            
            if price_displacement > 0:
                # Объём / смещение — чем выше, тем больше поглощение
                attempt.absorption_score = (
                    tape.total_volume_1s / price_displacement
                )
    
    def _check_false_breakout(
        self,
        attempt: FalseBreakoutAttempt,
        context: MarketContext,
    ) -> tuple:
        """
        Проверяет признаки ложного пробоя.
        
        Возвращает (is_false, reasons).
        """
        reasons: List[str] = []
        current_price = context.last_price
        
        # ============================================
        # Признак 1: Цена вернулась за уровень
        # ============================================
        return_buffer = self._config.return_inside_buffer_ticks * self._tick_size
        
        if attempt.is_long_breakout:
            # Для лонга: цена должна вернуться ниже уровня
            if current_price <= attempt.level_price + return_buffer:
                reasons.append("RETURNED_INSIDE")
        else:
            # Для шорта: цена должна вернуться выше уровня
            if current_price >= attempt.level_price - return_buffer:
                reasons.append("RETURNED_INSIDE")
        
        # ============================================
        # Признак 2: Удержание не подтверждено
        # ============================================
        elapsed_ms = (context.ts_ns - attempt.cross_ts_ns) / 1_000_000
        
        if elapsed_ms >= self._config.min_hold_ms:
            if attempt.hold_duration_ms < self._config.min_hold_ms:
                reasons.append("HOLD_NOT_CONFIRMED")
        
        # ============================================
        # Признак 3: Слабый объём
        # ============================================
        if context.tape_metrics is not None:
            tape = context.tape_metrics
            
            if attempt.is_long_breakout:
                # Для лонга: проверяем buy объём
                if tape.buy_volume_median_1s > 0:
                    volume_ratio = tape.buy_volume_1s / tape.buy_volume_median_1s
                    if volume_ratio < self._config.min_volume_burst_ratio:
                        reasons.append("WEAK_VOLUME")
            else:
                # Для шорта: проверяем sell объём
                if tape.sell_volume_median_1s > 0:
                    volume_ratio = tape.sell_volume_1s / tape.sell_volume_median_1s
                    if volume_ratio < self._config.min_volume_burst_ratio:
                        reasons.append("WEAK_VOLUME")
        
        # ============================================
        # Признак 4: Недостаточное продолжение
        # ============================================
        if attempt.max_extension_ticks < self._config.min_extension_ticks:
            reasons.append("INSUFFICIENT_EXTENSION")
        
        # ============================================
        # Признак 5: Поглощение (absorption)
        # ============================================
        if attempt.absorption_score > self._config.max_absorption_score:
            reasons.append("ABSORPTION")
        
        # ============================================
        # Признак 6: Плотность не съедена (проверяем через контекст)
        # ============================================
        if context.book_liquidity is not None:
            book = context.book_liquidity
            
            if attempt.is_long_breakout:
                # Для лонга: проверяем, съедена ли ask стена
                if not book.ask_wall_consumed:
                    reasons.append("WALL_NOT_CONSUMED")
            else:
                # Для шорта: проверяем, съедена ли bid стена
                if not book.bid_wall_consumed:
                    reasons.append("WALL_NOT_CONSUMED")
        
        # Ложный пробой подтверждён, если есть хотя бы 2 признака
        # (или признак возврата за уровень)
        is_false = (
            "RETURNED_INSIDE" in reasons
            or len(reasons) >= 2
        )
        
        return is_false, reasons
    
    def _calculate_stop_price(
        self,
        attempt: FalseBreakoutAttempt,
        signal_side: OrderSide,
    ) -> float:
        """Рассчитывает стоп-лосс для сигнала на ложном пробое."""
        buffer = self._config.return_inside_buffer_ticks * self._tick_size
        
        if signal_side == OrderSide.BUY:
            # Для лонга: стоп ниже уровня
            return attempt.level_price - buffer
        else:
            # Для шорта: стоп выше уровня
            return attempt.level_price + buffer
    
    def _calculate_confidence(self, attempt: FalseBreakoutAttempt) -> float:
        """
        Рассчитывает уверенность в сигнале ложного пробоя.
        
        Чем больше признаков ложного пробоя, тем выше уверенность.
        """
        base_confidence = 0.5
        
        # Бонус за возврат за уровень
        if "RETURNED_INSIDE" in attempt.detection_reasons:
            base_confidence += 0.2
        
        # Бонус за поглощение
        if "ABSORPTION" in attempt.detection_reasons:
            base_confidence += 0.15
        
        # Бонус за несъеденную стену
        if "WALL_NOT_CONSUMED" in attempt.detection_reasons:
            base_confidence += 0.1
        
        # Бонус за слабое продолжение
        if "INSUFFICIENT_EXTENSION" in attempt.detection_reasons:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)


# ============================================================
# Менеджер движков ложного пробоя
# ============================================================

class FalseBreakoutEngineManager:
    """
    Менеджер FalseBreakoutEngine для множества символов.
    
    Использование:
        manager = FalseBreakoutEngineManager(tick_sizes={"BTCUSDT": 0.1})
        
        # Получаем движок для символа
        engine = manager.get_or_create("BTCUSDT")
        
        # При пересечении уровня
        engine.start_monitoring(level, cross_price, side, ts_ns)
        
        # На каждом обновлении рынка
        result = manager.evaluate("BTCUSDT", context)
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._engines: Dict[str, FalseBreakoutEngine] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[FalseBreakoutConfig] = None,
    ) -> FalseBreakoutEngine:
        """Возвращает движок для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._engines:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._engines[symbol] = FalseBreakoutEngine(
                config=config,
                tick_size=tick_size,
            )
        
        return self._engines[symbol]
    
    def start_monitoring(
        self,
        symbol: str,
        level: Level,
        cross_price: float,
        breakout_side: OrderSide,
        ts_ns: int,
    ) -> Optional[FalseBreakoutAttempt]:
        """Начинает мониторинг для символа."""
        engine = self._engines.get(symbol.upper())
        if engine is None:
            return None
        return engine.start_monitoring(level, cross_price, breakout_side, ts_ns)
    
    def cancel_monitoring(self, symbol: str, level_id: str) -> None:
        """Отменяет мониторинг для уровня."""
        engine = self._engines.get(symbol.upper())
        if engine is not None:
            engine.cancel_monitoring(level_id)
    
    def evaluate(self, symbol: str, context: MarketContext) -> EngineResult:
        """Оценивает ситуацию для символа."""
        engine = self._engines.get(symbol.upper())
        if engine is None:
            return EngineResult.no_setup()
        return engine.evaluate(context)
    
    def get_confirmed_false_breakouts(
        self,
        symbol: str,
    ) -> List[FalseBreakoutAttempt]:
        """Возвращает подтверждённые ложные пробои для символа."""
        engine = self._engines.get(symbol.upper())
        if engine is None:
            return []
        return engine.get_confirmed_false_breakouts()
    
    def reset_all(self) -> None:
        """Сбрасывает все движки."""
        for engine in self._engines.values():
            engine.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._engines.keys())