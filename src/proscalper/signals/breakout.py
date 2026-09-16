"""
State machine пробоя уровня.

Финальный модуль детекции пробоя. Объединяет:
- ApproachDetector (состояния подхода)
- ImpulseScoreCalculator (сила пробоя)
- LevelDetector (уровни)

Выдаёт финальный сигнал на вход.

Жизненный цикл попытки пробоя:
    IDLE → CROSSING → CONFIRMING → CONFIRMED (вход)
                          ↓
                       FAILED (ложный пробой)
                          ↓
                       EXPIRED (таймаут)

Используется в связке с:
- SignalGenerator (финальный сигнал)
- LevelTracker (судьба уровня после пробоя)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.core.types import OrderSide
from proscalper.features.levels import Level
from proscalper.features.impulse_score import (
    ImpulseMetrics,
    ImpulseScore,
    ImpulseScoreCalculator,
    ImpulseStrength,
)


class BreakoutState(Enum):
    """
    Состояние попытки пробоя уровня.
    
    Жизненный цикл:
        IDLE → CROSSING → CONFIRMING → CONFIRMED (вход)
                              ↓
                           FAILED (ложный пробой)
                              ↓
                           EXPIRED (таймаут)
    """
    IDLE = auto()           # Нет попытки пробоя
    CROSSING = auto()       # Цена пересекает уровень
    CONFIRMING = auto()     # Ждём подтверждения (500мс)
    CONFIRMED = auto()      # Пробой подтверждён (вход)
    FAILED = auto()         # Ложный пробой (закол)
    EXPIRED = auto()        # Попытка устарела


@dataclass
class BreakoutConfig:
    """
    Конфигурация детектора пробоя.
    
    Все пороги подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Время подтверждения пробоя
    crossing_confirmation_ms: int = 500
    
    # Условия пробоя
    require_volume_burst: bool = True       # требовать всплеск объёма
    require_wall_consumed: bool = True      # требовать проедание стены
    min_displacement_ticks: float = 3.0     # минимальное смещение за уровень
    
    # Ограничения
    max_active_attempts: int = 3            # максимум одновременных попыток
    attempt_timeout_ms: int = 5000          # таймаут попытки
    
    # Интервал проверки
    check_interval_ms: int = 50
    
    # Пороги силы пробоя
    min_fast_score: float = 0.4             # минимальный скор для входа
    min_full_score: float = 0.5             # минимальный скор для подтверждения


@dataclass
class BreakoutAttempt:
    """
    Попытка пробоя конкретного уровня.
    
    Хранит всю информацию о текущей попытке пробоя
    и её состоянии.
    """
    attempt_id: str
    level_id: str
    level_center: float
    level_side: str  # "SUPPORT" или "RESISTANCE"
    
    # Текущее состояние
    state: BreakoutState = BreakoutState.IDLE
    state_entered_ts_ns: int = 0
    
    # Цена и время пересечения
    crossing_price: float = 0.0
    crossing_ts_ns: int = 0
    
    # Направление пробоя
    is_long: bool = False  # True для пробоя вверх (resistance), False для вниз (support)
    
    # Оценки силы
    fast_score: Optional[ImpulseScore] = None
    full_score: Optional[ImpulseScore] = None
    
    # Метрики импульса
    impulse_metrics: Optional[ImpulseMetrics] = None
    
    # Время в текущем состоянии (мс)
    time_in_state_ms: int = 0
    
    @property
    def direction(self) -> OrderSide:
        """Направление сделки для пробоя."""
        return OrderSide.BUY if self.is_long else OrderSide.SELL
    
    @property
    def is_crossing(self) -> bool:
        """Цена пересекает уровень."""
        return self.state == BreakoutState.CROSSING
    
    @property
    def is_confirming(self) -> bool:
        """Ждём подтверждения."""
        return self.state == BreakoutState.CONFIRMING
    
    @property
    def is_confirmed(self) -> bool:
        """Пробой подтверждён."""
        return self.state == BreakoutState.CONFIRMED
    
    @property
    def is_failed(self) -> bool:
        """Ложный пробой."""
        return self.state == BreakoutState.FAILED
    
    def time_since_crossing_ms(self, now_ns: int) -> float:
        """Время с момента пересечения уровня."""
        if self.crossing_ts_ns == 0:
            return 0.0
        return (now_ns - self.crossing_ts_ns) / 1_000_000


@dataclass
class BreakoutSignal:
    """
    Финальный сигнал на вход.
    
    Генерируется при подтверждении пробоя.
    Передаётся в RiskManager для проверки рисков
    и в ExecutionEngine для исполнения.
    """
    signal_id: str
    symbol: str
    level_id: str
    level_center: float
    
    # Направление сделки
    direction: OrderSide
    
    # Цена входа (текущая цена при генерации сигнала)
    entry_price: float
    
    # Оценки силы пробоя
    fast_score: ImpulseScore
    full_score: Optional[ImpulseScore] = None
    
    # Время создания
    created_ts_ns: int = 0
    
    # Дополнительные метрики (для логирования)
    displacement_ticks: float = 0.0
    volume_burst_ratio: float = 1.0
    wall_consumption_ratio: float = 0.0
    
    @property
    def is_strong(self) -> bool:
        """Сильный пробой."""
        return self.fast_score.is_strong
    
    @property
    def is_medium(self) -> bool:
        """Средний пробой."""
        return self.fast_score.is_medium


class BreakoutDetector:
    """
    State machine пробоя уровня для одного символа.
    
    Работает на основе данных от:
    - Текущей цены (пересечение уровня)
    - ImpulseScoreCalculator (сила пробоя)
    - Уровней (активные уровни)
    
    Использование:
        detector = BreakoutDetector("BTCUSDT", tick_size=0.1)
        
        # На каждом обновлении цены
        detector.on_price_update(price, ts_ns)
        
        # Периодически проверяем попытки пробоя
        signals = detector.check_breakouts(active_levels, impulse_calculator)
        
        # Обрабатываем подтверждённые сигналы
        for signal in signals:
            print(f"Пробой уровня {signal.level_center}, вход {signal.direction}")
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[BreakoutConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or BreakoutConfig()
        
        # Текущая цена
        self._last_price: float = 0.0
        self._last_price_ts_ns: int = 0
        
        # Активные попытки пробоя
        # Ключ: level_id
        self._attempts: Dict[str, BreakoutAttempt] = {}
        
        # Подтверждённые сигналы (ожидают обработки)
        self._confirmed_signals: List[BreakoutSignal] = []
        
        # Время последней проверки
        self._last_check_ts_ns: int = 0
    
    def on_price_update(self, price: float, ts_ns: int) -> None:
        """
        Обработка обновления цены.
        
        Вызывается на каждом тике цены (из bookTicker или последней сделки).
        """
        if price <= 0:
            return
        
        self._last_price = price
        self._last_price_ts_ns = ts_ns
    
    def on_impulse_update(self, metrics: ImpulseMetrics) -> None:
        """
        Обработка обновления метрик импульса.
        
        Вызывается при получении новых данных от ленты/стакана.
        """
        # Обновляем метрики для активных попыток
        for attempt in self._attempts.values():
            if attempt.state in (BreakoutState.CROSSING, BreakoutState.CONFIRMING):
                attempt.impulse_metrics = metrics
    
    def check_breakouts(
        self,
        levels: List[Level],
        impulse_calculator: ImpulseScoreCalculator,
    ) -> List[BreakoutSignal]:
        """
        Проверка попыток пробоя для активных уровней.
        
        Вызывается периодически (каждые 50мс).
        
        Возвращает список подтверждённых сигналов.
        
        Args:
            levels: Список активных уровней
            impulse_calculator: Калькулятор силы импульса
            
        Returns:
            Список новых подтверждённых сигналов
        """
        now_ns = self._last_price_ts_ns or time.time_ns()
        
        # Проверяем интервал
        time_since_last_check_ms = (now_ns - self._last_check_ts_ns) / 1_000_000
        if time_since_last_check_ms < self.config.check_interval_ms:
            return []
        
        self._last_check_ts_ns = now_ns
        
        # Обновляем существующие попытки
        self._update_existing_attempts(now_ns, impulse_calculator)
        
        # Проверяем новые попытки пробоя
        self._check_new_breakouts(levels, now_ns)
        
        # Удаляем устаревшие попытки
        self._prune_expired_attempts(now_ns)
        
        # Возвращаем подтверждённые сигналы
        signals = self._confirmed_signals.copy()
        self._confirmed_signals.clear()
        
        return signals
    
    def get_active_attempts(self) -> List[BreakoutAttempt]:
        """Возвращает все активные попытки пробоя."""
        return [
            attempt for attempt in self._attempts.values()
            if attempt.state in (BreakoutState.CROSSING, BreakoutState.CONFIRMING)
        ]
    
    def get_attempt(self, level_id: str) -> Optional[BreakoutAttempt]:
        """Возвращает попытку пробоя для конкретного уровня."""
        return self._attempts.get(level_id)
    
    def reset(self) -> None:
        """Сбрасывает состояние детектора."""
        self._attempts.clear()
        self._confirmed_signals.clear()
        self._last_price = 0.0
        self._last_price_ts_ns = 0
        self._last_check_ts_ns = 0
    
    def _check_new_breakouts(
        self,
        levels: List[Level],
        now_ns: int,
    ) -> None:
        """Проверяет новые попытки пробоя."""
        # Проверяем лимит одновременных попыток
        active_count = len(self.get_active_attempts())
        if active_count >= self.config.max_active_attempts:
            return
        
        for level in levels:
            # Пропускаем уровни, для которых уже есть попытка
            if level.id in self._attempts:
                continue
            
            # Проверяем, пересекает ли цена уровень
            is_crossing = self._is_price_crossing_level(level)
            
            if is_crossing:
                # Создаём новую попытку пробоя
                self._create_new_attempt(level, now_ns)
    
    def _is_price_crossing_level(self, level: Level) -> bool:
        """
        Проверяет, пересекает ли цена уровень.
        
        Для пробоя вверх (сопротивление): цена > уровень + смещение
        Для пробоя вниз (поддержка): цена < уровень - смещение
        """
        min_displacement = self.config.min_displacement_ticks * self.tick_size
        
        if level.side.name == "RESISTANCE":
            # Пробой сопротивления вверх
            return self._last_price > level.center + min_displacement
        else:
            # Пробой поддержки вниз
            return self._last_price < level.center - min_displacement
    
    def _create_new_attempt(self, level: Level, now_ns: int) -> None:
        """Создаёт новую попытку пробоя."""
        attempt = BreakoutAttempt(
            attempt_id=str(uuid.uuid4()),
            level_id=level.id,
            level_center=level.center,
            level_side=level.side.name,
            state=BreakoutState.CROSSING,
            state_entered_ts_ns=now_ns,
            crossing_price=self._last_price,
            crossing_ts_ns=now_ns,
            is_long=(level.side.name == "RESISTANCE"),  # Пробой сопротивления = лонг
        )
        
        self._attempts[level.id] = attempt
    
    def _update_existing_attempts(
        self,
        now_ns: int,
        impulse_calculator: ImpulseScoreCalculator,
    ) -> None:
        """Обновляет существующие попытки пробоя."""
        for level_id, attempt in list(self._attempts.items()):
            # Обновляем время в состоянии
            if attempt.state_entered_ts_ns > 0:
                attempt.time_in_state_ms = (
                    (now_ns - attempt.state_entered_ts_ns) / 1_000_000
                )
            
            # Обрабатываем состояние
            if attempt.state == BreakoutState.CROSSING:
                self._process_crossing_state(attempt, now_ns, impulse_calculator)
            
            elif attempt.state == BreakoutState.CONFIRMING:
                self._process_confirming_state(attempt, now_ns, impulse_calculator)
    
    def _process_crossing_state(
        self,
        attempt: BreakoutAttempt,
        now_ns: int,
        impulse_calculator: ImpulseScoreCalculator,
    ) -> None:
        """
        Обработка состояния CROSSING.
        
        Проверяем условия пробоя и переходим в CONFIRMING.
        """
        # Проверяем, что цена всё ещё за уровнем
        if not self._is_price_still_beyond_level(attempt):
            # Цена вернулась - ложный пробой
            self._change_state(attempt, BreakoutState.FAILED, now_ns)
            return
        
        # Проверяем условия пробоя
        if self._check_breakout_conditions(attempt):
            # Рассчитываем быструю оценку силы
            if attempt.impulse_metrics is not None:
                fast_score = impulse_calculator.calculate_fast(attempt.impulse_metrics)
                attempt.fast_score = fast_score
                
                # Проверяем минимальный скор для входа
                if fast_score.score >= self.config.min_fast_score:
                    self._change_state(attempt, BreakoutState.CONFIRMING, now_ns)
                else:
                    # Недостаточная сила - ложный пробой
                    self._change_state(attempt, BreakoutState.FAILED, now_ns)
            else:
                # Нет метрик импульса - переходим в CONFIRMING без оценки
                self._change_state(attempt, BreakoutState.CONFIRMING, now_ns)
    
    def _process_confirming_state(
        self,
        attempt: BreakoutAttempt,
        now_ns: int,
        impulse_calculator: ImpulseScoreCalculator,
    ) -> None:
        """
        Обработка состояния CONFIRMING.
        
        Ждём подтверждения пробоя в течение 500мс.
        """
        # Проверяем, что цена всё ещё за уровнем
        if not self._is_price_still_beyond_level(attempt):
            # Цена вернулась - ложный пробой
            self._change_state(attempt, BreakoutState.FAILED, now_ns)
            return
        
        # Проверяем время подтверждения
        time_confirming_ms = (now_ns - attempt.state_entered_ts_ns) / 1_000_000
        
        if time_confirming_ms >= self.config.crossing_confirmation_ms:
            # Рассчитываем полную оценку силы
            if attempt.impulse_metrics is not None:
                full_score = impulse_calculator.calculate_full(attempt.impulse_metrics)
                attempt.full_score = full_score
                
                # Проверяем минимальный скор для подтверждения
                if full_score.score >= self.config.min_full_score:
                    # Пробой подтверждён - генерируем сигнал
                    self._change_state(attempt, BreakoutState.CONFIRMED, now_ns)
                    self._generate_signal(attempt, now_ns)
                else:
                    # Недостаточная сила - ложный пробой
                    self._change_state(attempt, BreakoutState.FAILED, now_ns)
            else:
                # Нет метрик - подтверждаем без оценки
                self._change_state(attempt, BreakoutState.CONFIRMED, now_ns)
                self._generate_signal(attempt, now_ns)
    
    def _is_price_still_beyond_level(self, attempt: BreakoutAttempt) -> bool:
        """Проверяет, что цена всё ещё за уровнем."""
        if attempt.is_long:
            # Для пробоя вверх цена должна быть выше уровня
            return self._last_price > attempt.level_center
        else:
            # Для пробоя вниз цена должна быть ниже уровня
            return self._last_price < attempt.level_center
    
    def _check_breakout_conditions(self, attempt: BreakoutAttempt) -> bool:
        """
        Проверяет условия пробоя.
        
        Условия:
        1. Всплеск объёма (если требуется)
        2. Проедание стены (если требуется)
        """
        if attempt.impulse_metrics is None:
            # Нет метрик - считаем условия выполненными
            return True
        
        metrics = attempt.impulse_metrics
        
        # Проверяем всплеск объёма
        if self.config.require_volume_burst:
            if not metrics.has_volume_burst:
                return False
        
        # Проверяем проедание стены
        if self.config.require_wall_consumed:
            if not metrics.has_wall_consumed:
                return False
        
        return True
    
    def _change_state(
        self,
        attempt: BreakoutAttempt,
        new_state: BreakoutState,
        now_ns: int,
    ) -> None:
        """Меняет состояние попытки пробоя."""
        attempt.state = new_state
        attempt.state_entered_ts_ns = now_ns
        attempt.time_in_state_ms = 0
    
    def _generate_signal(self, attempt: BreakoutAttempt, now_ns: int) -> None:
        """Генерирует сигнал на вход."""
        signal = BreakoutSignal(
            signal_id=str(uuid.uuid4()),
            symbol=self.symbol,
            level_id=attempt.level_id,
            level_center=attempt.level_center,
            direction=attempt.direction,
            entry_price=self._last_price,
            fast_score=attempt.fast_score,
            full_score=attempt.full_score,
            created_ts_ns=now_ns,
            displacement_ticks=(
                abs(self._last_price - attempt.level_center) / self.tick_size
            ),
            volume_burst_ratio=(
                attempt.impulse_metrics.volume_burst_ratio
                if attempt.impulse_metrics else 1.0
            ),
            wall_consumption_ratio=(
                attempt.impulse_metrics.wall_consumption_ratio
                if attempt.impulse_metrics else 0.0
            ),
        )
        
        self._confirmed_signals.append(signal)
    
    def _prune_expired_attempts(self, now_ns: int) -> None:
        """Удаляет устаревшие попытки пробоя."""
        timeout_ns = self.config.attempt_timeout_ms * 1_000_000
        
        keys_to_remove = []
        for level_id, attempt in self._attempts.items():
            # Проверяем таймаут для активных попыток
            if attempt.state in (BreakoutState.CROSSING, BreakoutState.CONFIRMING):
                age_ns = now_ns - attempt.state_entered_ts_ns
                if age_ns > timeout_ns:
                    attempt.state = BreakoutState.EXPIRED
                    keys_to_remove.append(level_id)
            
            # Удаляем завершённые попытки
            elif attempt.state in (
                BreakoutState.CONFIRMED,
                BreakoutState.FAILED,
                BreakoutState.EXPIRED,
            ):
                keys_to_remove.append(level_id)
        
        for key in keys_to_remove:
            del self._attempts[key]


class BreakoutManager:
    """
    Менеджер BreakoutDetector'ов для множества символов.
    
    Использование:
        manager = BreakoutManager(tick_sizes={"BTCUSDT": 0.1})
        
        # Получаем детектор для символа
        detector = manager.get_or_create("BTCUSDT")
        
        # Передаём обновления цены
        manager.on_price_update(price, "BTCUSDT", ts_ns)
        
        # Периодически проверяем попытки пробоя
        signals = manager.check_breakouts("BTCUSDT", levels, impulse_calculator)
        
        # Обрабатываем подтверждённые сигналы
        for signal in signals:
            print(f"Пробой уровня {signal.level_center}")
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._detectors: Dict[str, BreakoutDetector] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[BreakoutConfig] = None,
    ) -> BreakoutDetector:
        """Возвращает детектор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._detectors:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._detectors[symbol] = BreakoutDetector(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        
        return self._detectors[symbol]
    
    def on_price_update(
        self,
        price: float,
        symbol: str,
        ts_ns: int,
    ) -> None:
        """Обработка обновления цены для нужного символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is not None:
            detector.on_price_update(price, ts_ns)
    
    def check_breakouts(
        self,
        symbol: str,
        levels: List[Level],
        impulse_calculator: ImpulseScoreCalculator,
    ) -> List[BreakoutSignal]:
        """Проверка попыток пробоя для нужного символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return []
        return detector.check_breakouts(levels, impulse_calculator)
    
    def get_active_attempts(self, symbol: str) -> List[BreakoutAttempt]:
        """Возвращает активные попытки пробоя для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return []
        return detector.get_active_attempts()
    
    def reset_all(self) -> None:
        """Сбрасывает все детекторы."""
        for detector in self._detectors.values():
            detector.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._detectors.keys())