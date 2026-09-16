"""
Модуль торговли ретеста уровней после пробоя.

Ретест — это отдельный сетап системы ProScalping:
- Resistance пробит вверх → становится support
- Цена возвращается к пробитому уровню
- Если цена отскакивает от уровня (не пробивает обратно) → вход в лонг

Для шорта зеркально:
- Support пробит вниз → становится resistance
- Цена возвращается к уровню снизу
- Если цена отскакивает вниз → вход в шорт

Логика ретеста:
1. Уровень пробит истинным пробоем → переводится в RETESTABLE
2. Цена возвращается к зоне уровня (подход)
3. Объём на откате снижен (нет агрессивного давления)
4. Цена показывает реакцию от уровня (отскок)
5. Подтверждение: цена удерживается за уровнем
6. Вход: при подтверждении отскока

Отличие от пробоя:
- Пробой: входим при пересечении уровня с импульсом
- Ретест: входим при возврате к пробитому уровню и отскоке

Используется в связке с:
- LevelTracker (состояния уровней, RETESTABLE)
- TapeAnalyzer (объём отката, дельта)
- BookAnalyzer (поддержка/давление в стакане)
- SignalGenerator (агрегация сигналов)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.core.types import OrderSide
from proscalper.features.levels import Level, LevelSide, LevelState


# ============================================================
# Конфигурация ретеста
# ============================================================

@dataclass
class RetestConfig:
    """
    Конфигурация детектора ретестов.
    
    Все параметры подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Окно для ретеста после пробоя
    retest_window_sec: float = 300.0        # 5 минут после пробоя
    
    # Зона подхода к уровню
    approach_zone_ticks: int = 10           # зона "возврата к уровню"
    
    # Глубина ретеста
    max_retest_depth_ticks: int = 5         # макс. глубина входа в зону
    
    # Параметры отката
    min_pullback_volume_decay: float = 0.5  # объём отката < 50% от пробоя
    max_pullback_delta_ratio: float = 0.3   # дельта отката < 30% от пробоя
    
    # Подтверждение отскока
    confirmation_ms: int = 300              # время подтверждения отскока
    min_bounce_ticks: int = 3               # мин. отскок от уровня в тиках
    
    # Требования к стакану
    require_book_support: bool = True       # требовать поддержку в стакане
    min_support_score: float = 2.0          # минимальный скор поддержки
    
    # Ограничения
    max_retest_attempts_per_level: int = 2  # макс. попыток ретеста на уровень
    cooldown_after_failed_retest_sec: float = 60.0  # кулдаун после неудачи


# ============================================================
# Состояния ретеста
# ============================================================

class RetestState(Enum):
    """
    Состояние попытки ретеста.
    
    Жизненный цикл:
        WAITING → APPROACHING → TESTING → CONFIRMING → CONFIRMED
                                            ↓
                                         FAILED
    """
    WAITING = auto()        # Уровень пробит, ждём возврата цены
    APPROACHING = auto()    # Цена возвращается к уровню
    TESTING = auto()        # Цена тестирует уровень (в зоне)
    CONFIRMING = auto()     # Подтверждаем отскок от уровня
    CONFIRMED = auto()      # Ретест подтверждён, можно входить
    FAILED = auto()         # Ретест не удался (пробой обратно)
    EXPIRED = auto()        # Окно ретеста истекло


# ============================================================
# Попытка ретеста
# ============================================================

@dataclass
class RetestAttempt:
    """
    Попытка ретеста конкретного уровня.
    
    Хранит всю информацию о текущей попытке ретеста
    и её состоянии.
    """
    # Идентификация
    attempt_id: str
    level_id: str
    symbol: str
    
    # Направление ретеста
    # Лонг: бывший resistance пробит вверх, цена возвращается сверху
    # Шорт: бывший support пробит вниз, цена возвращается снизу
    direction: OrderSide
    
    # Цены
    level_price: float
    breakout_price: float             # цена пробоя
    breakout_ts_ns: int               # время пробоя
    
    # Состояние
    state: RetestState = RetestState.WAITING
    state_entered_ts_ns: int = 0
    
    # Метрики отката
    pullback_volume: float = 0.0      # объём отката
    pullback_delta: float = 0.0       # дельта отката
    breakout_volume: float = 0.0      # объём пробоя (для сравнения)
    
    # Метрики отскока
    bounce_ticks: float = 0.0         # отскок от уровня в тиках
    confirmation_started_ts_ns: int = 0
    
    # Стакан
    support_score: float = 0.0        # скор поддержки (для лонга)
    pressure_score: float = 0.0       # скор давления (для шорта)
    
    # Результат
    attempts_count: int = 0
    is_valid: bool = True
    failure_reason: str = ""
    
    @property
    def is_long_retest(self) -> bool:
        """Ретест в лонг (бывший resistance → support)."""
        return self.direction == OrderSide.BUY
    
    @property
    def is_short_retest(self) -> bool:
        """Ретест в шорт (бывший support → resistance)."""
        return self.direction == OrderSide.SELL
    
    @property
    def time_since_breakout_sec(self) -> float:
        """Время с момента пробоя в секундах."""
        now_ns = time.time_ns()
        return (now_ns - self.breakout_ts_ns) / 1_000_000_000
    
    @property
    def is_window_expired(self) -> bool:
        """Окно ретеста истекло."""
        return False  # определяется в детекторе


# ============================================================
# Детектор ретестов
# ============================================================

class RetestDetector:
    """
    Детектор ретестов для одного символа.
    
    Отслеживает пробитые уровни в состоянии RETESTABLE
    и детектит возврат цены к ним с последующим отскоком.
    
    Использование:
        detector = RetestDetector("BTCUSDT", tick_size=0.1)
        
        # При пробое уровня
        detector.on_breakout(level, breakout_price, volume)
        
        # На каждом обновлении цены
        detector.on_price_update(price, tape_metrics, book_liquidity)
        
        # Получаем подтверждённые ретесты
        confirmed = detector.get_confirmed_retests()
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[RetestConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or RetestConfig()
        
        # Активные попытки ретеста
        self._attempts: Dict[str, RetestAttempt] = {}
        
        # Счётчик попыток
        self._attempt_counter: int = 0
        
        # Подтверждённые ретесты (ожидают обработки)
        self._confirmed: List[RetestAttempt] = []
        
        # Кулдауны после неудачных ретестов
        self._cooldown_until: Dict[str, int] = {}  # level_id -> ts_ns
    
    def on_breakout(
        self,
        level: Level,
        breakout_price: float,
        breakout_volume: float,
        ts_ns: int,
    ) -> Optional[RetestAttempt]:
        """
        Обработка пробоя уровня.
        
        Вызывается из BreakoutDetector при подтверждении пробоя.
        Создаёт новую попытку ретеста для уровня.
        
        Для лонга: бывший resistance пробит вверх
        Для шорта: бывший support пробит вниз
        """
        # Проверяем кулдаун
        cooldown_until = self._cooldown_until.get(level.id, 0)
        if ts_ns < cooldown_until:
            return None
        
        # Проверяем лимит попыток
        existing = self._attempts.get(level.id)
        if existing is not None and existing.attempts_count >= self.config.max_retest_attempts_per_level:
            return None
        
        # Определяем направление ретеста
        if level.side == LevelSide.RESISTANCE:
            # Resistance пробит вверх → ретест в лонг
            direction = OrderSide.BUY
        else:
            # Support пробит вниз → ретест в шорт
            direction = OrderSide.SELL
        
        # Создаём попытку
        self._attempt_counter += 1
        attempt_id = f"retest_{self.symbol}_{self._attempt_counter}"
        
        attempt = RetestAttempt(
            attempt_id=attempt_id,
            level_id=level.id,
            symbol=self.symbol,
            direction=direction,
            level_price=level.center,
            breakout_price=breakout_price,
            breakout_ts_ns=ts_ns,
            breakout_volume=breakout_volume,
            state=RetestState.WAITING,
            state_entered_ts_ns=ts_ns,
        )
        
        self._attempts[level.id] = attempt
        
        return attempt
    
    def on_price_update(
        self,
        price: float,
        ts_ns: int,
        pullback_volume: float = 0.0,
        pullback_delta: float = 0.0,
        support_score: float = 0.0,
        pressure_score: float = 0.0,
    ) -> None:
        """
        Обработка обновления цены.
        
        Вызывается на каждом обновлении цены (из bookTicker или ленты).
        Обновляет состояния всех активных попыток ретеста.
        """
        for level_id, attempt in list(self._attempts.items()):
            self._update_attempt(
                attempt=attempt,
                price=price,
                ts_ns=ts_ns,
                pullback_volume=pullback_volume,
                pullback_delta=pullback_delta,
                support_score=support_score,
                pressure_score=pressure_score,
            )
    
    def get_confirmed_retests(self) -> List[RetestAttempt]:
        """
        Возвращает подтверждённые ретесты.
        
        Вызывается из SignalGenerator для генерации сигналов.
        """
        confirmed = self._confirmed.copy()
        self._confirmed.clear()
        return confirmed
    
    def get_active_attempts(self) -> List[RetestAttempt]:
        """Возвращает все активные попытки ретеста."""
        return [
            attempt for attempt in self._attempts.values()
            if attempt.state in (
                RetestState.WAITING,
                RetestState.APPROACHING,
                RetestState.TESTING,
                RetestState.CONFIRMING,
            )
        ]
    
    def get_attempt(self, level_id: str) -> Optional[RetestAttempt]:
        """Возвращает попытку ретеста для уровня."""
        return self._attempts.get(level_id)
    
    def reset(self) -> None:
        """Сбрасывает состояние детектора."""
        self._attempts.clear()
        self._confirmed.clear()
        self._cooldown_until.clear()
        self._attempt_counter = 0
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    def _update_attempt(
        self,
        attempt: RetestAttempt,
        price: float,
        ts_ns: int,
        pullback_volume: float,
        pullback_delta: float,
        support_score: float,
        pressure_score: float,
    ) -> None:
        """Обновляет состояние одной попытки ретеста."""
        # Проверяем истечение окна ретеста
        time_since_breakout_sec = (ts_ns - attempt.breakout_ts_ns) / 1_000_000_000
        if time_since_breakout_sec > self.config.retest_window_sec:
            attempt.state = RetestState.EXPIRED
            self._attempts.pop(attempt.level_id, None)
            return
        
        # Обновляем метрики
        attempt.pullback_volume = pullback_volume
        attempt.pullback_delta = pullback_delta
        attempt.support_score = support_score
        attempt.pressure_score = pressure_score
        
        # Обрабатываем текущее состояние
        if attempt.state == RetestState.WAITING:
            self._process_waiting(attempt, price, ts_ns)
        
        elif attempt.state == RetestState.APPROACHING:
            self._process_approaching(attempt, price, ts_ns)
        
        elif attempt.state == RetestState.TESTING:
            self._process_testing(attempt, price, ts_ns)
        
        elif attempt.state == RetestState.CONFIRMING:
            self._process_confirming(attempt, price, ts_ns)
    
    def _process_waiting(
        self,
        attempt: RetestAttempt,
        price: float,
        ts_ns: int,
    ) -> None:
        """
        Обработка состояния WAITING.
        
        Ждём, пока цена начнёт возвращаться к уровню.
        """
        approach_distance = self.config.approach_zone_ticks * self.tick_size
        
        if attempt.is_long_retest:
            # Для лонга: цена должна вернуться сверху к уровню
            if price <= attempt.level_price + approach_distance:
                self._change_state(attempt, RetestState.APPROACHING, ts_ns)
        else:
            # Для шорта: цена должна вернуться снизу к уровню
            if price >= attempt.level_price - approach_distance:
                self._change_state(attempt, RetestState.APPROACHING, ts_ns)
    
    def _process_approaching(
        self,
        attempt: RetestAttempt,
        price: float,
        ts_ns: int,
    ) -> None:
        """
        Обработка состояния APPROACHING.
        
        Цена возвращается к уровню. Проверяем:
        - Объём отката снижен
        - Дельта отката слабая
        """
        approach_distance = self.config.approach_zone_ticks * self.tick_size
        zone_distance = self.config.max_retest_depth_ticks * self.tick_size
        
        # Проверяем, вошла ли цена в зону уровня
        if attempt.is_long_retest:
            # Для лонга: цена входит в зону сверху
            if price <= attempt.level_price + zone_distance:
                # Проверяем объём отката
                if self._is_pullback_weak(attempt):
                    self._change_state(attempt, RetestState.TESTING, ts_ns)
                else:
                    # Объём отката слишком большой — ретест не удался
                    self._fail_attempt(attempt, "PULLBACK_VOLUME_TOO_HIGH")
        else:
            # Для шорта: цена входит в зону снизу
            if price >= attempt.level_price - zone_distance:
                if self._is_pullback_weak(attempt):
                    self._change_state(attempt, RetestState.TESTING, ts_ns)
                else:
                    self._fail_attempt(attempt, "PULLBACK_VOLUME_TOO_HIGH")
    
    def _process_testing(
        self,
        attempt: RetestAttempt,
        price: float,
        ts_ns: int,
    ) -> None:
        """
        Обработка состояния TESTING.
        
        Цена тестирует уровень. Проверяем:
        - Цена не пробивает уровень обратно
        - Есть поддержка/давление в стакане
        - Начинается отскок
        """
        zone_distance = self.config.max_retest_depth_ticks * self.tick_size
        bounce_distance = self.config.min_bounce_ticks * self.tick_size
        
        if attempt.is_long_retest:
            # Для лонга: цена не должна уйти под уровень
            if price < attempt.level_price - zone_distance:
                # Цена пробила уровень обратно — ретест не удался
                self._fail_attempt(attempt, "PRICE_BROKE_BACK")
                return
            
            # Проверяем поддержку в стакане
            if self.config.require_book_support:
                if attempt.support_score < self.config.min_support_score:
                    # Недостаточная поддержка — ждём
                    return
            
            # Проверяем отскок
            bounce = price - attempt.level_price
            if bounce >= bounce_distance:
                attempt.bounce_ticks = bounce / self.tick_size
                self._change_state(attempt, RetestState.CONFIRMING, ts_ns)
                attempt.confirmation_started_ts_ns = ts_ns
        
        else:
            # Для шорта: цена не должна уйти над уровень
            if price > attempt.level_price + zone_distance:
                self._fail_attempt(attempt, "PRICE_BROKE_BACK")
                return
            
            # Проверяем давление в стакане
            if self.config.require_book_support:
                if attempt.pressure_score < self.config.min_support_score:
                    return
            
            # Проверяем отскок вниз
            bounce = attempt.level_price - price
            if bounce >= bounce_distance:
                attempt.bounce_ticks = bounce / self.tick_size
                self._change_state(attempt, RetestState.CONFIRMING, ts_ns)
                attempt.confirmation_started_ts_ns = ts_ns
    
    def _process_confirming(
        self,
        attempt: RetestAttempt,
        price: float,
        ts_ns: int,
    ) -> None:
        """
        Обработка состояния CONFIRMING.
        
        Подтверждаем отскок: цена должна удерживаться
        за уровнем в течение confirmation_ms.
        """
        zone_distance = self.config.max_retest_depth_ticks * self.tick_size
        
        # Проверяем, не пробила ли цена уровень обратно
        if attempt.is_long_retest:
            if price < attempt.level_price - zone_distance:
                self._fail_attempt(attempt, "PRICE_BROKE_BACK_DURING_CONFIRMATION")
                return
        else:
            if price > attempt.level_price + zone_distance:
                self._fail_attempt(attempt, "PRICE_BROKE_BACK_DURING_CONFIRMATION")
                return
        
        # Проверяем время подтверждения
        confirmation_elapsed_ms = (ts_ns - attempt.confirmation_started_ts_ns) / 1_000_000
        
        if confirmation_elapsed_ms >= self.config.confirmation_ms:
            # Ретест подтверждён!
            attempt.state = RetestState.CONFIRMED
            self._confirmed.append(attempt)
    
    def _is_pullback_weak(self, attempt: RetestAttempt) -> bool:
        """
        Проверяет, что откат слабый (нет агрессивного давления).
        
        Объём отката должен быть меньше, чем объём пробоя.
        Дельта отката должна быть слабой.
        """
        # Проверка объёма
        if attempt.breakout_volume > 0:
            volume_ratio = attempt.pullback_volume / attempt.breakout_volume
            if volume_ratio > self.config.min_pullback_volume_decay:
                return False
        
        # Проверка дельты
        if attempt.breakout_volume > 0:
            delta_ratio = abs(attempt.pullback_delta) / attempt.breakout_volume
            if delta_ratio > self.config.max_pullback_delta_ratio:
                return False
        
        return True
    
    def _change_state(
        self,
        attempt: RetestAttempt,
        new_state: RetestState,
        ts_ns: int,
    ) -> None:
        """Меняет состояние попытки."""
        attempt.state = new_state
        attempt.state_entered_ts_ns = ts_ns
    
    def _fail_attempt(
        self,
        attempt: RetestAttempt,
        reason: str,
    ) -> None:
        """Помечает попытку как неудачную."""
        attempt.state = RetestState.FAILED
        attempt.is_valid = False
        attempt.failure_reason = reason
        attempt.attempts_count += 1
        
        # Устанавливаем кулдаун
        cooldown_ns = int(self.config.cooldown_after_failed_retest_sec * 1_000_000_000)
        self._cooldown_until[attempt.level_id] = time.time_ns() + cooldown_ns
        
        # Удаляем попытку
        self._attempts.pop(attempt.level_id, None)


# ============================================================
# Менеджер детекторов ретестов
# ============================================================

class RetestDetectorManager:
    """
    Менеджер RetestDetector'ов для множества символов.
    
    Использование:
        manager = RetestDetectorManager(tick_sizes={"BTCUSDT": 0.1})
        
        # Получаем детектор для символа
        detector = manager.get_or_create("BTCUSDT")
        
        # При пробое уровня
        manager.on_breakout("BTCUSDT", level, price, volume)
        
        # На каждом обновлении цены
        manager.on_price_update("BTCUSDT", price, ts_ns)
        
        # Получаем подтверждённые ретесты
        confirmed = manager.get_confirmed_retests("BTCUSDT")
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._detectors: Dict[str, RetestDetector] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[RetestConfig] = None,
    ) -> RetestDetector:
        """Возвращает детектор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._detectors:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._detectors[symbol] = RetestDetector(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        
        return self._detectors[symbol]
    
    def on_breakout(
        self,
        symbol: str,
        level: Level,
        breakout_price: float,
        breakout_volume: float,
        ts_ns: int,
    ) -> Optional[RetestAttempt]:
        """Обработка пробоя уровня для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return None
        return detector.on_breakout(level, breakout_price, breakout_volume, ts_ns)
    
    def on_price_update(
        self,
        symbol: str,
        price: float,
        ts_ns: int,
        pullback_volume: float = 0.0,
        pullback_delta: float = 0.0,
        support_score: float = 0.0,
        pressure_score: float = 0.0,
    ) -> None:
        """Обработка обновления цены для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is not None:
            detector.on_price_update(
                price=price,
                ts_ns=ts_ns,
                pullback_volume=pullback_volume,
                pullback_delta=pullback_delta,
                support_score=support_score,
                pressure_score=pressure_score,
            )
    
    def get_confirmed_retests(self, symbol: str) -> List[RetestAttempt]:
        """Возвращает подтверждённые ретесты для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return []
        return detector.get_confirmed_retests()
    
    def get_active_attempts(self, symbol: str) -> List[RetestAttempt]:
        """Возвращает активные попытки ретеста для символа."""
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