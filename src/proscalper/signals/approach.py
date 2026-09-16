"""
Детектор подготовки уровня к пробою.

Объединяет данные от трёх модулей:
- CompressionDetector (консолидация у уровня)
- BookAnalyzer (плотности в стакане)
- LevelDetector (уровни)

Выдаёт состояние готовности уровня к пробою:
    IDLE → APPROACHING → CONSOLIDATING → READY → BREAKING → CONFIRMED / FAILED

Используется в связке с:
- BreakoutDetector (сигнал пробоя)
- SignalGenerator (финальный сигнал)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.features.levels import Level
from proscalper.features.compression import CompressionMetrics, CompressionDetector
from proscalper.features.book_analyzer import LevelLiquidity, BookAnalyzer


class ApproachState(Enum):
    """
    Состояние подхода цены к уровню.
    
    Жизненный цикл:
        IDLE → APPROACHING → CONSOLIDATING → READY → BREAKING → CONFIRMED
                                                        ↓
                                                     FAILED
    """
    IDLE = auto()           # Уровень не в фокусе
    APPROACHING = auto()    # Цена движется к уровню
    CONSOLIDATING = auto()  # Цена консолидируется у уровня
    READY = auto()          # Консолидация + плотность, готов к пробою
    BREAKING = auto()       # Пробой начался (цена пересекает уровень)
    CONFIRMED = auto()      # Пробой подтверждён (вход)
    FAILED = auto()         # Ложный пробой (закол)
    INVALID = auto()        # Уровень устарел или разрушен


@dataclass
class ApproachConfig:
    """
    Конфигурация детектора подхода.
    
    Все пороги подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Интервал проверки готовности (мс)
    check_interval_ms: int = 100
    
    # Максимум отслеживаемых уровней
    max_levels_to_track: int = 5
    
    # Зоны
    approach_zone_ticks: int = 20       # зона "подхода к уровню"
    ready_zone_ticks: int = 10          # зона "готовности к пробою"
    breaking_zone_ticks: int = 3        # зона "пробой начался"
    
    # Время в состояниях (мс)
    min_time_approaching_ms: int = 500      # минимум в состоянии APPROACHING
    min_time_consolidating_ms: int = 1000   # минимум в состоянии CONSOLIDATING
    
    # Условия готовности
    require_compression: bool = True    # требовать консолидацию
    require_wall: bool = True           # требовать плотность
    require_low_volume: bool = False    # требовать снижение объёма (опционально)
    
    # Хранение
    max_state_history: int = 100        # максимум истории состояний


@dataclass
class LevelApproach:
    """
    Состояние подхода к конкретному уровню.
    
    Хранит всю информацию о текущем состоянии уровня
    и его готовности к пробою.
    """
    level_id: str
    level_center: float
    side: str  # "SUPPORT" или "RESISTANCE"
    
    # Текущее состояние
    state: ApproachState = ApproachState.IDLE
    state_entered_ts_ns: int = 0
    
    # Время в текущем состоянии (мс)
    time_in_state_ms: int = 0
    
    # Метрики консолидации
    compression_metrics: Optional[CompressionMetrics] = None
    
    # Ликвидность уровня
    level_liquidity: Optional[LevelLiquidity] = None
    
    # Цена при входе в состояние
    price_at_state_enter: float = 0.0
    
    # Расстояние до уровня (в тиках)
    distance_to_level_ticks: float = 0.0
    
    # Флаги готовности
    has_compression: bool = False
    has_wall: bool = False
    has_low_volume: bool = False
    is_ready_for_breakout: bool = False
    
    @property
    def is_approaching(self) -> bool:
        """Цена движется к уровню."""
        return self.state == ApproachState.APPROACHING
    
    @property
    def is_consolidating(self) -> bool:
        """Цена консолидируется у уровня."""
        return self.state == ApproachState.CONSOLIDATING
    
    @property
    def is_ready(self) -> bool:
        """Уровень готов к пробою."""
        return self.state == ApproachState.READY
    
    @property
    def is_breaking(self) -> bool:
        """Пробой начался."""
        return self.state == ApproachState.BREAKING
    
    def time_in_state_sec(self) -> float:
        """Время в текущем состоянии в секундах."""
        return self.time_in_state_ms / 1000.0


class ApproachDetector:
    """
    Детектор подготовки уровня к пробою для одного символа.
    
    Работает на основе данных от:
    - CompressionDetector (консолидация)
    - BookAnalyzer (плотности)
    - Текущей цены (расстояние до уровня)
    
    Использование:
        detector = ApproachDetector("BTCUSDT", tick_size=0.1)
        
        # На каждом обновлении цены
        detector.on_price_update(price, ts_ns)
        
        # Периодически проверяем уровни
        detector.check_levels(active_levels, compression_metrics, book_analyzer)
        
        # Получаем уровни, готовые к пробою
        ready_levels = detector.get_ready_levels()
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[ApproachConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or ApproachConfig()
        
        # Текущая цена
        self._last_price: float = 0.0
        self._last_price_ts_ns: int = 0
        
        # Состояния подхода для уровней
        # Ключ: level_id
        self._approaches: Dict[str, LevelApproach] = {}
        
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
    
    def check_levels(
        self,
        levels: List[Level],
        compression_metrics: Optional[CompressionMetrics],
        book_analyzer: Optional[BookAnalyzer],
    ) -> None:
        """
        Проверка уровней на готовность к пробою.
        
        Вызывается периодически (каждые 100мс).
        
        Args:
            levels: Список активных уровней
            compression_metrics: Метрики консолидации
            book_analyzer: Анализатор стакана
        """
        now_ns = self._last_price_ts_ns or time.time_ns()
        
        # Проверяем интервал
        time_since_last_check_ms = (now_ns - self._last_check_ts_ns) / 1_000_000
        if time_since_last_check_ms < self.config.check_interval_ms:
            return
        
        self._last_check_ts_ns = now_ns
        
        # Сортируем уровни по близости к цене
        sorted_levels = self._sort_levels_by_distance(levels)
        
        # Берём топ-N ближайших уровней
        top_levels = sorted_levels[:self.config.max_levels_to_track]
        
        # Обновляем состояния для топ-уровней
        for level in top_levels:
            self._update_level_approach(
                level, compression_metrics, book_analyzer, now_ns
            )
        
        # Удаляем уровни, которые больше не в топ-N
        self._prune_old_approaches(top_levels)
    
    def get_approach_states(self) -> Dict[str, LevelApproach]:
        """Возвращает состояния всех отслеживаемых уровней."""
        return self._approaches.copy()
    
    def get_ready_levels(self) -> List[LevelApproach]:
        """Возвращает уровни, готовые к пробою."""
        return [
            approach for approach in self._approaches.values()
            if approach.is_ready
        ]
    
    def get_breaking_levels(self) -> List[LevelApproach]:
        """Возвращает уровни, где пробой начался."""
        return [
            approach for approach in self._approaches.values()
            if approach.is_breaking
        ]
    
    def get_approach(self, level_id: str) -> Optional[LevelApproach]:
        """Возвращает состояние подхода для конкретного уровня."""
        return self._approaches.get(level_id)
    
    def mark_level_confirmed(self, level_id: str) -> None:
        """Помечает уровень как подтверждённый пробой."""
        approach = self._approaches.get(level_id)
        if approach is not None:
            self._change_state(approach, ApproachState.CONFIRMED, self._last_price_ts_ns)
    
    def mark_level_failed(self, level_id: str) -> None:
        """Помечает уровень как ложный пробой."""
        approach = self._approaches.get(level_id)
        if approach is not None:
            self._change_state(approach, ApproachState.FAILED, self._last_price_ts_ns)
    
    def reset(self) -> None:
        """Сбрасывает состояние детектора."""
        self._approaches.clear()
        self._last_price = 0.0
        self._last_price_ts_ns = 0
        self._last_check_ts_ns = 0
    
    def _sort_levels_by_distance(self, levels: List[Level]) -> List[Level]:
        """Сортирует уровни по близости к текущей цене."""
        if not levels:
            return []
        
        def distance_key(level: Level) -> float:
            return level.distance_to(self._last_price)
        
        return sorted(levels, key=distance_key)
    
    def _update_level_approach(
        self,
        level: Level,
        compression_metrics: Optional[CompressionMetrics],
        book_analyzer: Optional[BookAnalyzer],
        now_ns: int,
    ) -> None:
        """Обновляет состояние подхода для одного уровня."""
        # Получаем или создаём состояние подхода
        if level.id not in self._approaches:
            self._approaches[level.id] = LevelApproach(
                level_id=level.id,
                level_center=level.center,
                side=level.side.name,
            )
        
        approach = self._approaches[level.id]
        
        # Обновляем расстояние до уровня
        distance = level.distance_to(self._last_price)
        approach.distance_to_level_ticks = distance / self.tick_size
        
        # Обновляем метрики консолидации
        approach.compression_metrics = compression_metrics
        
        # Обновляем ликвидность уровня
        if book_analyzer is not None:
            liquidity = book_analyzer.get_level_liquidity(level)
            approach.level_liquidity = liquidity
            
            # Проверяем наличие плотности
            approach.has_wall = liquidity.has_significant_walls
        else:
            approach.has_wall = False
        
        # Проверяем консолидацию
        if compression_metrics is not None:
            approach.has_compression = compression_metrics.is_compressing
            approach.has_low_volume = compression_metrics.is_low_volume
        else:
            approach.has_compression = False
            approach.has_low_volume = False
        
        # Обновляем время в текущем состоянии
        if approach.state_entered_ts_ns > 0:
            approach.time_in_state_ms = (now_ns - approach.state_entered_ts_ns) / 1_000_000
        
        # Определяем новое состояние
        new_state = self._determine_new_state(approach, level)
        
        # Меняем состояние, если оно изменилось
        if new_state != approach.state:
            self._change_state(approach, new_state, now_ns)
        
        # Обновляем флаг готовности
        approach.is_ready_for_breakout = (
            approach.state == ApproachState.READY
        )
    
    def _determine_new_state(
        self,
        approach: LevelApproach,
        level: Level,
    ) -> ApproachState:
        """
        Определяет новое состояние подхода.
        
        Логика:
        1. Если цена далеко от уровня → IDLE
        2. Если цена движется к уровню → APPROACHING
        3. Если цена у уровня и консолидируется → CONSOLIDATING
        4. Если есть консолидация + плотность → READY
        5. Если цена пересекает уровень → BREAKING
        """
        distance_ticks = approach.distance_to_level_ticks
        
        # Проверяем пробой (цена пересекает уровень)
        if distance_ticks <= self.config.breaking_zone_ticks:
            # Проверяем, пересекает ли цена уровень
            if self._is_crossing_level(approach, level):
                return ApproachState.BREAKING
        
        # Проверяем готовность к пробою
        if distance_ticks <= self.config.ready_zone_ticks:
            # Проверяем все условия готовности
            has_compression = not self.config.require_compression or approach.has_compression
            has_wall = not self.config.require_wall or approach.has_wall
            has_low_volume = not self.config.require_low_volume or approach.has_low_volume
            
            if has_compression and has_wall and has_low_volume:
                return ApproachState.READY
            
            # Если есть только консолидация, но нет плотности
            if approach.has_compression:
                return ApproachState.CONSOLIDATING
        
        # Проверяем подход к уровню
        if distance_ticks <= self.config.approach_zone_ticks:
            return ApproachState.APPROACHING
        
        # Цена далеко от уровня
        return ApproachState.IDLE
    
    def _is_crossing_level(
        self,
        approach: LevelApproach,
        level: Level,
    ) -> bool:
        """
        Проверяет, пересекает ли цена уровень.
        
        Для пробоя вверх (сопротивление): цена > уровень
        Для пробоя вниз (поддержка): цена < уровень
        """
        if approach.side == "RESISTANCE":
            # Пробой сопротивления вверх
            return self._last_price > level.center
        else:
            # Пробой поддержки вниз
            return self._last_price < level.center
    
    def _change_state(
        self,
        approach: LevelApproach,
        new_state: ApproachState,
        now_ns: int,
    ) -> None:
        """Меняет состояние подхода."""
        old_state = approach.state
        
        # Проверяем минимальное время в состоянии
        if not self._can_leave_state(approach, old_state):
            return
        
        # Меняем состояние
        approach.state = new_state
        approach.state_entered_ts_ns = now_ns
        approach.time_in_state_ms = 0
        approach.price_at_state_enter = self._last_price
    
    def _can_leave_state(
        self,
        approach: LevelApproach,
        state: ApproachState,
    ) -> bool:
        """
        Проверяет, можно ли покинуть текущее состояние.
        
        Некоторые состояния требуют минимального времени пребывания.
        """
        if state == ApproachState.APPROACHING:
            return approach.time_in_state_ms >= self.config.min_time_approaching_ms
        
        if state == ApproachState.CONSOLIDATING:
            return approach.time_in_state_ms >= self.config.min_time_consolidating_ms
        
        # Остальные состояния можно покинуть сразу
        return True
    
    def _prune_old_approaches(self, active_levels: List[Level]) -> None:
        """Удаляет состояния уровней, которые больше не в топ-N."""
        active_ids = {level.id for level in active_levels}
        
        keys_to_remove = [
            level_id for level_id in self._approaches
            if level_id not in active_ids
        ]
        
        for key in keys_to_remove:
            del self._approaches[key]


class ApproachManager:
    """
    Менеджер ApproachDetector'ов для множества символов.
    
    Использование:
        manager = ApproachManager(tick_sizes={"BTCUSDT": 0.1})
        
        # Получаем детектор для символа
        detector = manager.get_or_create("BTCUSDT")
        
        # Передаём обновления цены
        manager.on_price_update(price, "BTCUSDT", ts_ns)
        
        # Периодически проверяем уровни
        manager.check_levels("BTCUSDT", levels, compression, book_analyzer)
        
        # Получаем уровни, готовые к пробою
        ready = manager.get_ready_levels("BTCUSDT")
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._detectors: Dict[str, ApproachDetector] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[ApproachConfig] = None,
    ) -> ApproachDetector:
        """Возвращает детектор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._detectors:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._detectors[symbol] = ApproachDetector(
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
    
    def check_levels(
        self,
        symbol: str,
        levels: List[Level],
        compression_metrics: Optional[CompressionMetrics],
        book_analyzer: Optional[BookAnalyzer],
    ) -> None:
        """Проверка уровней для нужного символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is not None:
            detector.check_levels(levels, compression_metrics, book_analyzer)
    
    def get_ready_levels(self, symbol: str) -> List[LevelApproach]:
        """Возвращает уровни, готовые к пробою."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return []
        return detector.get_ready_levels()
    
    def get_breaking_levels(self, symbol: str) -> List[LevelApproach]:
        """Возвращает уровни, где пробой начался."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return []
        return detector.get_breaking_levels()
    
    def reset_all(self) -> None:
        """Сбрасывает все детекторы."""
        for detector in self._detectors.values():
            detector.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._detectors.keys())