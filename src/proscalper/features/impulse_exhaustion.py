"""
Детектор исчерпания импульса.

Определяет, когда импульс пробоя исчерпан и пора выходить
из позиции в точке экстремума.

Метрики исчерпания:
- Угасание объёма (объём падает после всплеска)
- Появление стен против движения (крупные лимитные заявки)
- Разворот дельты (дельта меняет направление)
- Замедление движения (скорость цены падает)

Используется в связке с:
- PositionManager (выход из позиции)
- TapeAnalyzer (метрики ленты)
- BookAnalyzer (метрики стакана)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class ExhaustionConfig:
    """
    Конфигурация детектора исчерпания импульса.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    # Окна анализа
    analysis_window_sec: int = 30         # окно анализа в секундах
    
    # Угасание объёма
    volume_decay_threshold: float = 0.5   # объём упал до 50% от пика
    
    # Появление стен
    wall_appearance_threshold: float = 3.0  # стена > 3× медианы
    
    # Разворот дельты
    delta_reversal_ticks: int = 5         # дельта изменилась на 5 тиков
    
    # Замедление движения
    speed_decay_threshold: float = 0.3    # скорость упала до 30% от пика
    
    # Итоговый порог
    exhaustion_score_threshold: float = 0.6  # порог исчерпания


@dataclass
class ExhaustionMetrics:
    """Метрики исчерпания импульса."""
    ts_ns: int
    symbol: str
    
    # Угасание объёма
    volume_current: float = 0.0
    volume_peak: float = 0.0
    volume_decay_ratio: float = 0.0       # 0 = нет угасания, 1 = полное
    
    # Появление стен
    walls_against_direction: int = 0
    wall_notional_against: float = 0.0
    
    # Разворот дельты
    delta_current: float = 0.0
    delta_reversed: bool = False
    
    # Замедление движения
    speed_current: float = 0.0
    speed_peak: float = 0.0
    speed_decay_ratio: float = 0.0        # 0 = нет замедления, 1 = полная остановка
    
    # Итоговый скор исчерпания
    exhaustion_score: float = 0.0
    is_exhausted: bool = False


class ImpulseExhaustionDetector:
    """
    Детектор исчерпания импульса для одного символа.
    
    Анализирует метрики ленты и стакана для определения
    момента, когда импульс пробоя исчерпан.
    
    Использование:
        detector = ImpulseExhaustionDetector("BTCUSDT")
        
        # Обновляем метрики
        detector.update(
            volume_current=100_000,
            volume_peak=500_000,
            walls_against_direction=2,
            delta_current=-500,
            speed_current=0.5,
            speed_peak=2.0,
        )
        
        # Проверяем исчерпание
        if detector.is_exhausted():
            print("Импульс исчерпан, выходим из позиции")
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[ExhaustionConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.config = config or ExhaustionConfig()
        
        # Текущие метрики
        self._metrics = ExhaustionMetrics(
            ts_ns=time.time_ns(),
            symbol=symbol,
        )
        
        # История для расчёта пиков
        self._volume_history: Deque[float] = deque(maxlen=1000)
        self._speed_history: Deque[float] = deque(maxlen=1000)
    
    def update(
        self,
        volume_current: float = 0.0,
        walls_against_direction: int = 0,
        wall_notional_against: float = 0.0,
        delta_current: float = 0.0,
        speed_current: float = 0.0,
    ) -> ExhaustionMetrics:
        """
        Обновляет метрики исчерпания.
        
        Вызывается на каждом обновлении данных (например, каждую секунду).
        
        Возвращает обновлённые метрики.
        """
        now_ns = time.time_ns()
        
        # Обновляем историю
        self._volume_history.append(volume_current)
        self._speed_history.append(speed_current)
        
        # Рассчитываем пики
        volume_peak = max(self._volume_history) if self._volume_history else 0.0
        speed_peak = max(self._speed_history) if self._speed_history else 0.0
        
        # Рассчитываем угасание объёма
        volume_decay_ratio = 0.0
        if volume_peak > 0:
            volume_decay_ratio = 1.0 - (volume_current / volume_peak)
        
        # Рассчитываем замедление движения
        speed_decay_ratio = 0.0
        if speed_peak > 0:
            speed_decay_ratio = 1.0 - (speed_current / speed_peak)
        
        # Проверяем разворот дельты
        # Для простоты: если дельта отрицательная для лонга или положительная для шорта
        # считаем, что дельта развернулась
        # В реальной реализации нужно знать направление позиции
        delta_reversed = False  # будет установлено извне
        
        # Обновляем метрики
        self._metrics.ts_ns = now_ns
        self._metrics.volume_current = volume_current
        self._metrics.volume_peak = volume_peak
        self._metrics.volume_decay_ratio = volume_decay_ratio
        self._metrics.walls_against_direction = walls_against_direction
        self._metrics.wall_notional_against = wall_notional_against
        self._metrics.delta_current = delta_current
        self._metrics.delta_reversed = delta_reversed
        self._metrics.speed_current = speed_current
        self._metrics.speed_peak = speed_peak
        self._metrics.speed_decay_ratio = speed_decay_ratio
        
        # Рассчитываем итоговый скор исчерпания
        exhaustion_score = self._calculate_exhaustion_score()
        self._metrics.exhaustion_score = exhaustion_score
        self._metrics.is_exhausted = exhaustion_score >= self.config.exhaustion_score_threshold
        
        return self._metrics
    
    def set_delta_reversed(self, reversed_flag: bool) -> None:
        """
        Устанавливает флаг разворота дельты.
        
        Вызывается извне, когда известно направление позиции.
        """
        self._metrics.delta_reversed = reversed_flag
    
    def is_exhausted(self) -> bool:
        """Проверяет, исчерпан ли импульс."""
        return self._metrics.is_exhausted
    
    def get_metrics(self) -> ExhaustionMetrics:
        """Возвращает текущие метрики."""
        return self._metrics
    
    def get_exhaustion_score(self) -> float:
        """Возвращает скор исчерпания [0, 1]."""
        return self._metrics.exhaustion_score
    
    def reset(self) -> None:
        """Сбрасывает состояние детектора."""
        self._volume_history.clear()
        self._speed_history.clear()
        self._metrics = ExhaustionMetrics(
            ts_ns=time.time_ns(),
            symbol=self.symbol,
        )
    
    def _calculate_exhaustion_score(self) -> float:
        """
        Рассчитывает итоговый скор исчерпания.
        
        Скор [0, 1], где:
        - 0 = импульс не исчерпан
        - 1 = импульс полностью исчерпан
        
        Формула:
            скор = 0.3 × угасание_объёма
                 + 0.2 × появление_стен
                 + 0.2 × разворот_дельты
                 + 0.3 × замедление_движения
        """
        # Угасание объёма
        volume_score = min(1.0, self._metrics.volume_decay_ratio / 0.5)
        
        # Появление стен против движения
        wall_score = min(1.0, self._metrics.walls_against_direction / 3.0)
        
        # Разворот дельты
        delta_score = 1.0 if self._metrics.delta_reversed else 0.0
        
        # Замедление движения
        speed_score = min(1.0, self._metrics.speed_decay_ratio / 0.7)
        
        # Веса
        w_volume = 0.3
        w_walls = 0.2
        w_delta = 0.2
        w_speed = 0.3
        
        score = (
            w_volume * volume_score
            + w_walls * wall_score
            + w_delta * delta_score
            + w_speed * speed_score
        )
        
        return max(0.0, min(1.0, score))


class ImpulseExhaustionManager:
    """
    Менеджер детекторов исчерпания импульса для множества символов.
    
    Использование:
        manager = ImpulseExhaustionManager()
        
        # Создаём детектор для символа
        detector = manager.get_or_create("BTCUSDT")
        
        # Обновляем метрики
        manager.update("BTCUSDT", volume_current=100_000)
        
        # Проверяем исчерпание
        if manager.is_exhausted("BTCUSDT"):
            print("Импульс исчерпан, выходим из позиции")
    """
    
    def __init__(self):
        self._detectors: Dict[str, ImpulseExhaustionDetector] = {}
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[ExhaustionConfig] = None,
    ) -> ImpulseExhaustionDetector:
        """Возвращает детектор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._detectors:
            self._detectors[symbol] = ImpulseExhaustionDetector(
                symbol=symbol,
                config=config,
            )
        
        return self._detectors[symbol]
    
    def update(
        self,
        symbol: str,
        volume_current: float = 0.0,
        walls_against_direction: int = 0,
        wall_notional_against: float = 0.0,
        delta_current: float = 0.0,
        speed_current: float = 0.0,
    ) -> ExhaustionMetrics:
        """Обновляет метрики для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return ExhaustionMetrics(ts_ns=time.time_ns(), symbol=symbol)
        
        return detector.update(
            volume_current=volume_current,
            walls_against_direction=walls_against_direction,
            wall_notional_against=wall_notional_against,
            delta_current=delta_current,
            speed_current=speed_current,
        )
    
    def is_exhausted(self, symbol: str) -> bool:
        """Проверяет исчерпание импульса для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return False
        return detector.is_exhausted()
    
    def get_exhaustion_score(self, symbol: str) -> float:
        """Возвращает скор исчерпания для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return 0.0
        return detector.get_exhaustion_score()
    
    def reset_all(self) -> None:
        """Сбрасывает все детекторы."""
        for detector in self._detectors.values():
            detector.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._detectors.keys())