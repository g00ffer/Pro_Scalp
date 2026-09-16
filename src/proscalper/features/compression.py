"""
Детектор консолидации/сжатия цены у уровня.

Консолидация - это обязательное условие для пробоя уровня.
Без консолидации пробой будет случайным.

Метрики консолидации:
- Сжатие диапазона (чем меньше, тем сильнее сжатие)
- Снижение объёма (признак накопления)
- Время пребывания цены у уровня
- Расстояние до ближайшего уровня

Используется в связке с:
- BookAnalyzer (плотности в стакане)
- ApproachDetector (подготовка к пробою)
- BreakoutDetector (сигнал пробоя)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from proscalper.market_data.bar_aggregator import Bar
from proscalper.features.levels import Level


@dataclass
class CompressionConfig:
    """
    Конфигурация детектора консолидации.
    
    Все пороги подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Параметры сжатия
    compression_threshold: float = 0.5      # range_10s / range_60s < 0.5 → сжатие
    min_consolidation_sec: float = 30.0     # минимальное время консолидации
    
    # Параметры объёма
    volume_low_threshold: float = 0.7       # volume < 0.7 × median → низкий объём
    volume_median_window_sec: int = 3600    # окно для медианы объёма (1 час)
    
    # Параметры близости к уровню
    near_level_ticks: int = 10              # зона "у уровня" в тиках
    min_time_near_level_sec: float = 15.0   # минимальное время у уровня
    
    # Параметры окон для расчёта диапазонов
    window_short_sec: int = 10              # короткое окно (быстрое сжатие)
    window_medium_sec: int = 30             # среднее окно
    window_long_sec: int = 60               # длинное окно (базовый диапазон)
    
    # Параметры хранения
    max_bars: int = 1000                    # максимум баров в истории
    max_price_updates: int = 10000          # максимум обновлений цены


@dataclass
class CompressionMetrics:
    """
    Снимок метрик консолидации в конкретный момент времени.
    
    Используется для:
    - Определения готовности уровня к пробою
    - Фильтрации ложных пробоев
    - Логирования в DecisionJournal
    """
    ts_ns: int
    symbol: str
    
    # Диапазоны за разные окна (в цене)
    range_short: float = 0.0        # диапазон за короткое окно (10с)
    range_medium: float = 0.0       # диапазон за среднее окно (30с)
    range_long: float = 0.0         # диапазон за длинное окно (60с)
    
    # Сжатие (чем меньше, тем сильнее сжатие)
    compression_ratio: float = 1.0  # range_short / range_long
    
    # Объёмы (в номинале)
    volume_short: float = 0.0       # объём за короткое окно
    volume_medium: float = 0.0      # объём за среднее окно
    volume_median: float = 0.0      # медиана объёма за час
    volume_ratio: float = 1.0       # volume_short / volume_median
    
    # Цена относительно уровня
    distance_to_level_ticks: float = 0.0    # расстояние до уровня в тиках
    time_near_level_sec: float = 0.0        # время у уровня в секундах
    
    # Производные флаги
    is_compressing: bool = False    # идёт ли сжатие
    is_low_volume: bool = False     # объём ниже медианы
    is_near_level: bool = False     # цена у уровня
    is_consolidating: bool = False  # полная консолидация (все условия)
    
    @property
    def compression_strength(self) -> float:
        """
        Сила сжатия (0 = нет сжатия, 1 = максимальное сжатие).
        
        Используется для ранжирования уровней по готовности к пробою.
        """
        if self.range_long <= 0:
            return 0.0
        return max(0.0, 1.0 - self.compression_ratio)
    
    @property
    def volume_depletion(self) -> float:
        """
        Степень снижения объёма (0 = нет снижения, 1 = объём на нуле).
        """
        if self.volume_median <= 0:
            return 0.0
        return max(0.0, 1.0 - self.volume_ratio)


class _PricePoint:
    """Внутренняя структура для точки цены."""
    __slots__ = ('ts_ns', 'price')
    
    def __init__(self, ts_ns: int, price: float):
        self.ts_ns = ts_ns
        self.price = price


class _BarPoint:
    """Внутренняя структура для бара."""
    __slots__ = ('ts_ns', 'high', 'low', 'notional')
    
    def __init__(self, ts_ns: int, high: float, low: float, notional: float):
        self.ts_ns = ts_ns
        self.high = high
        self.low = low
        self.notional = notional


class CompressionDetector:
    """
    Детектор консолидации для одного символа.
    
    Работает на двух источниках данных:
    1. Бары (5с) - для расчёта диапазонов и объёмов
    2. Тики цены - для отслеживания времени у уровня
    
    Использование:
        detector = CompressionDetector("BTCUSDT", tick_size=0.1)
        
        # На каждом баре
        detector.on_bar(bar)
        
        # На каждом тике цены
        detector.on_price_update(price, ts_ns)
        
        # Периодически получаем метрики
        metrics = detector.snapshot(active_levels)
        
        if detector.is_ready_for_breakout(metrics, level):
            print(f"Уровень {level.center} готов к пробою!")
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[CompressionConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or CompressionConfig()
        
        # История баров для расчёта диапазонов и объёмов
        self._bars: Deque[_BarPoint] = deque(maxlen=self.config.max_bars)
        
        # История цен для отслеживания времени у уровня
        self._prices: Deque[_PricePoint] = deque(maxlen=self.config.max_price_updates)
        
        # Текущая цена
        self._last_price: float = 0.0
        self._last_price_ts_ns: int = 0
        
        # Отслеживание времени у уровня
        self._level_zone_start_ts_ns: int = 0
        self._is_in_level_zone: bool = False
        self._current_level_center: float = 0.0
        
        # Кэш для медианы объёма
        self._volume_median_cache: float = 0.0
        self._volume_median_last_update_ns: int = 0
    
    def on_bar(self, bar: Bar) -> None:
        """
        Обработка нового бара.
        
        Вызывается при закрытии каждого бара (обычно каждые 5 секунд).
        """
        # Игнорируем невалидные бары
        if bar.high <= 0 or bar.low <= 0:
            return
        
        point = _BarPoint(
            ts_ns=bar.ts_ns,
            high=bar.high,
            low=bar.low,
            notional=bar.notional,
        )
        self._bars.append(point)
    
    def on_price_update(self, price: float, ts_ns: int) -> None:
        """
        Обработка обновления цены.
        
        Вызывается на каждом тике цены (из bookTicker или последней сделки).
        Используется для отслеживания времени у уровня.
        """
        if price <= 0:
            return
        
        self._last_price = price
        self._last_price_ts_ns = ts_ns
        
        point = _PricePoint(ts_ns=ts_ns, price=price)
        self._prices.append(point)
    
    def snapshot(self, levels: List[Level]) -> CompressionMetrics:
        """
        Возвращает снимок метрик консолидации.
        
        Args:
            levels: Список активных уровней для проверки близости
            
        Returns:
            Метрики консолидации для текущего момента времени
        """
        now_ns = self._last_price_ts_ns or time.time_ns()
        
        # Рассчитываем диапазоны за разные окна
        range_short = self._calculate_range(self.config.window_short_sec, now_ns)
        range_medium = self._calculate_range(self.config.window_medium_sec, now_ns)
        range_long = self._calculate_range(self.config.window_long_sec, now_ns)
        
        # Рассчитываем коэффициент сжатия
        compression_ratio = (
            range_short / range_long if range_long > 0 else 1.0
        )
        
        # Рассчитываем объёмы
        volume_short = self._calculate_volume(self.config.window_short_sec, now_ns)
        volume_medium = self._calculate_volume(self.config.window_medium_sec, now_ns)
        volume_median = self._calculate_volume_median(now_ns)
        volume_ratio = (
            volume_short / volume_median if volume_median > 0 else 1.0
        )
        
        # Находим ближайший уровень
        distance_to_level_ticks = 0.0
        nearest_level_center = 0.0
        
        if levels and self._last_price > 0:
            min_distance = float('inf')
            
            for level in levels:
                distance = level.distance_to(self._last_price)
                if distance < min_distance:
                    min_distance = distance
                    nearest_level_center = level.center
            
            if min_distance < float('inf'):
                distance_to_level_ticks = min_distance / self.tick_size
        
        # Рассчитываем время у уровня
        time_near_level_sec = self._calculate_time_near_level(
            nearest_level_center, now_ns
        )
        
        # Определяем флаги
        is_compressing = compression_ratio < self.config.compression_threshold
        is_low_volume = volume_ratio < self.config.volume_low_threshold
        is_near_level = distance_to_level_ticks <= self.config.near_level_ticks
        is_consolidating = (
            is_compressing
            and is_near_level
            and time_near_level_sec >= self.config.min_consolidation_sec
        )
        
        return CompressionMetrics(
            ts_ns=now_ns,
            symbol=self.symbol,
            range_short=range_short,
            range_medium=range_medium,
            range_long=range_long,
            compression_ratio=compression_ratio,
            volume_short=volume_short,
            volume_medium=volume_medium,
            volume_median=volume_median,
            volume_ratio=volume_ratio,
            distance_to_level_ticks=distance_to_level_ticks,
            time_near_level_sec=time_near_level_sec,
            is_compressing=is_compressing,
            is_low_volume=is_low_volume,
            is_near_level=is_near_level,
            is_consolidating=is_consolidating,
        )
    
    def is_ready_for_breakout(
        self,
        metrics: CompressionMetrics,
        level: Level,
    ) -> bool:
        """
        Проверяет, готов ли уровень к пробою.
        
        Условия готовности:
        1. Консолидация у уровня (обязательно)
        2. Цена находится в зоне уровня
        3. Достаточное время консолидации
        
        Плотности в стакане проверяются отдельно в ApproachDetector.
        
        Args:
            metrics: Метрики консолидации
            level: Уровень для проверки
            
        Returns:
            True если уровень готов к пробою
        """
        # Проверяем консолидацию
        if not metrics.is_compressing:
            return False
        
        # Проверяем близость к уровню
        distance_to_level = level.distance_to(self._last_price)
        if distance_to_level > self.config.near_level_ticks * self.tick_size:
            return False
        
        # Проверяем время у уровня
        if metrics.time_near_level_sec < self.config.min_consolidation_sec:
            return False
        
        # Все условия выполнены
        return True
    
    def reset(self) -> None:
        """Сбрасывает состояние детектора."""
        self._bars.clear()
        self._prices.clear()
        self._last_price = 0.0
        self._last_price_ts_ns = 0
        self._level_zone_start_ts_ns = 0
        self._is_in_level_zone = False
        self._current_level_center = 0.0
        self._volume_median_cache = 0.0
        self._volume_median_last_update_ns = 0
    
    def _calculate_range(self, window_sec: int, now_ns: int) -> float:
        """
        Рассчитывает диапазон цен за указанное окно.
        
        Диапазон = max(баров за окно) - min(баров за окно)
        """
        if not self._bars:
            return 0.0
        
        cutoff_ns = now_ns - window_sec * 1_000_000_000
        
        max_price = 0.0
        min_price = float('inf')
        
        for bar in self._bars:
            if bar.ts_ns >= cutoff_ns:
                max_price = max(max_price, bar.high)
                min_price = min(min_price, bar.low)
        
        if min_price == float('inf'):
            return 0.0
        
        return max_price - min_price
    
    def _calculate_volume(self, window_sec: int, now_ns: int) -> float:
        """
        Рассчитывает суммарный объём за указанное окно.
        """
        if not self._bars:
            return 0.0
        
        cutoff_ns = now_ns - window_sec * 1_000_000_000
        
        total_volume = 0.0
        
        for bar in self._bars:
            if bar.ts_ns >= cutoff_ns:
                total_volume += bar.notional
        
        return total_volume
    
    def _calculate_volume_median(self, now_ns: int) -> float:
        """
        Рассчитывает медиану объёма за длинное окно (1 час).
        
        Медиана более устойчива к выбросам, чем среднее.
        Кэшируется на 10 секунд для экономии ресурсов.
        """
        # Проверяем кэш
        cache_ttl_ns = 10_000_000_000  # 10 секунд
        if (now_ns - self._volume_median_last_update_ns) < cache_ttl_ns:
            return self._volume_median_cache
        
        if not self._bars:
            return 0.0
        
        # Собираем объёмы баров за окно медианы
        window_ns = self.config.volume_median_window_sec * 1_000_000_000
        cutoff_ns = now_ns - window_ns
        
        volumes = []
        for bar in self._bars:
            if bar.ts_ns >= cutoff_ns and bar.notional > 0:
                volumes.append(bar.notional)
        
        if not volumes:
            self._volume_median_cache = 0.0
        else:
            volumes.sort()
            n = len(volumes)
            if n % 2 == 0:
                self._volume_median_cache = (volumes[n // 2 - 1] + volumes[n // 2]) / 2
            else:
                self._volume_median_cache = volumes[n // 2]
        
        self._volume_median_last_update_ns = now_ns
        return self._volume_median_cache
    
    def _calculate_time_near_level(
        self,
        level_center: float,
        now_ns: int,
    ) -> float:
        """
        Рассчитывает время, которое цена провела у уровня.
        
        Использует историю цен для точного расчёта.
        """
        if not self._prices or level_center <= 0:
            return 0.0
        
        zone_threshold = self.config.near_level_ticks * self.tick_size
        
        # Проходим по истории цен и считаем время у уровня
        time_near_level_ns = 0
        prev_ts_ns = None
        
        for point in self._prices:
            distance = abs(point.price - level_center)
            is_near = distance <= zone_threshold
            
            if is_near and prev_ts_ns is not None:
                time_near_level_ns += point.ts_ns - prev_ts_ns
            
            prev_ts_ns = point.ts_ns if is_near else None
        
        return time_near_level_ns / 1_000_000_000


class CompressionManager:
    """
    Менеджер CompressionDetector'ов для множества символов.
    
    Использование:
        manager = CompressionManager(tick_sizes={"BTCUSDT": 0.1, "ETHUSDT": 0.01})
        
        # Получаем детектор для символа
        detector = manager.get_or_create("BTCUSDT")
        
        # Передаём бары
        manager.on_bar(bar)
        
        # Передаём тики цены
        manager.on_price_update(price, "BTCUSDT", ts_ns)
        
        # Получаем метрики
        metrics = manager.snapshot("BTCUSDT", active_levels)
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._detectors: Dict[str, CompressionDetector] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[CompressionConfig] = None,
    ) -> CompressionDetector:
        """Возвращает детектор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._detectors:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._detectors[symbol] = CompressionDetector(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        
        return self._detectors[symbol]
    
    def on_bar(self, bar: Bar) -> None:
        """Обработка нового бара для нужного символа."""
        detector = self._detectors.get(bar.symbol.upper())
        if detector is not None:
            detector.on_bar(bar)
    
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
    
    def snapshot(
        self,
        symbol: str,
        levels: List[Level],
    ) -> Optional[CompressionMetrics]:
        """Возвращает метрики консолидации для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return None
        return detector.snapshot(levels)
    
    def is_ready_for_breakout(
        self,
        symbol: str,
        metrics: CompressionMetrics,
        level: Level,
    ) -> bool:
        """Проверяет готовность уровня к пробою."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return False
        return detector.is_ready_for_breakout(metrics, level)
    
    def reset_all(self) -> None:
        """Сбрасывает все детекторы."""
        for detector in self._detectors.values():
            detector.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._detectors.keys())