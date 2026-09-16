"""
Расчёт дисбаланса стакана (bid/ask imbalance).

Дисбаланс — это перекос между объёмами заявок на покупку и продажу
в заданной зоне вокруг текущей цены. Используется для:
- Подтверждения направления пробоя (дисбаланс в сторону пробоя)
- Фильтрации ложных пробоев (дисбаланс против пробоя)
- Оценки давления покупателей/продавцов

Формула простого дисбаланса:
    imbalance = (bid_notional - ask_notional) / (bid_notional + ask_notional)

Диапазон: [-1, +1]
    +1.0 = только биды (максимальное давление вверх)
    -1.0 = только аски (максимальное давление вниз)
     0.0 = идеальный баланс

Взвешенный дисбаланс учитывает близость к текущей цене:
заявки ближе к цене имеют больший вес, так как они более значимы
для немедленного исполнения.

Используется в связке с:
- BookAnalyzer (агрегирует метрики дисбаланса)
- BreakoutDetector (подтверждение направления пробоя)
- SignalGenerator (фильтр min_imbalance)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ImbalanceConfig:
    """
    Конфигурация расчёта дисбаланса.
    
    Все параметры подобраны как разумные значения по умолчанию
    согласно формализации системы.
    """
    # Зона расчёта (в тиках от текущей цены)
    band_ticks: int = 15
    
    # Взвешивание по близости к цене
    weighting_enabled: bool = True
    weight_decay: float = 0.9  # коэффициент затухания с расстоянием
    
    # Пороги значимости
    min_imbalance: float = 0.25       # минимальный значимый дисбаланс
    strong_imbalance: float = 0.5     # сильный дисбаланс
    
    # Максимальное количество уровней для расчёта
    max_levels_per_side: int = 50


@dataclass
class ImbalanceMetrics:
    """
    Метрики дисбаланса стакана в конкретный момент времени.
    
    Содержит как простой, так и взвешенный дисбаланс,
    а также детализацию по сторонам.
    """
    ts_ns: int
    symbol: str
    
    # Простой дисбаланс (без взвешивания)
    simple_imbalance: float = 0.0
    
    # Взвешенный дисбаланс (с учётом близости к цене)
    weighted_imbalance: float = 0.0
    
    # Итоговый дисбаланс (взвешенный, если включено)
    imbalance: float = 0.0
    
    # Детализация по сторонам (простой)
    bid_notional: float = 0.0
    ask_notional: float = 0.0
    
    # Детализация по сторонам (взвешенный)
    weighted_bid_notional: float = 0.0
    weighted_ask_notional: float = 0.0
    
    # Количество уровней в расчёте
    bid_levels_count: int = 0
    ask_levels_count: int = 0
    
    # Зона расчёта (для диагностики)
    band_lower: float = 0.0
    band_upper: float = 0.0
    
    @property
    def is_bid_dominant(self) -> bool:
        """Биды доминируют (давление вверх)."""
        return self.imbalance > 0
    
    @property
    def is_ask_dominant(self) -> bool:
        """Аски доминируют (давление вниз)."""
        return self.imbalance < 0
    
    @property
    def is_balanced(self) -> bool:
        """Стакан сбалансирован."""
        return abs(self.imbalance) < 0.1
    
    @property
    def is_significant(self) -> bool:
        """Дисбаланс значимый (выше порога)."""
        return abs(self.imbalance) >= 0.25
    
    @property
    def is_strong(self) -> bool:
        """Дисбаланс сильный."""
        return abs(self.imbalance) >= 0.5
    
    def supports_direction(self, is_buy: bool) -> bool:
        """
        Проверяет, поддерживает ли дисбаланс указанное направление.
        
        Для лонга (is_buy=True): дисбаланс должен быть положительным.
        Для шорта (is_buy=False): дисбаланс должен быть отрицательным.
        """
        if is_buy:
            return self.imbalance > 0
        return self.imbalance < 0
    
    def to_dict(self) -> dict:
        """Преобразует в словарь для журналирования."""
        return {
            "ts_ns": self.ts_ns,
            "symbol": self.symbol,
            "imbalance": round(self.imbalance, 4),
            "simple_imbalance": round(self.simple_imbalance, 4),
            "weighted_imbalance": round(self.weighted_imbalance, 4),
            "bid_notional": round(self.bid_notional, 2),
            "ask_notional": round(self.ask_notional, 2),
            "weighted_bid_notional": round(self.weighted_bid_notional, 2),
            "weighted_ask_notional": round(self.weighted_ask_notional, 2),
            "bid_levels_count": self.bid_levels_count,
            "ask_levels_count": self.ask_levels_count,
        }


class ImbalanceCalculator:
    """
    Калькулятор дисбаланса стакана.
    
    Поддерживает два режима:
    1. Простой дисбаланс: сумма номиналов бидов vs асков
    2. Взвешенный дисбаланс: с учётом близости к текущей цене
    
    Использование:
        calculator = ImbalanceCalculator(config)
        
        # Получаем срезы стакана (из FastOrderBook)
        bids = book.top_bids(50)
        asks = book.top_asks(50)
        mid_price = book.mid_price
        
        # Рассчитываем дисбаланс
        metrics = calculator.calculate(
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            tick_size=0.1,
            symbol="BTCUSDT",
        )
        
        if metrics.imbalance > 0.25:
            print("Дисбаланс в сторону лонга")
    """
    
    def __init__(self, config: Optional[ImbalanceConfig] = None):
        self.config = config or ImbalanceConfig()
    
    def calculate(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        mid_price: float,
        tick_size: float,
        symbol: str = "",
        ts_ns: Optional[int] = None,
    ) -> ImbalanceMetrics:
        """
        Рассчитывает дисбаланс стакана.
        
        Args:
            bids: список (цена, количество) для бидов, от лучших к худшим
            asks: список (цена, количество) для асков, от лучших к худшим
            mid_price: средняя цена (между лучшим бидом и аском)
            tick_size: размер тика инструмента
            symbol: символ инструмента
            ts_ns: временная метка (по умолчанию текущее время)
            
        Returns:
            ImbalanceMetrics с рассчитанными значениями
        """
        if ts_ns is None:
            ts_ns = time.time_ns()
        
        if mid_price <= 0 or tick_size <= 0:
            return ImbalanceMetrics(
                ts_ns=ts_ns,
                symbol=symbol,
            )
        
        # Определяем зону расчёта
        band_distance = self.config.band_ticks * tick_size
        band_lower = mid_price - band_distance
        band_upper = mid_price + band_distance
        
        # Фильтруем уровни в зоне
        bids_in_band = [
            (price, qty) for price, qty in bids
            if band_lower <= price <= mid_price and qty > 0
        ][:self.config.max_levels_per_side]
        
        asks_in_band = [
            (price, qty) for price, qty in asks
            if mid_price <= price <= band_upper and qty > 0
        ][:self.config.max_levels_per_side]
        
        # Рассчитываем простой дисбаланс
        bid_notional = sum(price * qty for price, qty in bids_in_band)
        ask_notional = sum(price * qty for price, qty in asks_in_band)
        
        simple_imbalance = self._calculate_imbalance(
            bid_notional, ask_notional
        )
        
        # Рассчитываем взвешенный дисбаланс
        if self.config.weighting_enabled:
            weighted_bid_notional = self._calculate_weighted_notional(
                levels=bids_in_band,
                mid_price=mid_price,
                tick_size=tick_size,
                is_bid_side=True,
            )
            weighted_ask_notional = self._calculate_weighted_notional(
                levels=asks_in_band,
                mid_price=mid_price,
                tick_size=tick_size,
                is_bid_side=False,
            )
            weighted_imbalance = self._calculate_imbalance(
                weighted_bid_notional, weighted_ask_notional
            )
        else:
            weighted_bid_notional = bid_notional
            weighted_ask_notional = ask_notional
            weighted_imbalance = simple_imbalance
        
        # Итоговый дисбаланс
        imbalance = weighted_imbalance if self.config.weighting_enabled else simple_imbalance
        
        return ImbalanceMetrics(
            ts_ns=ts_ns,
            symbol=symbol,
            simple_imbalance=simple_imbalance,
            weighted_imbalance=weighted_imbalance,
            imbalance=imbalance,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
            weighted_bid_notional=weighted_bid_notional,
            weighted_ask_notional=weighted_ask_notional,
            bid_levels_count=len(bids_in_band),
            ask_levels_count=len(asks_in_band),
            band_lower=band_lower,
            band_upper=band_upper,
        )
    
    def calculate_around_level(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        level_price: float,
        tick_size: float,
        symbol: str = "",
        ts_ns: Optional[int] = None,
    ) -> ImbalanceMetrics:
        """
        Рассчитывает дисбаланс вокруг конкретного уровня.
        
        Используется для оценки давления в зоне уровня
        при подходе цены к нему.
        
        В отличие от calculate(), центром зоны является цена уровня,
        а не текущая средняя цена.
        """
        # Используем цену уровня как центр
        return self.calculate(
            bids=bids,
            asks=asks,
            mid_price=level_price,
            tick_size=tick_size,
            symbol=symbol,
            ts_ns=ts_ns,
        )
    
    def _calculate_imbalance(
        self,
        bid_notional: float,
        ask_notional: float,
    ) -> float:
        """
        Рассчитывает дисбаланс из номиналов.
        
        Формула:
            imbalance = (bid - ask) / (bid + ask)
        
        Возвращает значение в диапазоне [-1, +1].
        """
        total = bid_notional + ask_notional
        if total <= 0:
            return 0.0
        return (bid_notional - ask_notional) / total
    
    def _calculate_weighted_notional(
        self,
        levels: List[Tuple[float, float]],
        mid_price: float,
        tick_size: float,
        is_bid_side: bool,
    ) -> float:
        """
        Рассчитывает взвешенный номинал уровней.
        
        Вес уровня зависит от его близости к средней цене:
            weight = weight_decay ^ distance_ticks
        
        Уровни ближе к цене имеют больший вес.
        """
        weighted_total = 0.0
        
        for price, qty in levels:
            # Расстояние в тиках от средней цены
            distance_ticks = abs(price - mid_price) / tick_size
            
            # Вес затухает с расстоянием
            weight = self.config.weight_decay ** distance_ticks
            
            # Взвешенный номинал
            notional = price * qty
            weighted_total += notional * weight
        
        return weighted_total


class ImbalanceTracker:
    """
    Трекер дисбаланса во времени для одного символа.
    
    Хранит историю значений дисбаланса и предоставляет:
    - Текущее значение
    - Среднее за окно
    - Тренд (растёт/падает)
    - Экстремумы
    
    Использование:
        tracker = ImbalanceTracker(config)
        
        # При каждом обновлении стакана
        tracker.update(metrics)
        
        # Получаем тренд
        if tracker.is_trending_up():
            print("Дисбаланс усиливается в сторону лонга")
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[ImbalanceConfig] = None,
        history_size: int = 100,
    ):
        self.symbol = symbol.upper()
        self.config = config or ImbalanceConfig()
        
        # История значений дисбаланса
        self._history: List[ImbalanceMetrics] = []
        self._history_size = history_size
        
        # Текущее значение
        self._current: Optional[ImbalanceMetrics] = None
    
    def update(self, metrics: ImbalanceMetrics) -> None:
        """
        Обновляет текущее значение дисбаланса.
        
        Вызывается при каждом расчёте дисбаланса.
        """
        self._current = metrics
        self._history.append(metrics)
        
        # Ограничиваем размер истории
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]
    
    def get_current(self) -> Optional[ImbalanceMetrics]:
        """Возвращает текущие метрики дисбаланса."""
        return self._current
    
    def get_current_value(self) -> float:
        """Возвращает текущее значение дисбаланса."""
        if self._current is None:
            return 0.0
        return self._current.imbalance
    
    def get_average(self, window: int = 10) -> float:
        """
        Возвращает среднее значение дисбаланса за последние N измерений.
        
        Используется для сглаживания шума.
        """
        if not self._history:
            return 0.0
        
        recent = self._history[-window:]
        return sum(m.imbalance for m in recent) / len(recent)
    
    def is_trending_up(self, window: int = 5) -> bool:
        """
        Проверяет, усиливается ли дисбаланс в сторону лонга.
        
        Возвращает True, если среднее за последние window измерений
        больше среднего за предыдущие window измерений.
        """
        if len(self._history) < window * 2:
            return False
        
        recent = self._history[-window:]
        previous = self._history[-window * 2:-window]
        
        recent_avg = sum(m.imbalance for m in recent) / len(recent)
        previous_avg = sum(m.imbalance for m in previous) / len(previous)
        
        return recent_avg > previous_avg
    
    def is_trending_down(self, window: int = 5) -> bool:
        """
        Проверяет, усиливается ли дисбаланс в сторону шорта.
        """
        if len(self._history) < window * 2:
            return False
        
        recent = self._history[-window:]
        previous = self._history[-window * 2:-window]
        
        recent_avg = sum(m.imbalance for m in recent) / len(recent)
        previous_avg = sum(m.imbalance for m in previous) / len(previous)
        
        return recent_avg < previous_avg
    
    def get_max(self, window: int = 20) -> float:
        """Возвращает максимальный дисбаланс за последние N измерений."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return max(m.imbalance for m in recent)
    
    def get_min(self, window: int = 20) -> float:
        """Возвращает минимальный дисбаланс за последние N измерений."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return min(m.imbalance for m in recent)
    
    def reset(self) -> None:
        """Сбрасывает историю."""
        self._history.clear()
        self._current = None


class ImbalanceManager:
    """
    Менеджер калькуляторов и трекеров дисбаланса для множества символов.
    
    Использование:
        manager = ImbalanceManager(tick_sizes={"BTCUSDT": 0.1})
        
        # Рассчитываем дисбаланс для символа
        metrics = manager.calculate(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            mid_price=76000.0,
        )
        
        # Получаем трекер для анализа тренда
        tracker = manager.get_tracker("BTCUSDT")
        if tracker.is_trending_up():
            print("Давление покупателей усиливается")
    """
    
    def __init__(self, tick_sizes: dict):
        self._tick_sizes = {k.upper(): v for k, v in tick_sizes.items()}
        self._calculator = ImbalanceCalculator()
        self._trackers: dict = {}
    
    def calculate(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        mid_price: float,
    ) -> Optional[ImbalanceMetrics]:
        """
        Рассчитывает дисбаланс для символа.
        
        Автоматически обновляет трекер для анализа тренда.
        """
        symbol = symbol.upper()
        tick_size = self._tick_sizes.get(symbol)
        
        if tick_size is None:
            return None
        
        metrics = self._calculator.calculate(
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            tick_size=tick_size,
            symbol=symbol,
        )
        
        # Обновляем трекер
        tracker = self.get_tracker(symbol)
        tracker.update(metrics)
        
        return metrics
    
    def calculate_around_level(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        level_price: float,
    ) -> Optional[ImbalanceMetrics]:
        """
        Рассчитывает дисбаланс вокруг уровня для символа.
        
        Используется при подходе цены к уровню.
        """
        symbol = symbol.upper()
        tick_size = self._tick_sizes.get(symbol)
        
        if tick_size is None:
            return None
        
        return self._calculator.calculate_around_level(
            bids=bids,
            asks=asks,
            level_price=level_price,
            tick_size=tick_size,
            symbol=symbol,
        )
    
    def get_tracker(self, symbol: str) -> ImbalanceTracker:
        """Возвращает трекер для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._trackers:
            self._trackers[symbol] = ImbalanceTracker(symbol=symbol)
        
        return self._trackers[symbol]
    
    def get_current(self, symbol: str) -> Optional[ImbalanceMetrics]:
        """Возвращает текущие метрики для символа."""
        tracker = self._trackers.get(symbol.upper())
        if tracker is None:
            return None
        return tracker.get_current()
    
    def get_current_value(self, symbol: str) -> float:
        """Возвращает текущее значение дисбаланса для символа."""
        tracker = self._trackers.get(symbol.upper())
        if tracker is None:
            return 0.0
        return tracker.get_current_value()
    
    def set_tick_size(self, symbol: str, tick_size: float) -> None:
        """Устанавливает тик-сайз для символа."""
        self._tick_sizes[symbol.upper()] = tick_size
    
    def reset_all(self) -> None:
        """Сбрасывает все трекеры."""
        for tracker in self._trackers.values():
            tracker.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._trackers.keys())