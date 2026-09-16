"""
Анализатор стакана: плотности, дисбаланс, проедание стен.

Критичный модуль для системы пробоя уровней.
Подход к уровню определяется через плотности в стакане.

Основные концепции:
- Стена (Wall) — крупная лимитная заявка в стакане
- Проедание — уменьшение размера стены из-за исполнения маркет-ордерами
- Спуфинг — ложная стена, которая снимается при подходе цены
- Дисбаланс — перекос объёмов в зоне уровня

Используется в связке с:
- CompressionDetector (консолидация у уровня)
- ApproachDetector (подготовка к пробою)
- BreakoutDetector (сигнал пробоя)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from proscalper.core.events import (
    BookSnapshotEvent,
    BookDeltaBatchEvent,
    BookLevel,
)
from proscalper.core.types import BookSide
from proscalper.features.levels import Level


class WallState(Enum):
    """Состояние стены в стакане."""
    FORMING = auto()      # Стена формируется (накапливается)
    ACTIVE = auto()       # Стена активна и значима
    CONSUMING = auto()    # Стена проедается маркет-ордерами
    CONSUMED = auto()     # Стена проедена (пробой)
    CANCELLED = auto()    # Стена снята (спуфинг)
    INVALID = auto()      # Стена устарела или неактуальна


@dataclass
class Wall:
    """
    Крупная лимитная заявка в стакане.
    
    Стена - это не одна заявка, а агрегация объёма на ценовом уровне,
    который превышает порог значимости.
    """
    price: float
    side: BookSide
    
    # Размеры
    current_notional: float = 0.0
    max_notional: float = 0.0
    
    # Время жизни
    created_ts_ns: int = 0
    last_update_ts_ns: int = 0
    
    # Объёмы исполнения и снятия
    executed_notional: float = 0.0
    cancelled_notional: float = 0.0
    replenish_count: int = 0
    
    # Состояние
    state: WallState = WallState.FORMING
    
    # Оценки
    spoof_score: float = 0.0
    iceberg_score: float = 0.0
    
    # Связь с уровнем
    linked_level_id: Optional[str] = None
    
    @property
    def age_ms(self) -> int:
        """Возраст стены в миллисекундах."""
        if self.created_ts_ns == 0:
            return 0
        return int((time.time_ns() - self.created_ts_ns) / 1_000_000)
    
    @property
    def consumption_ratio(self) -> float:
        """
        Доля проедания стены (0 = не проедена, 1 = полностью проедена).
        
        Используется для определения момента пробоя.
        """
        if self.max_notional <= 0:
            return 0.0
        return self.executed_notional / self.max_notional
    
    @property
    def is_significant(self) -> bool:
        """Является ли стена значимой (не спуфинг)."""
        return self.state == WallState.ACTIVE and self.spoof_score < 0.5
    
    def distance_to(self, price: float) -> float:
        """Расстояние от цены до стены."""
        return abs(self.price - price)


@dataclass
class ImbalanceMetrics:
    """Метрики дисбаланса в зоне цены."""
    ts_ns: int
    center_price: float
    zone_ticks: int
    
    # Объёмы в зоне
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    
    # Дисбаланс [-1, 1]
    # +1 = сильный перекос в сторону покупки (давление вверх)
    # -1 = сильный перекос в сторону продажи (давление вниз)
    imbalance: float = 0.0
    
    # Взвешенный дисбаланс (по близости к цене)
    weighted_imbalance: float = 0.0
    
    # Количество стен в зоне
    bid_walls_count: int = 0
    ask_walls_count: int = 0
    
    @property
    def is_bid_heavy(self) -> bool:
        """Сильный перекос в сторону покупки."""
        return self.imbalance > 0.5
    
    @property
    def is_ask_heavy(self) -> bool:
        """Сильный перекос в сторону продажи."""
        return self.imbalance < -0.5


@dataclass
class LevelLiquidity:
    """
    Снимок ликвидности вокруг уровня.
    
    Используется для определения готовности уровня к пробою.
    """
    level_id: str
    level_center: float
    
    # Стены выше уровня (сопротивления)
    ask_walls: List[Wall] = field(default_factory=list)
    
    # Стены ниже уровня (поддержки)
    bid_walls: List[Wall] = field(default_factory=list)
    
    # Проедены ли стены
    ask_wall_consumed: bool = False
    bid_wall_consumed: bool = False
    
    # Оценки поддержки/давления
    support_score: float = 0.0      # сила поддержки снизу
    pressure_score: float = 0.0     # сила давления сверху
    
    # Дисбаланс в зоне уровня
    imbalance: float = 0.0
    
    # Стены, которые нужно пробить для пробоя уровня
    walls_to_break: List[Wall] = field(default_factory=list)
    
    @property
    def has_significant_walls(self) -> bool:
        """Есть ли значимые стены на уровне."""
        return len(self.walls_to_break) > 0
    
    @property
    def is_ready_for_breakout(self) -> bool:
        """Готов ли уровень к пробою (стены проедаются)."""
        if not self.walls_to_break:
            return False
        
        # Хотя бы одна стена активно проедается
        for wall in self.walls_to_break:
            if wall.state == WallState.CONSUMING:
                return True
        
        return False


@dataclass
class BookAnalyzerConfig:
    """Конфигурация анализатора стакана."""
    
    # Определение "крупной" стены
    min_wall_notional: float = 5000.0       # абсолютный минимум
    wall_median_multiplier: float = 3.0     # относительно медианы
    wall_zscore_threshold: float = 2.0      # z-score порог
    
    # Время жизни стены
    wall_min_age_ms: int = 5000             # минимум для значимости
    wall_max_age_ms: int = 300_000          # максимум (5 минут)
    
    # Проедание
    consumed_threshold: float = 0.7         # 70% исполнено = проедена
    consuming_rate_threshold: float = 0.1   # 10% за секунду = активное проедание
    
    # Спуфинг
    spoof_cancel_threshold: float = 0.5     # 50% снято = спуфинг
    spoof_min_age_ms: int = 3000            # короткоживущие = спуфинг
    
    # Зоны
    level_zone_ticks: int = 5               # зона уровня для связи стен
    imbalance_zone_ticks: int = 20          # зона для расчёта дисбаланса
    
    # Хранение
    max_walls: int = 100                    # максимум стен в реестре
    median_window_sec: int = 300            # окно для медианы (5 минут)


class WallRegistry:
    """
    Реестр всех активных стен в стакане.
    
    Отслеживает стены для обоих сторон (bid/ask) и управляет их жизненным циклом.
    """
    
    def __init__(self, config: BookAnalyzerConfig):
        self.config = config
        
        # Стены по цене и стороне
        # Ключ: (price, side)
        self._walls: Dict[Tuple[float, BookSide], Wall] = {}
        
        # История размеров для медианы и z-score
        self._notional_history: List[float] = []
        self._notional_history_ts: List[int] = []
    
    def add_or_update_wall(
        self,
        price: float,
        side: BookSide,
        notional: float,
        ts_ns: int,
    ) -> Optional[Wall]:
        """
        Добавляет или обновляет стену.
        
        Возвращает стену, если она стала значимой, иначе None.
        """
        key = (price, side)
        
        if notional <= 0:
            # Стена удалена
            if key in self._walls:
                wall = self._walls[key]
                wall.state = WallState.CANCELLED
                wall.last_update_ts_ns = ts_ns
                # Не удаляем сразу, помечаем как снятую
            return None
        
        if key in self._walls:
            # Обновляем существующую стену
            wall = self._walls[key]
            
            # Отслеживаем снятие (спуфинг)
            if notional < wall.current_notional:
                decrease = wall.current_notional - notional
                wall.cancelled_notional += decrease
            
            # Отслеживаем пополнение
            elif notional > wall.current_notional:
                wall.replenish_count += 1
            
            wall.current_notional = notional
            wall.max_notional = max(wall.max_notional, notional)
            wall.last_update_ts_ns = ts_ns
            
            # Обновляем состояние
            self._update_wall_state(wall, ts_ns)
            
            # Добавляем в историю
            self._add_to_history(notional, ts_ns)
            
            return wall if wall.state == WallState.ACTIVE else None
        
        else:
            # Создаём новую стену
            wall = Wall(
                price=price,
                side=side,
                current_notional=notional,
                max_notional=notional,
                created_ts_ns=ts_ns,
                last_update_ts_ns=ts_ns,
                state=WallState.FORMING,
            )
            
            self._walls[key] = wall
            
            # Добавляем в историю
            self._add_to_history(notional, ts_ns)
            
            # Проверяем, является ли стена значимой
            if self._is_significant_notional(notional):
                wall.state = WallState.ACTIVE
                return wall
            
            return None
    
    def get_walls_near_level(
        self,
        level: Level,
        zone_ticks: int,
        tick_size: float,
    ) -> List[Wall]:
        """
        Возвращает стены в зоне уровня.
        
        Зона = уровень ± zone_ticks тиков.
        """
        zone_distance = zone_ticks * tick_size
        result = []
        
        for wall in self._walls.values():
            if wall.state in (WallState.CANCELLED, WallState.INVALID):
                continue
            
            distance = abs(wall.price - level.center)
            if distance <= zone_distance:
                result.append(wall)
        
        return result
    
    def get_active_walls(self) -> List[Wall]:
        """Возвращает все активные стены."""
        return [
            wall for wall in self._walls.values()
            if wall.state in (WallState.ACTIVE, WallState.CONSUMING)
        ]
    
    def get_wall_at(
        self,
        price: float,
        side: BookSide,
    ) -> Optional[Wall]:
        """Возвращает стену на конкретной цене."""
        return self._walls.get((price, side))
    
    def prune_old_walls(self, now_ns: int) -> None:
        """Удаляет устаревшие стены."""
        max_age_ns = self.config.wall_max_age_ms * 1_000_000
        
        keys_to_remove = []
        for key, wall in self._walls.items():
            age_ns = now_ns - wall.created_ts_ns
            if age_ns > max_age_ns:
                wall.state = WallState.INVALID
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._walls[key]
        
        # Также чистим историю
        self._prune_history(now_ns)
    
    def mark_wall_consumed(self, price: float, side: BookSide) -> None:
        """Помечает стену как проеденную."""
        key = (price, side)
        if key in self._walls:
            self._walls[key].state = WallState.CONSUMED
    
    def reset(self) -> None:
        """Сбрасывает реестр."""
        self._walls.clear()
        self._notional_history.clear()
        self._notional_history_ts.clear()
    
    def _update_wall_state(self, wall: Wall, ts_ns: int) -> None:
        """Обновляет состояние стены."""
        # Проверяем спуфинг
        if self._is_spoofing(wall, ts_ns):
            wall.state = WallState.CANCELLED
            wall.spoof_score = 1.0
            return
        
        # Проверяем проедание
        if wall.max_notional > 0:
            consumption = wall.executed_notional / wall.max_notional
            
            if consumption >= self.config.consumed_threshold:
                wall.state = WallState.CONSUMED
            elif consumption > 0.1:
                wall.state = WallState.CONSUMING
            elif wall.state == WallState.FORMING:
                # Проверяем, достаточно ли стена живёт
                age_ms = (ts_ns - wall.created_ts_ns) / 1_000_000
                if age_ms >= self.config.wall_min_age_ms:
                    wall.state = WallState.ACTIVE
    
    def _is_spoofing(self, wall: Wall, ts_ns: int) -> bool:
        """
        Проверяет, является ли стена спуфингом.
        
        Критерии:
        1. Короткое время жизни + большое снятие
        2. Снятие при подходе цены (проверяется в BookAnalyzer)
        """
        age_ms = (ts_ns - wall.created_ts_ns) / 1_000_000
        
        # Короткоживущая стена с большим снятием
        if age_ms < self.config.spoof_min_age_ms:
            if wall.max_notional > 0:
                cancel_ratio = wall.cancelled_notional / wall.max_notional
                if cancel_ratio >= self.config.spoof_cancel_threshold:
                    return True
        
        return False
    
    def _is_significant_notional(self, notional: float) -> bool:
        """
        Проверяет, является ли размер значимым.
        
        Гибрид: абсолютный порог + относительный + z-score.
        """
        # Абсолютный порог
        if notional < self.config.min_wall_notional:
            return False
        
        # Относительный порог (медиана)
        median = self._calculate_median()
        if median > 0 and notional < median * self.config.wall_median_multiplier:
            return False
        
        # Z-score порог
        if len(self._notional_history) >= 10:
            zscore = self._calculate_zscore(notional)
            if zscore < self.config.wall_zscore_threshold:
                return False
        
        return True
    
    def _calculate_median(self) -> float:
        """Рассчитывает медиану размеров стен."""
        if not self._notional_history:
            return 0.0
        
        sorted_history = sorted(self._notional_history)
        n = len(sorted_history)
        
        if n % 2 == 0:
            return (sorted_history[n // 2 - 1] + sorted_history[n // 2]) / 2
        else:
            return sorted_history[n // 2]
    
    def _calculate_zscore(self, value: float) -> float:
        """Рассчитывает z-score для значения."""
        if len(self._notional_history) < 2:
            return 0.0
        
        mean = sum(self._notional_history) / len(self._notional_history)
        variance = sum((x - mean) ** 2 for x in self._notional_history) / len(self._notional_history)
        std = variance ** 0.5
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    def _add_to_history(self, notional: float, ts_ns: int) -> None:
        """Добавляет размер в историю."""
        self._notional_history.append(notional)
        self._notional_history_ts.append(ts_ns)
    
    def _prune_history(self, now_ns: int) -> None:
        """Удаляет старые записи из истории."""
        window_ns = self.config.median_window_sec * 1_000_000_000
        cutoff_ns = now_ns - window_ns
        
        while self._notional_history_ts and self._notional_history_ts[0] < cutoff_ns:
            self._notional_history_ts.pop(0)
            self._notional_history.pop(0)


class BookAnalyzer:
    """
    Основной анализатор стакана.
    
    Работает на событиях стакана (снапшоты и дельты) и предоставляет:
    - Реестр стен (плотностей)
    - Расчёт дисбаланса в зонах
    - Снимки ликвидности вокруг уровней
    - Детекцию проедания стен
    
    Использование:
        analyzer = BookAnalyzer("BTCUSDT", tick_size=0.1)
        
        # При получении снапшота
        analyzer.on_snapshot(snapshot)
        
        # При получении дельт
        analyzer.on_delta_batch(delta_batch)
        
        # Периодически получаем ликвидность вокруг уровня
        liquidity = analyzer.get_level_liquidity(level)
        
        if liquidity.is_ready_for_breakout:
            print("Уровень готов к пробою!")
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[BookAnalyzerConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or BookAnalyzerConfig()
        
        # Реестр стен
        self._wall_registry = WallRegistry(self.config)
        
        # Текущее состояние стакана (для расчёта дисбаланса)
        self._bids: Dict[float, float] = {}  # price -> quantity
        self._asks: Dict[float, float] = {}
        
        # Лучшая цена (для отслеживания подхода к стенам)
        self._last_bid_price: float = 0.0
        self._last_ask_price: float = 0.0
        self._last_update_ts_ns: int = 0
    
    def on_snapshot(self, snapshot: BookSnapshotEvent) -> None:
        """
        Обработка снапшота стакана.
        
        Полностью перестраивает состояние.
        """
        self._bids.clear()
        self._asks.clear()
        self._wall_registry.reset()
        
        ts_ns = snapshot.ts_exchange_ns or snapshot.ts_local_ns
        
        # Применяем bids
        for level in snapshot.bids:
            self._bids[level.price] = level.quantity
            notional = level.price * level.quantity
            self._wall_registry.add_or_update_wall(
                level.price, BookSide.BID, notional, ts_ns
            )
        
        # Применяем asks
        for level in snapshot.asks:
            self._asks[level.price] = level.quantity
            notional = level.price * level.quantity
            self._wall_registry.add_or_update_wall(
                level.price, BookSide.ASK, notional, ts_ns
            )
        
        # Обновляем лучшие цены
        if snapshot.bids:
            self._last_bid_price = max(l.price for l in snapshot.bids)
        if snapshot.asks:
            self._last_ask_price = min(l.price for l in snapshot.asks)
        
        self._last_update_ts_ns = ts_ns
    
    def on_delta_batch(self, delta_batch: BookDeltaBatchEvent) -> None:
        """
        Обработка пакета дельт стакана.
        
        Обновляет состояние и реестр стен.
        """
        ts_ns = delta_batch.ts_exchange_ns or delta_batch.ts_local_ns
        
        # Применяем изменения для bids
        for level in delta_batch.bids:
            self._apply_level_change(
                self._bids, level.price, level.quantity, BookSide.BID, ts_ns
            )
        
        # Применяем изменения для asks
        for level in delta_batch.asks:
            self._apply_level_change(
                self._asks, level.price, level.quantity, BookSide.ASK, ts_ns
            )
        
        # Чистим старые стены
        self._wall_registry.prune_old_walls(ts_ns)
        
        self._last_update_ts_ns = ts_ns
    
    def get_active_walls(self) -> List[Wall]:
        """Возвращает все активные стены."""
        return self._wall_registry.get_active_walls()
    
    def get_imbalance(
        self,
        center_price: float,
        zone_ticks: Optional[int] = None,
    ) -> ImbalanceMetrics:
        """
        Рассчитывает дисбаланс в зоне цены.
        
        Дисбаланс = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Положительный дисбаланс = давление вверх
        Отрицательный дисбаланс = давление вниз
        """
        if zone_ticks is None:
            zone_ticks = self.config.imbalance_zone_ticks
        
        zone_distance = zone_ticks * self.tick_size
        
        bid_volume = 0.0
        ask_volume = 0.0
        weighted_bid = 0.0
        weighted_ask = 0.0
        
        # Считаем bids в зоне
        for price, quantity in self._bids.items():
            distance = abs(price - center_price)
            if distance <= zone_distance:
                notional = price * quantity
                bid_volume += notional
                
                # Взвешиваем по близости (чем ближе, тем больше вес)
                weight = 1.0 - (distance / zone_distance) if zone_distance > 0 else 1.0
                weighted_bid += notional * weight
        
        # Считаем asks в зоне
        for price, quantity in self._asks.items():
            distance = abs(price - center_price)
            if distance <= zone_distance:
                notional = price * quantity
                ask_volume += notional
                
                weight = 1.0 - (distance / zone_distance) if zone_distance > 0 else 1.0
                weighted_ask += notional * weight
        
        # Рассчитываем дисбаланс
        total = bid_volume + ask_volume
        imbalance = (
            (bid_volume - ask_volume) / total if total > 0 else 0.0
        )
        
        weighted_total = weighted_bid + weighted_ask
        weighted_imbalance = (
            (weighted_bid - weighted_ask) / weighted_total if weighted_total > 0 else 0.0
        )
        
        # Считаем стены в зоне
        bid_walls_count = 0
        ask_walls_count = 0
        
        for wall in self._wall_registry.get_active_walls():
            if wall.distance_to(center_price) <= zone_distance:
                if wall.side == BookSide.BID:
                    bid_walls_count += 1
                else:
                    ask_walls_count += 1
        
        return ImbalanceMetrics(
            ts_ns=self._last_update_ts_ns,
            center_price=center_price,
            zone_ticks=zone_ticks,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
            weighted_imbalance=weighted_imbalance,
            bid_walls_count=bid_walls_count,
            ask_walls_count=ask_walls_count,
        )
    
    def get_level_liquidity(self, level: Level) -> LevelLiquidity:
        """
        Возвращает снимок ликвидности вокруг уровня.
        
        Используется для определения готовности уровня к пробою.
        """
        zone_ticks = self.config.level_zone_ticks
        
        # Получаем стены в зоне уровня
        walls_near_level = self._wall_registry.get_walls_near_level(
            level, zone_ticks, self.tick_size
        )
        
        # Разделяем на bid и ask стены
        bid_walls = [w for w in walls_near_level if w.side == BookSide.BID]
        ask_walls = [w for w in walls_near_level if w.side == BookSide.ASK]
        
        # Определяем, какие стены нужно пробить
        # Для пробоя вверх (пробой сопротивления) - нужно пробить ask стены
        # Для пробоя вниз (пробой поддержки) - нужно пробить bid стены
        walls_to_break = []
        
        # Стены выше уровня (сопротивления для пробоя вверх)
        for wall in ask_walls:
            if wall.price >= level.center:
                walls_to_break.append(wall)
        
        # Стены ниже уровня (поддержки для пробоя вниз)
        for wall in bid_walls:
            if wall.price <= level.center:
                walls_to_break.append(wall)
        
        # Проверяем, проедены ли стены
        ask_wall_consumed = any(
            w.state == WallState.CONSUMED for w in ask_walls
        )
        bid_wall_consumed = any(
            w.state == WallState.CONSUMED for w in bid_walls
        )
        
        # Рассчитываем оценки поддержки и давления
        support_score = sum(
            w.current_notional for w in bid_walls
            if w.state in (WallState.ACTIVE, WallState.CONSUMING)
        )
        pressure_score = sum(
            w.current_notional for w in ask_walls
            if w.state in (WallState.ACTIVE, WallState.CONSUMING)
        )
        
        # Рассчитываем дисбаланс в зоне уровня
        imbalance_metrics = self.get_imbalance(level.center, zone_ticks)
        
        return LevelLiquidity(
            level_id=level.id,
            level_center=level.center,
            ask_walls=ask_walls,
            bid_walls=bid_walls,
            ask_wall_consumed=ask_wall_consumed,
            bid_wall_consumed=bid_wall_consumed,
            support_score=support_score,
            pressure_score=pressure_score,
            imbalance=imbalance_metrics.imbalance,
            walls_to_break=walls_to_break,
        )
    
    def is_wall_consumed(self, price: float, side: BookSide) -> bool:
        """Проверяет, проедена ли стена на конкретной цене."""
        wall = self._wall_registry.get_wall_at(price, side)
        if wall is None:
            return False
        return wall.state == WallState.CONSUMED
    
    def update_best_prices(
        self,
        bid_price: float,
        ask_price: float,
        ts_ns: int,
    ) -> None:
        """
        Обновляет лучшие цены (из bookTicker).
        
        Используется для отслеживания подхода цены к стенам.
        """
        self._last_bid_price = bid_price
        self._last_ask_price = ask_price
        self._last_update_ts_ns = ts_ns
        
        # Проверяем снятие стен при подходе цены (спуфинг)
        self._check_spoofing_on_approach(bid_price, ask_price)
    
    def reset(self) -> None:
        """Сбрасывает состояние анализатора."""
        self._bids.clear()
        self._asks.clear()
        self._wall_registry.reset()
        self._last_bid_price = 0.0
        self._last_ask_price = 0.0
        self._last_update_ts_ns = 0
    
    def _apply_level_change(
        self,
        book: Dict[float, float],
        price: float,
        quantity: float,
        side: BookSide,
        ts_ns: int,
    ) -> None:
        """Применяет изменение уровня в стакане."""
        if quantity <= 0:
            # Уровень удалён
            if price in book:
                del book[price]
            # Отслеживаем как снятие стены
            self._wall_registry.add_or_update_wall(price, side, 0.0, ts_ns)
        else:
            # Уровень обновлён
            book[price] = quantity
            notional = price * quantity
            self._wall_registry.add_or_update_wall(price, side, notional, ts_ns)
    
    def _check_spoofing_on_approach(
        self,
        bid_price: float,
        ask_price: float,
    ) -> None:
        """
        Проверяет снятие стен при подходе цены (спуфинг).
        
        Если стена снимается, когда цена подходит к ней - это спуфинг.
        """
        # Проверяем ask стены при подходе цены снизу
        for wall in self._wall_registry.get_active_walls():
            if wall.side == BookSide.ASK:
                # Если цена подходит к стене снизу и стена снята
                if (ask_price < wall.price and 
                    wall.distance_to(ask_price) < 5 * self.tick_size and
                    wall.state == WallState.CANCELLED):
                    wall.spoof_score = 1.0
            
            elif wall.side == BookSide.BID:
                # Если цена подходит к стене сверху и стена снята
                if (bid_price > wall.price and 
                    wall.distance_to(bid_price) < 5 * self.tick_size and
                    wall.state == WallState.CANCELLED):
                    wall.spoof_score = 1.0


class BookAnalyzerManager:
    """
    Менеджер BookAnalyzer'ов для множества символов.
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._analyzers: Dict[str, BookAnalyzer] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[BookAnalyzerConfig] = None,
    ) -> BookAnalyzer:
        """Возвращает анализатор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._analyzers:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._analyzers[symbol] = BookAnalyzer(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        
        return self._analyzers[symbol]
    
    def on_snapshot(self, snapshot: BookSnapshotEvent) -> None:
        """Обработка снапшота для нужного символа."""
        analyzer = self._analyzers.get(snapshot.symbol.upper())
        if analyzer is not None:
            analyzer.on_snapshot(snapshot)
    
    def on_delta_batch(self, delta_batch: BookDeltaBatchEvent) -> None:
        """Обработка дельт для нужного символа."""
        analyzer = self._analyzers.get(delta_batch.symbol.upper())
        if analyzer is not None:
            analyzer.on_delta_batch(delta_batch)
    
    def get_level_liquidity(
        self,
        symbol: str,
        level: Level,
    ) -> Optional[LevelLiquidity]:
        """Возвращает ликвидность вокруг уровня."""
        analyzer = self._analyzers.get(symbol.upper())
        if analyzer is None:
            return None
        return analyzer.get_level_liquidity(level)
    
    def reset_all(self) -> None:
        """Сбрасывает все анализаторы."""
        for analyzer in self._analyzers.values():
            analyzer.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._analyzers.keys())