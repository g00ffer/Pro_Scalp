"""
Детектор плотностей (стен) в стакане.

Плотность (wall) — это крупная лимитная заявка, которая может:
- Блокировать движение цены (реальная ликвидность)
- Быть спуфингом (ложная ликвидность)
- Быть айсбергом (скрытая ликвидность)

Модуль отвечает за:
- Обнаружение плотностей по номиналу
- Отслеживание появления/исчезновения
- Расчёт времени жизни
- Гистерезис для избежания мерцания

Используется в связке с:
- WallRegistry (реестр активных стен)
- BookAnalyzer (анализ проедания)
- SignalGenerator (подтверждение пробоя)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from proscalper.core.types import BookSide


class WallState(Enum):
    """Состояние плотности в стакане."""
    FORMING = auto()       # Появилась, но ещё не подтверждена
    ACTIVE = auto()        # Активна и значима
    CONSUMING = auto()     # Проедается маркет-ордерами
    CONSUMED = auto()      # Проедена (исполнена)
    CANCELLED = auto()     # Снята (возможно спуфинг)
    EXPIRED = auto()       # Устарела


@dataclass
class DensityConfig:
    """
    Конфигурация детектора плотностей.
    
    Все пороги относительные (адаптивные к инструменту).
    """
    # Порог входа в реестр стен (в номинале)
    # Стена добавляется, если notional >= wall_entry_notional
    wall_entry_notional: float = 10_000.0
    
    # Порог выхода из реестра (гистерезис)
    # Стена удаляется, если notional <= wall_exit_notional
    # Должен быть меньше entry для избежания мерцания
    wall_exit_notional: float = 5_000.0
    
    # Адаптивный порог (кратный медиане)
    # Если включён, используется вместо абсолютного порога
    use_adaptive_threshold: bool = True
    adaptive_multiplier: float = 3.0  # стена = 3× медианы
    
    # Время подтверждения стены
    # Стена считается активной после этого времени
    wall_confirmation_ms: int = 500
    
    # Максимальное время жизни стены
    # После этого стена считается устаревшей
    wall_max_age_ms: int = 300_000  # 5 минут
    
    # Зона поиска стен (в тиках от текущей цены)
    search_zone_ticks: int = 50
    
    # Максимальное количество отслеживаемых стен
    max_walls_per_side: int = 20


@dataclass
class WallSnapshot:
    """
    Снимок плотности в конкретный момент времени.
    
    Используется для передачи данных между модулями.
    """
    # Идентификация
    price: float
    side: BookSide
    
    # Размеры
    current_notional: float
    max_notional: float
    displayed_quantity: float
    
    # Время
    first_seen_ts_ns: int
    last_update_ts_ns: int
    age_ms: int
    
    # История изменений
    executed_notional: float = 0.0
    cancelled_notional: float = 0.0
    replenish_count: int = 0
    
    # Состояние
    state: WallState = WallState.FORMING
    
    @property
    def is_active(self) -> bool:
        """Стена активна."""
        return self.state in (WallState.ACTIVE, WallState.CONSUMING)
    
    @property
    def is_consumed(self) -> bool:
        """Стена проедена."""
        return self.state == WallState.CONSUMED
    
    @property
    def consumption_ratio(self) -> float:
        """Доля проедания (0 = не проедена, 1 = полностью)."""
        if self.max_notional <= 0:
            return 0.0
        return self.executed_notional / self.max_notional


class _TrackedWall:
    """
    Внутренняя структура для отслеживания стены.
    
    Хранит полную историю изменений стены.
    """
    __slots__ = (
        'price', 'side', 'current_quantity', 'current_notional',
        'max_notional', 'first_seen_ts_ns', 'last_update_ts_ns',
        'executed_notional', 'cancelled_notional', 'replenish_count',
        'state', '_prev_quantity',
    )
    
    def __init__(
        self,
        price: float,
        side: BookSide,
        quantity: float,
        notional: float,
        ts_ns: int,
    ):
        self.price = price
        self.side = side
        self.current_quantity = quantity
        self.current_notional = notional
        self.max_notional = notional
        self.first_seen_ts_ns = ts_ns
        self.last_update_ts_ns = ts_ns
        self.executed_notional = 0.0
        self.cancelled_notional = 0.0
        self.replenish_count = 0
        self.state = WallState.FORMING
        self._prev_quantity = quantity
    
    def update(
        self,
        new_quantity: float,
        new_notional: float,
        ts_ns: int,
    ) -> None:
        """
        Обновляет состояние стены.
        
        Отслеживает:
        - Уменьшение (исполнение или снятие)
        - Увеличение (пополнение)
        """
        delta = new_quantity - self._prev_quantity
        
        if delta < 0:
            # Уменьшение размера
            decrease_notional = abs(delta) * self.price
            
            # Эвристика: если цена была близка к стене,
            # считаем что это исполнение, иначе снятие
            # (в реальной реализации нужна информация о сделках)
            # Для упрощения считаем всё исполнением
            self.executed_notional += decrease_notional
        
        elif delta > 0:
            # Увеличение размера (пополнение)
            self.replenish_count += 1
        
        self.current_quantity = new_quantity
        self.current_notional = new_notional
        self.max_notional = max(self.max_notional, new_notional)
        self.last_update_ts_ns = ts_ns
        self._prev_quantity = new_quantity
    
    def mark_cancelled(self, ts_ns: int) -> None:
        """Помечает стену как снятую."""
        if self.current_quantity > 0:
            self.cancelled_notional += self.current_notional
        self.current_quantity = 0.0
        self.current_notional = 0.0
        self.state = WallState.CANCELLED
        self.last_update_ts_ns = ts_ns
    
    def to_snapshot(self, now_ns: int) -> WallSnapshot:
        """Создаёт снимок стены."""
        age_ms = int((now_ns - self.first_seen_ts_ns) / 1_000_000)
        
        return WallSnapshot(
            price=self.price,
            side=self.side,
            current_notional=self.current_notional,
            max_notional=self.max_notional,
            displayed_quantity=self.current_quantity,
            first_seen_ts_ns=self.first_seen_ts_ns,
            last_update_ts_ns=self.last_update_ts_ns,
            age_ms=age_ms,
            executed_notional=self.executed_notional,
            cancelled_notional=self.cancelled_notional,
            replenish_count=self.replenish_count,
            state=self.state,
        )


class DensityDetector:
    """
    Детектор плотностей в стакане.
    
    Работает на срезах стакана и отслеживает крупные заявки.
    
    Использование:
        detector = DensityDetector(config)
        
        # На каждом обновлении стакана
        detector.on_book_update(
            bids=[(price, qty), ...],
            asks=[(price, qty), ...],
            ts_ns=time.time_ns(),
        )
        
        # Получаем активные стены
        walls = detector.get_active_walls()
    """
    
    def __init__(
        self,
        config: Optional[DensityConfig] = None,
        tick_size: float = 0.01,
    ):
        self.config = config or DensityConfig()
        self.tick_size = tick_size
        
        # Отслеживаемые стены: (side, price) -> _TrackedWall
        self._walls: Dict[Tuple[BookSide, float], _TrackedWall] = {}
        
        # Медианы для адаптивного порога
        self._bid_notional_history: List[float] = []
        self._ask_notional_history: List[float] = []
        self._median_cache: Dict[BookSide, float] = {
            BookSide.BID: 0.0,
            BookSide.ASK: 0.0,
        }
        self._median_cache_ts: int = 0
    
    def on_book_update(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ns: int,
    ) -> None:
        """
        Обработка обновления стакана.
        
        Вызывается на каждом применении дельт или снапшота.
        
        Args:
            bids: список (цена, количество) для бидов
            asks: список (цена, количество) для асков
            ts_ns: временная метка обновления
        """
        # Обновляем медианы для адаптивного порога
        self._update_medians(bids, asks, ts_ns)
        
        # Получаем пороги
        bid_threshold = self._get_threshold(BookSide.BID)
        ask_threshold = self._get_threshold(BookSide.ASK)
        
        # Обрабатываем биды
        self._update_side(
            side=BookSide.BID,
            levels=bids,
            threshold=bid_threshold,
            ts_ns=ts_ns,
        )
        
        # Обрабатываем аски
        self._update_side(
            side=BookSide.ASK,
            levels=asks,
            threshold=ask_threshold,
            ts_ns=ts_ns,
        )
        
        # Удаляем исчезнувшие стены
        self._cleanup_missing_walls(bids, asks, ts_ns)
        
        # Обновляем состояния стен
        self._update_wall_states(ts_ns)
    
    def get_active_walls(self) -> List[WallSnapshot]:
        """Возвращает все активные стены."""
        now_ns = time.time_ns()
        return [
            wall.to_snapshot(now_ns)
            for wall in self._walls.values()
            if wall.state in (WallState.ACTIVE, WallState.CONSUMING)
        ]
    
    def get_walls_by_side(self, side: BookSide) -> List[WallSnapshot]:
        """Возвращает активные стены для указанной стороны."""
        now_ns = time.time_ns()
        return [
            wall.to_snapshot(now_ns)
            for wall in self._walls.values()
            if wall.side == side and wall.state in (WallState.ACTIVE, WallState.CONSUMING)
        ]
    
    def get_wall_at_price(
        self,
        side: BookSide,
        price: float,
    ) -> Optional[WallSnapshot]:
        """Возвращает стену на конкретной цене."""
        key = (side, price)
        wall = self._walls.get(key)
        if wall is None:
            return None
        return wall.to_snapshot(time.time_ns())
    
    def get_all_walls(self) -> List[WallSnapshot]:
        """Возвращает все отслеживаемые стены (включая неактивные)."""
        now_ns = time.time_ns()
        return [
            wall.to_snapshot(now_ns)
            for wall in self._walls.values()
        ]
    
    def get_stats(self) -> Dict:
        """Возвращает статистику детектора."""
        active = sum(
            1 for w in self._walls.values()
            if w.state in (WallState.ACTIVE, WallState.CONSUMING)
        )
        forming = sum(
            1 for w in self._walls.values()
            if w.state == WallState.FORMING
        )
        consumed = sum(
            1 for w in self._walls.values()
            if w.state == WallState.CONSUMED
        )
        
        return {
            "total_tracked": len(self._walls),
            "active": active,
            "forming": forming,
            "consumed": consumed,
            "bid_threshold": self._get_threshold(BookSide.BID),
            "ask_threshold": self._get_threshold(BookSide.ASK),
        }
    
    def reset(self) -> None:
        """Сбрасывает состояние детектора."""
        self._walls.clear()
        self._bid_notional_history.clear()
        self._ask_notional_history.clear()
        self._median_cache = {BookSide.BID: 0.0, BookSide.ASK: 0.0}
        self._median_cache_ts = 0
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    def _update_side(
        self,
        side: BookSide,
        levels: List[Tuple[float, float]],
        threshold: float,
        ts_ns: int,
    ) -> None:
        """Обновляет стены для одной стороны стакана."""
        for price, quantity in levels:
            if quantity <= 0:
                continue
            
            notional = price * quantity
            key = (side, price)
            
            if notional >= threshold:
                # Стена должна быть в реестре
                if key in self._walls:
                    # Обновляем существующую
                    self._walls[key].update(quantity, notional, ts_ns)
                else:
                    # Создаём новую (если не превышен лимит)
                    side_count = sum(
                        1 for k in self._walls
                        if k[0] == side
                    )
                    if side_count < self.config.max_walls_per_side:
                        self._walls[key] = _TrackedWall(
                            price=price,
                            side=side,
                            quantity=quantity,
                            notional=notional,
                            ts_ns=ts_ns,
                        )
            
            elif key in self._walls:
                # Стена ниже порога выхода — проверяем гистерезис
                wall = self._walls[key]
                exit_threshold = self.config.wall_exit_notional
                
                if self.config.use_adaptive_threshold:
                    exit_threshold = threshold * 0.5  # гистерезис 50%
                
                if notional <= exit_threshold:
                    # Стена исчезла
                    wall.mark_cancelled(ts_ns)
    
    def _cleanup_missing_walls(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ns: int,
    ) -> None:
        """Удаляет стены, которых больше нет в стакане."""
        bid_prices = {price for price, _ in bids}
        ask_prices = {price for price, _ in asks}
        
        keys_to_remove = []
        
        for key, wall in self._walls.items():
            side, price = key
            
            if side == BookSide.BID and price not in bid_prices:
                wall.mark_cancelled(ts_ns)
                keys_to_remove.append(key)
            elif side == BookSide.ASK and price not in ask_prices:
                wall.mark_cancelled(ts_ns)
                keys_to_remove.append(key)
        
        # Удаляем помеченные стены (оставляем для истории)
        # На самом деле удаляем только совсем старые
        for key in keys_to_remove:
            wall = self._walls[key]
            age_ms = (ts_ns - wall.last_update_ts_ns) / 1_000_000
            if age_ms > 5000:  # удаляем через 5 секунд после исчезновения
                del self._walls[key]
    
    def _update_wall_states(self, ts_ns: int) -> None:
        """Обновляет состояния стен."""
        for wall in self._walls.values():
            if wall.state == WallState.FORMING:
                # Проверяем время подтверждения
                age_ms = (ts_ns - wall.first_seen_ts_ns) / 1_000_000
                if age_ms >= self.config.wall_confirmation_ms:
                    wall.state = WallState.ACTIVE
            
            elif wall.state == WallState.ACTIVE:
                # Проверяем проедание
                if wall.max_notional > 0:
                    consumption = wall.executed_notional / wall.max_notional
                    if consumption >= 0.7:
                        wall.state = WallState.CONSUMED
                    elif consumption >= 0.2:
                        wall.state = WallState.CONSUMING
                
                # Проверяем устаревание
                age_ms = (ts_ns - wall.first_seen_ts_ns) / 1_000_000
                if age_ms > self.config.wall_max_age_ms:
                    wall.state = WallState.EXPIRED
    
    def _update_medians(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ns: int,
    ) -> None:
        """Обновляет медианы номиналов для адаптивного порога."""
        # Обновляем кэш раз в секунду
        if ts_ns - self._median_cache_ts < 1_000_000_000:
            return
        
        # Собираем номиналы топ-20 уровней
        bid_notionals = [p * q for p, q in bids[:20] if q > 0]
        ask_notionals = [p * q for p, q in asks[:20] if q > 0]
        
        if bid_notionals:
            self._bid_notional_history.append(self._median(bid_notionals))
            # Ограничиваем историю
            if len(self._bid_notional_history) > 100:
                self._bid_notional_history.pop(0)
        
        if ask_notionals:
            self._ask_notional_history.append(self._median(ask_notionals))
            if len(self._ask_notional_history) > 100:
                self._ask_notional_history.pop(0)
        
        # Обновляем кэш
        if self._bid_notional_history:
            self._median_cache[BookSide.BID] = self._median(self._bid_notional_history)
        if self._ask_notional_history:
            self._median_cache[BookSide.ASK] = self._median(self._ask_notional_history)
        
        self._median_cache_ts = ts_ns
    
    def _get_threshold(self, side: BookSide) -> float:
        """Возвращает порог для определения стены."""
        if self.config.use_adaptive_threshold:
            median = self._median_cache.get(side, 0.0)
            if median > 0:
                return median * self.config.adaptive_multiplier
        
        return self.config.wall_entry_notional
    
    @staticmethod
    def _median(values: List[float]) -> float:
        """Рассчитывает медиану списка значений."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        return sorted_values[n // 2]