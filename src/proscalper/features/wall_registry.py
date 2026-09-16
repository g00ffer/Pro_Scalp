"""
Реестр активных стен (плотностей) в стакане.

Расширяет DensityDetector дополнительными возможностями:
- Привязка стен к уровням цен
- История изменений каждой стены (для анализа спуфинга)
- Детекция спуфинга (ложных плотностей)
- Детекция айсбергов (скрытой ликвидности)

Используется в связке с:
- DensityDetector (источник данных о стенах)
- BookAnalyzer (анализ проедания стен)
- LevelDetector (привязка стен к уровням)
- SignalGenerator (подтверждение пробоя)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from proscalper.core.types import BookSide


# ============================================================
# Типы изменений стен
# ============================================================

class WallChangeType(Enum):
    """Тип изменения стены."""
    APPEARED = auto()       # Стена появилась
    UPDATED = auto()        # Размер обновился
    CONSUMED = auto()       # Стена проедена
    CANCELLED = auto()      # Стена снята
    MOVED = auto()          # Стена переместилась на другую цену
    REPLENISHED = auto()    # Стена пополнена (признак айсберга)


@dataclass
class WallChange:
    """
    Запись об изменении стены.
    
    Хранится в истории для анализа спуфинга и айсбергов.
    """
    ts_ns: int
    change_type: WallChangeType
    notional_before: float
    notional_after: float
    price: float
    
    # Для отслеживания перемещений
    prev_price: Optional[float] = None
    
    @property
    def notional_delta(self) -> float:
        """Изменение номинала."""
        return self.notional_after - self.notional_before


# ============================================================
# Конфигурация реестра
# ============================================================

@dataclass
class WallRegistryConfig:
    """
    Конфигурация реестра стен.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    # История изменений
    history_window_sec: int = 300         # окно истории (5 минут)
    max_history_entries: int = 100        # максимум записей на стену
    
    # Детекция спуфинга
    spoof_detection_enabled: bool = True
    spoof_cancel_ratio_threshold: float = 0.5   # >50% снято = спуфинг
    spoof_min_age_ms: int = 3000                # короткоживущие = спуфинг
    spoof_min_executed_notional: float = 100.0  # минимум исполнения
    
    # Детекция айсбергов
    iceberg_detection_enabled: bool = True
    iceberg_exec_mult: float = 2.0              # исполнено > 2× текущего
    iceberg_min_replenish_count: int = 3        # минимум пополнений
    
    # Привязка к уровням
    level_binding_distance_ticks: int = 10      # расстояние до уровня


# ============================================================
# Расширенная запись стены
# ============================================================

@dataclass
class WallEntry:
    """
    Расширенная запись стены с историей изменений.
    
    Содержит:
    - Текущее состояние стены
    - Историю изменений
    - Привязку к уровню
    - Метрики для спуфинга/айсбергов
    """
    # Идентификация
    wall_id: str
    symbol: str
    side: BookSide
    price: float
    
    # Текущее состояние
    current_notional: float = 0.0
    displayed_quantity: float = 0.0
    
    # Пиковые значения
    max_notional: float = 0.0
    
    # Время
    first_seen_ts_ns: int = 0
    last_update_ts_ns: int = 0
    
    # История изменений
    changes: List[WallChange] = field(default_factory=list)
    
    # Привязка к уровню
    bound_level_id: Optional[str] = None
    distance_to_level_ticks: float = 0.0
    
    # Метрики для анализа
    spoof_score: float = 0.0
    iceberg_score: float = 0.0
    
    @property
    def age_ms(self) -> int:
        """Возраст стены в миллисекундах."""
        if self.first_seen_ts_ns == 0:
            return 0
        now_ns = time.time_ns()
        return int((now_ns - self.first_seen_ts_ns) / 1_000_000)
    
    @property
    def time_since_update_ms(self) -> int:
        """Время с последнего обновления."""
        if self.last_update_ts_ns == 0:
            return 0
        now_ns = time.time_ns()
        return int((now_ns - self.last_update_ts_ns) / 1_000_000)
    
    @property
    def total_consumed_notional(self) -> float:
        """Суммарный проеденный номинал."""
        consumed = 0.0
        for change in self.changes:
            if change.change_type == WallChangeType.CONSUMED:
                consumed += abs(change.notional_delta)
        return consumed
    
    @property
    def total_cancelled_notional(self) -> float:
        """Суммарный снятый номинал."""
        cancelled = 0.0
        for change in self.changes:
            if change.change_type == WallChangeType.CANCELLED:
                cancelled += abs(change.notional_delta)
        return cancelled
    
    @property
    def replenish_count(self) -> int:
        """Количество пополнений (признак айсберга)."""
        return sum(
            1 for change in self.changes
            if change.change_type == WallChangeType.REPLENISHED
        )
    
    @property
    def cancel_ratio(self) -> float:
        """Отношение снятого объёма к максимальному."""
        if self.max_notional <= 0:
            return 0.0
        return self.total_cancelled_notional / self.max_notional
    
    @property
    def executed_ratio(self) -> float:
        """Отношение исполненного объёма к максимальному."""
        if self.max_notional <= 0:
            return 0.0
        return self.total_consumed_notional / self.max_notional
    
    def add_change(self, change: WallChange) -> None:
        """Добавляет запись об изменении."""
        self.changes.append(change)
        self.last_update_ts_ns = change.ts_ns
        
        # Ограничиваем размер истории
        if len(self.changes) > 100:
            self.changes = self.changes[-100:]
    
    def to_snapshot(self) -> Dict:
        """Создаёт снимок стены для передачи."""
        return {
            "wall_id": self.wall_id,
            "side": self.side.value,
            "price": self.price,
            "current_notional": self.current_notional,
            "max_notional": self.max_notional,
            "age_ms": self.age_ms,
            "executed_notional": self.total_consumed_notional,
            "cancelled_notional": self.total_cancelled_notional,
            "replenish_count": self.replenish_count,
            "spoof_score": self.spoof_score,
            "iceberg_score": self.iceberg_score,
            "bound_level_id": self.bound_level_id,
        }


# ============================================================
# Реестр стен
# ============================================================

class WallRegistry:
    """
    Реестр активных стен для одного символа.
    
    Отвечает за:
    - Хранение активных стен
    - Привязку стен к уровням
    - Отслеживание истории изменений
    - Детекцию спуфинга и айсбергов
    
    Использование:
        registry = WallRegistry(config)
        
        # При обновлении детектора плотностей
        registry.update_from_detector(detector)
        
        # Привязка к уровням
        registry.bind_to_levels(active_levels)
        
        # Получение стен
        walls = registry.get_walls_near_level(level)
        
        # Анализ спуфинга
        spoof_score = registry.calculate_spoof_score(wall)
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[WallRegistryConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or WallRegistryConfig()
        
        # Активные стены: (side, price) -> WallEntry
        self._walls: Dict[Tuple[BookSide, float], WallEntry] = {}
        
        # Счётчик для генерации ID
        self._wall_counter = 0
    
    def update_from_detector(
        self,
        detector_snapshots: List[Dict],
        ts_ns: int,
    ) -> None:
        """
        Обновляет реестр из снимков детектора плотностей.
        
        Вызывается при каждом обновлении DensityDetector.
        
        Args:
            detector_snapshots: список снимков стен из детектора
            ts_ns: временная метка обновления
        """
        # Собираем текущие цены в стакане
        current_prices = set()
        
        for snapshot in detector_snapshots:
            side_str = snapshot.get("side", "")
            side = BookSide.BID if side_str == "BID" else BookSide.ASK
            price = snapshot.get("price", 0.0)
            current_notional = snapshot.get("current_notional", 0.0)
            
            if price <= 0:
                continue
            
            current_prices.add((side, price))
            
            key = (side, price)
            
            if key in self._walls:
                # Обновляем существующую стену
                wall = self._walls[key]
                prev_notional = wall.current_notional
                
                if current_notional != prev_notional:
                    # Определяем тип изменения
                    if current_notional > prev_notional:
                        # Увеличение — пополнение или обновление
                        if wall.current_notional < wall.max_notional * 0.5:
                            change_type = WallChangeType.REPLENISHED
                        else:
                            change_type = WallChangeType.UPDATED
                    else:
                        # Уменьшение — проедание или снятие
                        decrease_ratio = (prev_notional - current_notional) / prev_notional
                        if decrease_ratio > 0.8:
                            change_type = WallChangeType.CANCELLED
                        else:
                            change_type = WallChangeType.CONSUMED
                    
                    change = WallChange(
                        ts_ns=ts_ns,
                        change_type=change_type,
                        notional_before=prev_notional,
                        notional_after=current_notional,
                        price=price,
                    )
                    wall.add_change(change)
                
                wall.current_notional = current_notional
                wall.max_notional = max(wall.max_notional, current_notional)
            
            else:
                # Создаём новую стену
                self._wall_counter += 1
                wall_id = f"{self.symbol}_{side.value}_{self._wall_counter}"
                
                wall = WallEntry(
                    wall_id=wall_id,
                    symbol=self.symbol,
                    side=side,
                    price=price,
                    current_notional=current_notional,
                    max_notional=current_notional,
                    first_seen_ts_ns=ts_ns,
                    last_update_ts_ns=ts_ns,
                )
                
                # Добавляем запись о появлении
                change = WallChange(
                    ts_ns=ts_ns,
                    change_type=WallChangeType.APPEARED,
                    notional_before=0.0,
                    notional_after=current_notional,
                    price=price,
                )
                wall.add_change(change)
                
                self._walls[key] = wall
        
        # Удаляем стены, которых больше нет в стакане
        keys_to_remove = []
        for key in self._walls:
            if key not in current_prices:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._walls[key]
    
    def bind_to_levels(
        self,
        active_levels: List,
    ) -> None:
        """
        Привязывает стены к уровням цен.
        
        Вызывается периодически для обновления привязок.
        
        Args:
            active_levels: список активных уровней из LevelDetector
        """
        binding_distance = self.config.level_binding_distance_ticks * self.tick_size
        
        for wall in self._walls.values():
            wall.bound_level_id = None
            wall.distance_to_level_ticks = float('inf')
            
            for level in active_levels:
                distance = abs(wall.price - level.center)
                
                if distance <= binding_distance:
                    wall.bound_level_id = level.id
                    wall.distance_to_level_ticks = distance / self.tick_size
                    break
    
    def get_wall_at_price(
        self,
        side: BookSide,
        price: float,
    ) -> Optional[WallEntry]:
        """Возвращает стену на конкретной цене."""
        return self._walls.get((side, price))
    
    def get_walls_by_side(self, side: BookSide) -> List[WallEntry]:
        """Возвращает все стены для указанной стороны."""
        return [
            wall for key, wall in self._walls.items()
            if key[0] == side
        ]
    
    def get_walls_near_level(
        self,
        level_center: float,
        distance_ticks: int = 10,
    ) -> List[WallEntry]:
        """Возвращает стены вблизи уровня."""
        distance = distance_ticks * self.tick_size
        
        return [
            wall for wall in self._walls.values()
            if abs(wall.price - level_center) <= distance
        ]
    
    def get_walls_bound_to_level(self, level_id: str) -> List[WallEntry]:
        """Возвращает стены, привязанные к уровню."""
        return [
            wall for wall in self._walls.values()
            if wall.bound_level_id == level_id
        ]
    
    def calculate_spoof_score(self, wall: WallEntry) -> float:
        """
        Рассчитывает скор спуфинга для стены.
        
        Признаки спуфинга:
        - Высокое отношение снятого объёма к максимальному
        - Короткое время жизни
        - Низкий исполненный объём
        
        Возвращает значение от 0 (не спуфинг) до 1 (явный спуфинг).
        """
        if not self.config.spoof_detection_enabled:
            return 0.0
        
        score = 0.0
        
        # Признак 1: высокий процент снятий
        if wall.cancel_ratio > self.config.spoof_cancel_ratio_threshold:
            score += 0.5
        
        # Признак 2: короткое время жизни
        if wall.age_ms < self.config.spoof_min_age_ms:
            score += 0.2
        
        # Признак 3: низкий исполненный объём
        if wall.total_consumed_notional < self.config.spoof_min_executed_notional:
            score += 0.3
        
        return min(1.0, score)
    
    def calculate_iceberg_score(self, wall: WallEntry) -> float:
        """
        Рассчитывает скор айсберга для стены.
        
        Признаки айсберга:
        - Исполнено значительно больше, чем отображается
        - Множественные пополнения
        
        Возвращает значение от 0 (не айсберг) до 1 (явный айсберг).
        """
        if not self.config.iceberg_detection_enabled:
            return 0.0
        
        # Признак 1: исполнено больше, чем отображается
        if wall.current_notional > 0:
            exec_ratio = wall.total_consumed_notional / wall.current_notional
            if exec_ratio < self.config.iceberg_exec_mult:
                return 0.0
        
        # Признак 2: множественные пополнения
        if wall.replenish_count < self.config.iceberg_min_replenish_count:
            return 0.0
        
        # Рассчитываем скор на основе количества пополнений
        return min(1.0, wall.replenish_count / 10.0)
    
    def update_spoof_iceberg_scores(self) -> None:
        """Обновляет скоры спуфинга и айсбергов для всех стен."""
        for wall in self._walls.values():
            wall.spoof_score = self.calculate_spoof_score(wall)
            wall.iceberg_score = self.calculate_iceberg_score(wall)
    
    def get_wall_history(
        self,
        side: BookSide,
        price: float,
    ) -> List[WallChange]:
        """Возвращает историю изменений стены."""
        wall = self._walls.get((side, price))
        if wall is None:
            return []
        return wall.changes.copy()
    
    def get_stats(self) -> Dict:
        """Возвращает статистику реестра."""
        bid_walls = len(self.get_walls_by_side(BookSide.BID))
        ask_walls = len(self.get_walls_by_side(BookSide.ASK))
        
        bound_walls = sum(
            1 for wall in self._walls.values()
            if wall.bound_level_id is not None
        )
        
        suspicious_walls = sum(
            1 for wall in self._walls.values()
            if wall.spoof_score > 0.5
        )
        
        return {
            "total_walls": len(self._walls),
            "bid_walls": bid_walls,
            "ask_walls": ask_walls,
            "bound_to_levels": bound_walls,
            "suspicious_walls": suspicious_walls,
        }
    
    def reset(self) -> None:
        """Сбрасывает реестр."""
        self._walls.clear()
        self._wall_counter = 0


# ============================================================
# Менеджер реестров стен
# ============================================================

class WallRegistryManager:
    """
    Менеджер WallRegistry для множества символов.
    
    Использование:
        manager = WallRegistryManager(tick_sizes={"BTCUSDT": 0.1})
        
        # Получаем реестр для символа
        registry = manager.get_or_create("BTCUSDT")
        
        # Обновляем из детектора
        registry.update_from_detector(snapshots, ts_ns)
        
        # Привязываем к уровням
        registry.bind_to_levels(active_levels)
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._registries: Dict[str, WallRegistry] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[WallRegistryConfig] = None,
    ) -> WallRegistry:
        """Возвращает реестр для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._registries:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._registries[symbol] = WallRegistry(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        
        return self._registries[symbol]
    
    def update_all_from_detector(
        self,
        symbol: str,
        detector_snapshots: List[Dict],
        ts_ns: int,
    ) -> None:
        """Обновляет реестр из детектора для символа."""
        registry = self._registries.get(symbol.upper())
        if registry is not None:
            registry.update_from_detector(detector_snapshots, ts_ns)
            registry.update_spoof_iceberg_scores()
    
    def bind_all_to_levels(
        self,
        symbol: str,
        active_levels: List,
    ) -> None:
        """Привязывает стены к уровням для символа."""
        registry = self._registries.get(symbol.upper())
        if registry is not None:
            registry.bind_to_levels(active_levels)
    
    def get_walls_near_level(
        self,
        symbol: str,
        level_center: float,
        distance_ticks: int = 10,
    ) -> List[WallEntry]:
        """Возвращает стены вблизи уровня."""
        registry = self._registries.get(symbol.upper())
        if registry is None:
            return []
        return registry.get_walls_near_level(level_center, distance_ticks)
    
    def reset_all(self) -> None:
        """Сбрасывает все реестры."""
        for registry in self._registries.values():
            registry.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._registries.keys())