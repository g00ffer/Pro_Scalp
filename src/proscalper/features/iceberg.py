"""
Детектор айсбергов (скрытой ликвидности) в стакане.

Айсберг — это крупная заявка, у которой в стакане отображается
только малая часть (visible size), а при её исполнении автоматически
появляется новая порция. Реальный размер скрыт от участников рынка.

Признаки айсберга (из формализации системы):
1. executed_notional >> current_notional (исполнено больше, чем видно)
2. Многократный replenish (уровень восстанавливается после исполнения)
3. Уровень не исчезает при агрессивном проходе цены
4. Малый displayed size при больших исполнениях на том же уровне

Отличие от спуфинга:
- спуфинг снимается без исполнения
- айсберг постоянно исполняется и восстанавливается

Используется в связке с:
- WallRegistry (источник событий стен)
- SpoofDetector (айсберг — противоположность спуфингу)
- BookAnalyzer (учёт скрытой ликвидности в LevelLiquidity)
- SignalGenerator (учёт реальной поддержки/сопротивления)

Архитектурное решение:
- event-driven: получает события стены
- iceberg_score ∈ [0, 1]
- все пороги конфигурируемые
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from proscalper.core.types import BookSide


# ============================================================
# Типы и конфигурация
# ============================================================

class IcebergPattern(Enum):
    """Доминирующий паттерн айсберга."""
    NONE = auto()                # не айсберг
    REPLENISHING = auto()        # восстанавливающаяся ликвидность
    HIGH_EXECUTION = auto()      # много исполнений при малом displayed
    PERSISTENT = auto()          # длительное удержание уровня при проходах


@dataclass(frozen=True)
class IcebergConfig:
    """
    Конфигурация детектора айсбергов.

    Все пороги — разумные значения по умолчанию.
    """
    # --- Основной признак: execution_mult ---
    # executed_notional / current_notional > threshold → подозрение на айсберг
    execution_mult_threshold: float = 3.0

    # --- Replenish ---
    # Сколько раз уровень должен восстановиться, чтобы считаться айсбергом
    min_replenish_count: int = 3

    # --- Persistence ---
    # Минимальное время жизни уровня для айсберга
    min_age_ms: int = 5_000

    # --- Показатели для скоринга ---
    replenish_saturation: int = 10    # replenish_count для score = 1.0
    execution_ratio_saturation: float = 5.0  # exec/current для score = 1.0

    # --- Итоговый порог ---
    iceberg_score_threshold: float = 0.6

    # --- Веса компонентов ---
    w_replenish: float = 0.45
    w_execution_ratio: float = 0.35
    w_persistence: float = 0.20

    # --- Ограничения истории ---
    max_events_per_wall: int = 2000


# ============================================================
# Внутренние структуры
# ============================================================

@dataclass
class _IcebergWallState:
    """Внутреннее состояние стены для анализа на айсберг."""
    price: float
    side: BookSide
    first_seen_ts_ns: int
    last_seen_ts_ns: int = 0

    # Накопленные метрики
    total_executed_notional: float = 0.0
    total_cancelled_notional: float = 0.0
    max_current_notional: float = 0.0
    last_current_notional: float = 0.0

    # Счётчик восстановлений: сколько раз размер рос
    # после падения (признак replenishment)
    replenish_count: int = 0

    # Сколько раз уровень полностью исчезал из стакана
    # (для айсберга это не характерно — обычно он живёт)
    disappeared_count: int = 0

    # История размеров — для детекции pattern fill
    size_history: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class IcebergAnalysis:
    """
    Результат анализа стены на айсберг.
    """
    price: float
    side: BookSide

    iceberg_score: float = 0.0           # [0, 1]
    pattern: IcebergPattern = IcebergPattern.NONE

    # Компоненты
    replenish_score: float = 0.0
    execution_ratio_score: float = 0.0
    persistence_score: float = 0.0

    # Детали
    age_ms: int = 0
    replenish_count: int = 0
    executed_notional: float = 0.0
    current_notional: float = 0.0
    execution_ratio: float = 0.0        # exec / current (или / max, если current=0)

    @property
    def is_likely_iceberg(self) -> bool:
        return self.iceberg_score >= 0.6

    @property
    def is_hidden_liquidity(self) -> bool:
        """Признак наличия скрытой ликвидности (ниже порога, но выше 0.3)."""
        return 0.3 <= self.iceberg_score < 0.6


# ============================================================
# Основной детектор
# ============================================================

class IcebergDetector:
    """
    Детектор айсбергов для одного символа.

    Использование:
        detector = IcebergDetector("BTCUSDT", tick_size=0.1)

        # События от WallRegistry
        detector.on_wall_seen(price, side, current_notional, market_price, ts_ns)
        detector.on_wall_executed(price, side, executed_notional, ts_ns)
        detector.on_wall_cancelled(price, side, cancelled_notional, ts_ns)

        # Анализ
        analysis = detector.analyze_wall(price, side)
        if analysis.is_likely_iceberg:
            ...
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[IcebergConfig] = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size должен быть положительным")

        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or IcebergConfig()

        self._walls: Dict[Tuple[float, BookSide], _IcebergWallState] = {}

    # ============================================
    # События стены
    # ============================================

    def on_wall_seen(
        self,
        price: float,
        side: BookSide,
        current_notional: float,
        market_price: float,
        ts_ns: int,
    ) -> None:
        """
        Вызывается на каждом снимке/обновлении стены (когда стена
        присутствует в стакане).

        Ключевая логика: считаем replenish_count — сколько раз размер
        вырос после падения.
        """
        key = (price, side)

        if key not in self._walls:
            self._walls[key] = _IcebergWallState(
                price=price,
                side=side,
                first_seen_ts_ns=ts_ns,
                last_seen_ts_ns=ts_ns,
                max_current_notional=current_notional,
                last_current_notional=current_notional,
            )
            state = self._walls[key]
            state.size_history.append((ts_ns, current_notional))
            return

        state = self._walls[key]
        prev = state.last_current_notional

        # Восстановление (replenish): размер вырос после падения
        # (даже незначительное падение → рост)
        if current_notional > prev and prev > 0:
            # Проверяем, что рост существенный (≥ 20% от предыдущего)
            growth = (current_notional - prev) / prev
            if growth >= 0.2:
                state.replenish_count += 1

        state.last_current_notional = current_notional
        state.max_current_notional = max(state.max_current_notional, current_notional)
        state.last_seen_ts_ns = ts_ns

        state.size_history.append((ts_ns, current_notional))
        self._trim_history(state)

    def on_wall_executed(
        self,
        price: float,
        side: BookSide,
        executed_notional: float,
        ts_ns: int,
    ) -> None:
        """Часть стены была исполнена маркет-ордерами."""
        state = self._walls.get((price, side))
        if state is None:
            return
        state.total_executed_notional += executed_notional
        state.last_seen_ts_ns = ts_ns

    def on_wall_cancelled(
        self,
        price: float,
        side: BookSide,
        cancelled_notional: float,
        ts_ns: int,
    ) -> None:
        """Часть стены была снята (не исполнена)."""
        state = self._walls.get((price, side))
        if state is None:
            return
        state.total_cancelled_notional += cancelled_notional
        state.last_seen_ts_ns = ts_ns

    def on_wall_disappeared(
        self,
        price: float,
        side: BookSide,
        ts_ns: int,
    ) -> None:
        """Стена полностью исчезла из стакана."""
        state = self._walls.get((price, side))
        if state is None:
            return
        state.disappeared_count += 1
        state.last_seen_ts_ns = ts_ns

    # ============================================
    # Анализ
    # ============================================

    def analyze_wall(
        self,
        price: float,
        side: BookSide,
        now_ns: Optional[int] = None,
    ) -> IcebergAnalysis:
        """Анализирует стену на предмет айсберга."""
        if now_ns is None:
            now_ns = time.time_ns()

        result = IcebergAnalysis(price=price, side=side)
        state = self._walls.get((price, side))
        if state is None:
            return result

        cfg = self.config

        result.age_ms = (now_ns - state.first_seen_ts_ns) // 1_000_000
        result.replenish_count = state.replenish_count
        result.executed_notional = state.total_executed_notional
        result.current_notional = state.last_current_notional

        # Execution ratio: exec / current. Если current ~ 0 — используем max.
        denominator = state.last_current_notional
        if denominator <= 0:
            denominator = state.max_current_notional
        if denominator > 0:
            result.execution_ratio = state.total_executed_notional / denominator

        # --- Компоненты ---
        result.replenish_score = min(
            1.0,
            state.replenish_count / max(1, cfg.replenish_saturation),
        )

        if cfg.execution_ratio_saturation > 0:
            result.execution_ratio_score = min(
                1.0,
                result.execution_ratio / cfg.execution_ratio_saturation,
            )

        if result.age_ms >= cfg.min_age_ms:
            result.persistence_score = 1.0
        else:
            result.persistence_score = result.age_ms / cfg.min_age_ms

        # Штраф за disappeared_count (айсберг не должен исчезать)
        if state.disappeared_count > 0:
            penalty = min(1.0, state.disappeared_count * 0.15)
            result.persistence_score *= (1.0 - penalty)

        # --- Итог ---
        result.iceberg_score = min(
            1.0,
            cfg.w_replenish * result.replenish_score
            + cfg.w_execution_ratio * result.execution_ratio_score
            + cfg.w_persistence * result.persistence_score,
        )

        # Дополнительный жёсткий фильтр: без минимального replenish
        # айсберг не подтверждаем (иначе это просто долго живущая стена)
        if state.replenish_count < cfg.min_replenish_count:
            result.iceberg_score *= 0.5

        result.pattern = self._determine_pattern(result)
        return result

    def get_icebergs(
        self,
        now_ns: Optional[int] = None,
    ) -> List[IcebergAnalysis]:
        """Возвращает все стены, классифицированные как айсберги."""
        if now_ns is None:
            now_ns = time.time_ns()

        results: List[IcebergAnalysis] = []
        for (price, side) in self._walls.keys():
            a = self.analyze_wall(price, side, now_ns)
            if a.is_likely_iceberg:
                results.append(a)
        return results

    # ============================================
    # Обслуживание
    # ============================================

    def prune_old_walls(self, now_ns: int, max_idle_ms: int = 60_000) -> None:
        """
        Удаляет стены, которые давно не обновлялись.

        max_idle_ms — по умолчанию 60 секунд без обновлений.
        """
        cutoff = now_ns - max_idle_ms * 1_000_000
        dead = [
            k for k, s in self._walls.items()
            if s.last_seen_ts_ns < cutoff
        ]
        for k in dead:
            del self._walls[k]

    def reset(self) -> None:
        self._walls.clear()

    def get_stats(self) -> Dict[str, int]:
        return {
            "tracked_walls": len(self._walls),
            "iceberg_candidates": len(self.get_icebergs()),
        }

    # ============================================
    # Внутренние методы
    # ============================================

    def _determine_pattern(self, analysis: IcebergAnalysis) -> IcebergPattern:
        if analysis.iceberg_score < self.config.iceberg_score_threshold:
            return IcebergPattern.NONE

        # Выбор доминирующего паттерна
        if analysis.replenish_score >= analysis.execution_ratio_score:
            if analysis.replenish_score >= analysis.persistence_score:
                return IcebergPattern.REPLENISHING
            return IcebergPattern.PERSISTENT

        if analysis.execution_ratio_score >= analysis.persistence_score:
            return IcebergPattern.HIGH_EXECUTION
        return IcebergPattern.PERSISTENT

    def _trim_history(self, state: _IcebergWallState) -> None:
        max_len = self.config.max_events_per_wall
        if len(state.size_history) > max_len:
            state.size_history = state.size_history[-max_len:]


# ============================================================
# Менеджер для множества символов
# ============================================================

class IcebergDetectorManager:
    """Менеджер IcebergDetector'ов по символам."""

    def __init__(self, tick_sizes: Dict[str, float]) -> None:
        self._tick_sizes = {k.upper(): v for k, v in tick_sizes.items()}
        self._detectors: Dict[str, IcebergDetector] = {}

    def get_or_create(
        self,
        symbol: str,
        config: Optional[IcebergConfig] = None,
    ) -> IcebergDetector:
        symbol = symbol.upper()
        if symbol not in self._detectors:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None or tick_size <= 0:
                raise ValueError(f"Не найден tick_size для {symbol}")
            self._detectors[symbol] = IcebergDetector(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        return self._detectors[symbol]

    def get(self, symbol: str) -> Optional[IcebergDetector]:
        return self._detectors.get(symbol.upper())

    def reset_all(self) -> None:
        for d in self._detectors.values():
            d.reset()

    def all_symbols(self) -> List[str]:
        return list(self._detectors.keys())