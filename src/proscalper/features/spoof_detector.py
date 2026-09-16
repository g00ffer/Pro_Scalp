"""
Детектор спуфинга (ложных плотностей в стакане).

Спуфинг — размещение крупных лимитных заявок без намерения их исполнить,
с целью создать ложное впечатление о спросе/предложении.

Признаки спуфинга (из формализации системы):
1. Cancel on approach — заявка снимается при подходе цены
2. Short lived — короткое время жизни без исполнения
3. Layering — множество заявок на разных уровнях в узкой зоне
4. Flickering — многократное появление/исчезновение

Отличие от реальной плотности:
- реальная живёт долго, частично исполняется, не снимается при подходе
- спуфинг снимается, не исполняется, мерцает

Используется в связке с:
- WallRegistry (источник событий стен)
- BookAnalyzer (агрегация в LevelLiquidity)
- SignalGenerator (фильтр ложных сигналов)

Архитектурное решение:
- модуль event-driven: получает события стен и обновляет состояние
- analyze() возвращает оценку без побочных эффектов
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

class SpoofPattern(Enum):
    """Доминирующий паттерн спуфинга."""
    NONE = auto()                 # не спуфинг
    CANCEL_ON_APPROACH = auto()   # снятие при подходе цены
    SHORT_LIVED = auto()          # короткоживущая без исполнения
    LAYERING = auto()             # заявки на нескольких уровнях
    FLICKERING = auto()           # многократное появление/исчезновение


@dataclass(frozen=True)
class SpoofConfig:
    """
    Конфигурация детектора спуфинга.

    Все пороги — разумные значения по умолчанию, подлежат настройке
    на живых данных.
    """
    # --- Cancel on approach ---
    approach_distance_ticks: int = 5
    cancel_on_approach_threshold: float = 0.5
    cancel_on_approach_saturation: int = 2  # сколько снятий = макс. скор

    # --- Short lived ---
    min_age_for_real_ms: int = 3_000
    max_cancel_ratio_for_spoof: float = 0.7
    min_executed_ratio_for_real: float = 0.1

    # --- Layering ---
    layering_levels: int = 3
    layering_distance_ticks: int = 10
    layering_time_window_ms: int = 1_000

    # --- Flickering ---
    flicker_count_threshold: int = 3
    flicker_time_window_ms: int = 5_000

    # --- Итоговый порог ---
    spoof_score_threshold: float = 0.6

    # --- Веса компонентов (сумма ~1.0) ---
    w_cancel_on_approach: float = 0.35
    w_short_lived: float = 0.30
    w_layering: float = 0.20
    w_flickering: float = 0.15


# ============================================================
# Внутренние структуры
# ============================================================

@dataclass
class WallEvent:
    """Событие стены (для истории)."""
    ts_ns: int
    event_type: str  # "created" | "updated" | "cancelled" | "executed"
    price: float
    side: BookSide
    notional: float
    market_price: float  # цена рынка в момент события


@dataclass
class _WallState:
    """Внутреннее состояние одной стены."""
    price: float
    side: BookSide
    created_ts_ns: int
    max_notional: float = 0.0
    current_notional: float = 0.0
    executed_notional: float = 0.0
    cancelled_notional: float = 0.0
    events: List[WallEvent] = field(default_factory=list)
    create_ts_history: List[int] = field(default_factory=list)


@dataclass
class SpoofAnalysis:
    """
    Результат анализа одной стены на спуфинг.
    """
    price: float
    side: BookSide

    # Итог
    spoof_score: float = 0.0            # [0, 1]
    pattern: SpoofPattern = SpoofPattern.NONE

    # Компоненты
    cancel_on_approach_score: float = 0.0
    short_lived_score: float = 0.0
    layering_score: float = 0.0
    flickering_score: float = 0.0

    # Детали
    age_ms: int = 0
    executed_ratio: float = 0.0
    cancelled_ratio: float = 0.0
    cancel_on_approach_count: int = 0

    @property
    def is_likely_spoof(self) -> bool:
        return self.spoof_score >= 0.6

    @property
    def is_likely_real(self) -> bool:
        return self.spoof_score < 0.3


# ============================================================
# Основной детектор
# ============================================================

class SpoofDetector:
    """
    Детектор спуфинга для одного символа.

    Использование:
        detector = SpoofDetector("BTCUSDT", tick_size=0.1)

        # При событиях стены (из WallRegistry)
        detector.on_wall_created(price, side, notional, market_price, ts_ns)
        detector.on_wall_updated(price, side, notional, market_price, ts_ns)
        detector.on_wall_cancelled(price, side, notional, market_price, ts_ns)
        detector.on_wall_executed(price, side, notional, market_price, ts_ns)

        # Периодический анализ
        analysis = detector.analyze_wall(price, side)
        if analysis.is_likely_spoof:
            ...
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[SpoofConfig] = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size должен быть положительным")

        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or SpoofConfig()

        # Ключ: (price, side)
        self._walls: Dict[Tuple[float, BookSide], _WallState] = {}

        # Для быстрого prune
        self._last_prune_ts_ns: int = 0

    # ============================================
    # Обработка событий стен
    # ============================================

    def on_wall_created(
        self,
        price: float,
        side: BookSide,
        notional: float,
        market_price: float,
        ts_ns: int,
    ) -> None:
        """Стена появилась в стакане."""
        key = (price, side)

        if key not in self._walls:
            self._walls[key] = _WallState(
                price=price,
                side=side,
                created_ts_ns=ts_ns,
                max_notional=notional,
                current_notional=notional,
            )
        else:
            state = self._walls[key]
            state.current_notional = notional
            state.max_notional = max(state.max_notional, notional)

        state = self._walls[key]
        state.events.append(
            WallEvent(ts_ns, "created", price, side, notional, market_price)
        )
        state.create_ts_history.append(ts_ns)

        self._trim_history(state, ts_ns)

    def on_wall_updated(
        self,
        price: float,
        side: BookSide,
        notional: float,
        market_price: float,
        ts_ns: int,
    ) -> None:
        """Стена обновилась (размер изменился)."""
        state = self._walls.get((price, side))
        if state is None:
            return

        old = state.current_notional

        if notional < old:
            # уменьшение — трактуем как снятие (в отсутствие точных fill-событий)
            state.cancelled_notional += old - notional
        elif notional > old:
            state.max_notional = max(state.max_notional, notional)

        state.current_notional = notional
        state.events.append(
            WallEvent(ts_ns, "updated", price, side, notional, market_price)
        )

        self._trim_history(state, ts_ns)

    def on_wall_cancelled(
        self,
        price: float,
        side: BookSide,
        notional: float,
        market_price: float,
        ts_ns: int,
    ) -> None:
        """Стена снята (полностью или частично)."""
        state = self._walls.get((price, side))
        if state is None:
            return

        state.cancelled_notional += notional
        state.current_notional = max(0.0, state.current_notional - notional)
        state.events.append(
            WallEvent(ts_ns, "cancelled", price, side, notional, market_price)
        )

        self._trim_history(state, ts_ns)

    def on_wall_executed(
        self,
        price: float,
        side: BookSide,
        notional: float,
        market_price: float,
        ts_ns: int,
    ) -> None:
        """Стена исполнена (частично или полностью)."""
        state = self._walls.get((price, side))
        if state is None:
            return

        state.executed_notional += notional
        state.current_notional = max(0.0, state.current_notional - notional)
        state.events.append(
            WallEvent(ts_ns, "executed", price, side, notional, market_price)
        )

        self._trim_history(state, ts_ns)

    # ============================================
    # Анализ
    # ============================================

    def analyze_wall(
        self,
        price: float,
        side: BookSide,
        now_ns: Optional[int] = None,
    ) -> SpoofAnalysis:
        """
        Анализирует стену на предмет спуфинга.

        Возвращает SpoofAnalysis без побочных эффектов.
        """
        if now_ns is None:
            now_ns = time.time_ns()

        result = SpoofAnalysis(price=price, side=side)

        state = self._walls.get((price, side))
        if state is None:
            return result

        age_ms = (now_ns - state.created_ts_ns) // 1_000_000
        result.age_ms = age_ms

        if state.max_notional > 0:
            result.executed_ratio = state.executed_notional / state.max_notional
            result.cancelled_ratio = state.cancelled_notional / state.max_notional

        # --- Компоненты ---
        result.cancel_on_approach_count = self._count_cancel_on_approach(
            state, price, side
        )
        result.cancel_on_approach_score = self._score_cancel_on_approach(
            result.cancel_on_approach_count
        )
        result.short_lived_score = self._score_short_lived(
            age_ms, result.executed_ratio, result.cancelled_ratio
        )
        result.layering_score = self._score_layering(price, side, now_ns)
        result.flickering_score = self._score_flickering(state, now_ns)

        # --- Итог ---
        cfg = self.config
        result.spoof_score = min(
            1.0,
            cfg.w_cancel_on_approach * result.cancel_on_approach_score
            + cfg.w_short_lived * result.short_lived_score
            + cfg.w_layering * result.layering_score
            + cfg.w_flickering * result.flickering_score,
        )

        result.pattern = self._determine_pattern(result)
        return result

    def get_all_spoof_walls(
        self,
        now_ns: Optional[int] = None,
    ) -> List[SpoofAnalysis]:
        """Возвращает все стены, классифицированные как спуфинг."""
        if now_ns is None:
            now_ns = time.time_ns()

        results: List[SpoofAnalysis] = []
        for (price, side) in self._walls.keys():
            analysis = self.analyze_wall(price, side, now_ns)
            if analysis.is_likely_spoof:
                results.append(analysis)

        return results

    # ============================================
    # Обслуживание
    # ============================================

    def prune_old_walls(self, now_ns: int) -> None:
        """Удаляет устаревшие стены из истории."""
        max_age_ns = self.config.flicker_time_window_ms * 10 * 1_000_000
        cutoff = now_ns - max_age_ns

        dead = [k for k, s in self._walls.items() if s.created_ts_ns < cutoff]
        for k in dead:
            del self._walls[k]

    def reset(self) -> None:
        """Сбрасывает состояние."""
        self._walls.clear()
        self._last_prune_ts_ns = 0

    def get_stats(self) -> Dict[str, int]:
        """Диагностика."""
        return {
            "tracked_walls": len(self._walls),
            "spoof_candidates": sum(
                1 for s in self._walls.values()
                if self.analyze_wall(s.price, s.side).is_likely_spoof
            ),
        }

    # ============================================
    # Внутренние скореры
    # ============================================

    def _count_cancel_on_approach(
        self,
        state: _WallState,
        wall_price: float,
        side: BookSide,
    ) -> int:
        """
        Считает снятия стены при подходе цены.

        Ask-стена: цена подходит снизу (market_price < wall_price).
        Bid-стена: цена подходит сверху (market_price > wall_price).
        """
        approach_distance = self.config.approach_distance_ticks * self.tick_size
        count = 0

        for event in state.events:
            if event.event_type != "cancelled":
                continue

            if side == BookSide.ASK:
                distance = wall_price - event.market_price
            else:
                distance = event.market_price - wall_price

            if 0 < distance <= approach_distance:
                count += 1

        return count

    def _score_cancel_on_approach(self, count: int) -> float:
        if count <= 0:
            return 0.0
        saturation = max(1, self.config.cancel_on_approach_saturation)
        return min(1.0, count / saturation)

    def _score_short_lived(
        self,
        age_ms: int,
        executed_ratio: float,
        cancelled_ratio: float,
    ) -> float:
        """
        Оценка короткоживущей стены без исполнения.

        score ∈ [0, 1]:
        - 0.5 — за время жизни
        - 0.3 — за высокое снятие
        - 0.2 — за низкое исполнение
        """
        cfg = self.config
        score = 0.0

        # Время жизни
        if age_ms < cfg.min_age_for_real_ms:
            age_factor = 1.0 - (age_ms / cfg.min_age_for_real_ms)
            score += 0.5 * age_factor

        # Высокое снятие
        if cancelled_ratio > cfg.max_cancel_ratio_for_spoof:
            span = 1.0 - cfg.max_cancel_ratio_for_spoof
            cancel_factor = (cancelled_ratio - cfg.max_cancel_ratio_for_spoof) / max(span, 1e-9)
            score += 0.3 * cancel_factor

        # Низкое исполнение
        if executed_ratio < cfg.min_executed_ratio_for_real:
            exec_factor = 1.0 - (executed_ratio / cfg.min_executed_ratio_for_real)
            score += 0.2 * exec_factor

        return min(1.0, score)

    def _score_layering(
        self,
        price: float,
        side: BookSide,
        now_ns: int,
    ) -> float:
        """
        Оценка layering — множества недавних стен на одной стороне
        в узкой ценовой зоне.
        """
        cfg = self.config
        window_ns = cfg.layering_time_window_ms * 1_000_000
        cutoff = now_ns - window_ns
        zone = cfg.layering_distance_ticks * self.tick_size
        max_zone = zone * cfg.layering_levels

        layers = 0
        for (p, s), state in self._walls.items():
            if s != side:
                continue
            if state.created_ts_ns < cutoff:
                continue
            d = abs(p - price)
            if 0 < d <= max_zone:
                layers += 1

        if layers < cfg.layering_levels:
            return 0.0

        return min(1.0, layers / (cfg.layering_levels * 2))

    def _score_flickering(self, state: _WallState, now_ns: int) -> float:
        """Многократное появление стены в окне времени."""
        cfg = self.config
        cutoff = now_ns - cfg.flicker_time_window_ms * 1_000_000
        recent = sum(1 for ts in state.create_ts_history if ts >= cutoff)

        if recent < cfg.flicker_count_threshold:
            return 0.0

        return min(1.0, recent / (cfg.flicker_count_threshold * 2))

    def _determine_pattern(self, analysis: SpoofAnalysis) -> SpoofPattern:
        if analysis.spoof_score < self.config.spoof_score_threshold:
            return SpoofPattern.NONE

        scores = {
            SpoofPattern.CANCEL_ON_APPROACH: analysis.cancel_on_approach_score,
            SpoofPattern.SHORT_LIVED: analysis.short_lived_score,
            SpoofPattern.LAYERING: analysis.layering_score,
            SpoofPattern.FLICKERING: analysis.flickering_score,
        }
        return max(scores, key=scores.get)

    def _trim_history(self, state: _WallState, now_ns: int) -> None:
        """Ограничивает историю событий стены."""
        max_window_ns = self.config.flicker_time_window_ms * 1_000_000
        cutoff = now_ns - max_window_ns

        state.events = [e for e in state.events if e.ts_ns >= cutoff]
        state.create_ts_history = [
            ts for ts in state.create_ts_history if ts >= cutoff
        ]

        # Жёсткий предохранитель на размер
        if len(state.events) > 1000:
            state.events = state.events[-1000:]


# ============================================================
# Менеджер для множества символов
# ============================================================

class SpoofDetectorManager:
    """
    Менеджер SpoofDetector'ов по символам.
    """

    def __init__(self, tick_sizes: Dict[str, float]) -> None:
        self._tick_sizes = {k.upper(): v for k, v in tick_sizes.items()}
        self._detectors: Dict[str, SpoofDetector] = {}

    def get_or_create(
        self,
        symbol: str,
        config: Optional[SpoofConfig] = None,
    ) -> SpoofDetector:
        symbol = symbol.upper()
        if symbol not in self._detectors:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None or tick_size <= 0:
                raise ValueError(f"Не найден tick_size для {symbol}")
            self._detectors[symbol] = SpoofDetector(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        return self._detectors[symbol]

    def get(self, symbol: str) -> Optional[SpoofDetector]:
        return self._detectors.get(symbol.upper())

    def reset_all(self) -> None:
        for d in self._detectors.values():
            d.reset()

    def all_symbols(self) -> List[str]:
        return list(self._detectors.keys())