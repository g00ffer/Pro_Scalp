"""
Кросс-маркетный анализ: фьючерсы vs спот.

Цель — использовать спотовый стакан как подтверждение реальности
фьючерсных плотностей. Крупная заявка на фьючерсах, не подтверждённая
аналогичной на споте, часто является манипулятивной (spoofing).

Из формализации системы:
- НЕ жёсткий veto, а штраф/бонус к spoof_score
- только BTCUSDT и ETHUSDT для MVP
- только best bid/ask + top-N уровней

Фичи:
- spot_futures_basis         — расхождение mid-price spot/futures
- futures_ask_wall_notional  — номинал ask-стены на фьючерсах у уровня
- spot_ask_wall_notional     — номинал ask-стены на споте у того же уровня
- cross_confirmation_score   — [0, 1] подтверждение реальности ликвидности
- spoof_penalty              — штраф к spoof_score при расхождении

Логика для лонга (пробой сопротивления вверх):
- futures ask wall consumed + spot ask wall consumed → bonus
- futures bid wall большой, spot bid wall слабый → penalty

Используется в связке с:
- WallRegistry (источник стен)
- SpoofDetector (коррекция spoof_score)
- SignalGenerator (корректировка итогового скора)

Архитектурное решение:
- лёгкий модуль без собственных WS — принимает снимки через update_* методы
- двойной стакан (futures + spot) с одинаковым интерфейсом
- все веса и пороги конфигурируемые
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

class CrossMarketSignal(Enum):
    """Итоговый сигнал кросс-маркетного анализа."""
    NONE = auto()                # нет подтверждения
    CONFIRMED = auto()           # фьючерсная ликвидность подтверждена спотом
    WEAK_CONFIRMATION = auto()   # частичное подтверждение
    CONTRADICTED = auto()        # противоречие — вероятен спуфинг


@dataclass(frozen=True)
class CrossMarketConfig:
    """
    Конфигурация кросс-маркетного анализа.

    Пороги — разумные значения по умолчанию.
    """
    # --- Basis ---
    # Максимально допустимый модуль basis в % от spot mid price
    max_basis_pct: float = 0.3

    # --- Wall search ---
    # Зона поиска стен относительно уровня (в тиках)
    wall_search_ticks: int = 10

    # --- Минимальные размеры стен для учёта ---
    min_wall_notional: float = 10_000.0

    # --- Отношение размеров spot/futures для подтверждения ---
    # Если spot_ask_notional / futures_ask_notional >= confirm_ratio → подтверждено
    spot_futures_confirm_ratio: float = 0.4

    # Если spot_ask_notional / futures_ask_notional < contradict_ratio → противоречие
    spot_futures_contradict_ratio: float = 0.15

    # --- Веса для итогового скора ---
    w_confirmation: float = 0.6
    w_basis: float = 0.4

    # --- Штраф за противоречие ---
    max_spoof_penalty: float = 0.4

    # --- Пороги для классификации ---
    confirmed_threshold: float = 0.6
    contradicted_threshold: float = 0.3


# ============================================================
# Внутренние структуры
# ============================================================

@dataclass
class _MarketSnapshot:
    """Снимок одного рынка (futures или spot)."""
    symbol: str
    ts_ns: int
    mid_price: float
    bid_price: float
    ask_price: float
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class CrossMarketAnalysis:
    """
    Результат кросс-маркетного анализа для одного уровня.
    """
    symbol: str
    level_price: float
    side: BookSide    # сторона, которую анализируем (стена на этой стороне)

    # Итог
    signal: CrossMarketSignal = CrossMarketSignal.NONE
    cross_confirmation_score: float = 0.0   # [0, 1]
    spoof_penalty: float = 0.0              # [0, max_spoof_penalty]

    # Детали
    basis_pct: float = 0.0
    futures_wall_notional: float = 0.0
    spot_wall_notional: float = 0.0
    wall_ratio: float = 0.0     # spot / futures

    @property
    def is_confirmed(self) -> bool:
        return self.signal == CrossMarketSignal.CONFIRMED

    @property
    def is_contradicted(self) -> bool:
        return self.signal == CrossMarketSignal.CONTRADICTED


# ============================================================
# Основной анализатор
# ============================================================

class CrossMarketAnalyzer:
    """
    Кросс-маркетный анализатор для одного символа.

    Использование:
        analyzer = CrossMarketAnalyzer("BTCUSDT", tick_size=0.1)

        # При обновлении фьючерсного стакана
        analyzer.update_futures(mid, bid, ask, bids, asks, ts_ns)

        # При обновлении спотового стакана
        analyzer.update_spot(mid, bid, ask, bids, asks, ts_ns)

        # Анализ уровня
        analysis = analyzer.analyze_level(level_price, BookSide.ASK)
        if analysis.is_contradicted:
            spoof_score += analysis.spoof_penalty
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float,
        spot_symbol: Optional[str] = None,
        config: Optional[CrossMarketConfig] = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size должен быть положительным")

        self.symbol = symbol.upper()
        self.spot_symbol = (spot_symbol or symbol).upper()
        self.tick_size = tick_size
        self.config = config or CrossMarketConfig()

        self._futures: Optional[_MarketSnapshot] = None
        self._spot: Optional[_MarketSnapshot] = None

    # ============================================
    # Обновление снимков
    # ============================================

    def update_futures(
        self,
        mid_price: float,
        bid_price: float,
        ask_price: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ns: Optional[int] = None,
    ) -> None:
        if ts_ns is None:
            ts_ns = time.time_ns()
        self._futures = _MarketSnapshot(
            symbol=self.symbol,
            ts_ns=ts_ns,
            mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bids=list(bids),
            asks=list(asks),
        )

    def update_spot(
        self,
        mid_price: float,
        bid_price: float,
        ask_price: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ns: Optional[int] = None,
    ) -> None:
        if ts_ns is None:
            ts_ns = time.time_ns()
        self._spot = _MarketSnapshot(
            symbol=self.spot_symbol,
            ts_ns=ts_ns,
            mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bids=list(bids),
            asks=list(asks),
        )

    # ============================================
    # Анализ уровня
    # ============================================

    def analyze_level(
        self,
        level_price: float,
        side: BookSide,
    ) -> CrossMarketAnalysis:
        """
        Анализирует уровень на наличие подтверждения/противоречия
        между фьючерсным и спотовым стаканом.

        side — сторона стены, которую проверяем:
        - ASK: проверяем, подтверждена ли плотность на продажу
        - BID: проверяем, подтверждена ли плотность на покупку
        """
        result = CrossMarketAnalysis(
            symbol=self.symbol,
            level_price=level_price,
            side=side,
        )

        if self._futures is None or self._spot is None:
            return result

        # --- Basis ---
        if self._spot.mid_price > 0:
            basis = self._futures.mid_price - self._spot.mid_price
            result.basis_pct = 100.0 * basis / self._spot.mid_price

        # --- Стены у уровня ---
        result.futures_wall_notional = self._find_wall_notional(
            self._futures, side, level_price
        )
        result.spot_wall_notional = self._find_wall_notional(
            self._spot, side, level_price
        )

        if result.futures_wall_notional > 0:
            result.wall_ratio = (
                result.spot_wall_notional / result.futures_wall_notional
            )

        # --- Скоры компонентов ---
        confirm_score = self._score_wall_confirmation(result.wall_ratio)
        basis_score = self._score_basis(result.basis_pct)

        result.cross_confirmation_score = min(
            1.0,
            self.config.w_confirmation * confirm_score
            + self.config.w_basis * basis_score,
        )

        # --- Штраф за противоречие ---
        if (
            result.futures_wall_notional >= self.config.min_wall_notional
            and result.wall_ratio < self.config.spot_futures_contradict_ratio
        ):
            # Фьючерсная стена крупная, спот её не подтверждает
            contradiction_strength = 1.0 - (
                result.wall_ratio / max(self.config.spot_futures_contradict_ratio, 1e-9)
            )
            result.spoof_penalty = (
                self.config.max_spoof_penalty
                * max(0.0, min(1.0, contradiction_strength))
            )

        # --- Итоговый сигнал ---
        result.signal = self._classify_signal(
            result.cross_confirmation_score,
            result.spoof_penalty,
        )

        return result

    # ============================================
    # Внутренние методы
    # ============================================

    def _find_wall_notional(
        self,
        snapshot: _MarketSnapshot,
        side: BookSide,
        level_price: float,
    ) -> float:
        """
        Суммирует номинал уровней в зоне вокруг level_price на указанной
        стороне стакана.
        """
        zone = self.config.wall_search_ticks * self.tick_size
        low = level_price - zone
        high = level_price + zone

        levels = snapshot.asks if side == BookSide.ASK else snapshot.bids

        total = 0.0
        for price, qty in levels:
            if low <= price <= high and qty > 0:
                total += price * qty

        return total

    def _score_wall_confirmation(self, wall_ratio: float) -> float:
        """
        Подтверждение по стене:
        - ratio >= confirm_ratio → 1.0
        - ratio <= contradict_ratio → 0.0
        - линейная интерполяция в промежутке
        """
        cfg = self.config
        if wall_ratio >= cfg.spot_futures_confirm_ratio:
            return 1.0
        if wall_ratio <= cfg.spot_futures_contradict_ratio:
            return 0.0

        span = cfg.spot_futures_confirm_ratio - cfg.spot_futures_contradict_ratio
        return (wall_ratio - cfg.spot_futures_contradict_ratio) / max(span, 1e-9)

    def _score_basis(self, basis_pct: float) -> float:
        """Скор basis: чем меньше отклонение, тем выше скор."""
        cfg = self.config
        if cfg.max_basis_pct <= 0:
            return 1.0
        deviation = abs(basis_pct) / cfg.max_basis_pct
        return max(0.0, 1.0 - deviation)

    def _classify_signal(
        self,
        confirmation_score: float,
        spoof_penalty: float,
    ) -> CrossMarketSignal:
        cfg = self.config
        if spoof_penalty > 0:
            return CrossMarketSignal.CONTRADICTED
        if confirmation_score >= cfg.confirmed_threshold:
            return CrossMarketSignal.CONFIRMED
        if confirmation_score >= cfg.contradicted_threshold:
            return CrossMarketSignal.WEAK_CONFIRMATION
        return CrossMarketSignal.NONE

    # ============================================
    # Диагностика
    # ============================================

    def has_both_snapshots(self) -> bool:
        return self._futures is not None and self._spot is not None

    def get_basis_pct(self) -> float:
        if self._futures is None or self._spot is None:
            return 0.0
        if self._spot.mid_price <= 0:
            return 0.0
        return 100.0 * (self._futures.mid_price - self._spot.mid_price) / self._spot.mid_price

    def reset(self) -> None:
        self._futures = None
        self._spot = None


# ============================================================
# Менеджер для множества символов
# ============================================================

class CrossMarketManager:
    """
    Менеджер CrossMarketAnalyzer'ов по символам.

    Использование:
        manager = CrossMarketManager(
            tick_sizes={"BTCUSDT": 0.1, "ETHUSDT": 0.01},
            spot_symbols={"BTCUSDT": "BTCUSDT", "ETHUSDT": "ETHUSDT"},
        )

        # Обновления
        manager.update_futures("BTCUSDT", mid, bid, ask, bids, asks, ts_ns)
        manager.update_spot("BTCUSDT", mid, bid, ask, bids, asks, ts_ns)

        # Анализ
        analysis = manager.analyze_level("BTCUSDT", level_price, BookSide.ASK)
    """

    def __init__(
        self,
        tick_sizes: Dict[str, float],
        spot_symbols: Optional[Dict[str, str]] = None,
        config: Optional[CrossMarketConfig] = None,
    ) -> None:
        self._tick_sizes = {k.upper(): v for k, v in tick_sizes.items()}
        self._spot_symbols = {
            k.upper(): v.upper()
            for k, v in (spot_symbols or {}).items()
        }
        self._config = config or CrossMarketConfig()
        self._analyzers: Dict[str, CrossMarketAnalyzer] = {}

    def get_or_create(
        self,
        symbol: str,
        config: Optional[CrossMarketConfig] = None,
    ) -> CrossMarketAnalyzer:
        symbol = symbol.upper()
        if symbol not in self._analyzers:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None or tick_size <= 0:
                raise ValueError(f"Не найден tick_size для {symbol}")

            spot_symbol = self._spot_symbols.get(symbol, symbol)
            self._analyzers[symbol] = CrossMarketAnalyzer(
                symbol=symbol,
                tick_size=tick_size,
                spot_symbol=spot_symbol,
                config=config or self._config,
            )
        return self._analyzers[symbol]

    def get(self, symbol: str) -> Optional[CrossMarketAnalyzer]:
        return self._analyzers.get(symbol.upper())

    def update_futures(
        self,
        symbol: str,
        mid_price: float,
        bid_price: float,
        ask_price: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ns: Optional[int] = None,
    ) -> None:
        analyzer = self._analyzers.get(symbol.upper())
        if analyzer is None:
            return
        analyzer.update_futures(mid_price, bid_price, ask_price, bids, asks, ts_ns)

    def update_spot(
        self,
        symbol: str,
        mid_price: float,
        bid_price: float,
        ask_price: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ts_ns: Optional[int] = None,
    ) -> None:
        analyzer = self._analyzers.get(symbol.upper())
        if analyzer is None:
            return
        analyzer.update_spot(mid_price, bid_price, ask_price, bids, asks, ts_ns)

    def analyze_level(
        self,
        symbol: str,
        level_price: float,
        side: BookSide,
    ) -> Optional[CrossMarketAnalysis]:
        analyzer = self._analyzers.get(symbol.upper())
        if analyzer is None:
            return None
        return analyzer.analyze_level(level_price, side)

    def reset_all(self) -> None:
        for a in self._analyzers.values():
            a.reset()

    def all_symbols(self) -> List[str]:
        return list(self._analyzers.keys())