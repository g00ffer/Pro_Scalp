"""
Метрики инструмента (Symbol Metrics).

Считает текущее состояние рынка по каждому символу:
- спред (в тиках и %)
- top depth notional (bid + ask)
- trade rate (trades/sec)
- волатильность (rolling range, percentile)
- mid-price и его изменение

Используется в связке с:
- filters.py (FastGuard: спред, глубина, trade rate)
- InstrumentSelector (отбор инструментов по ликвидности/волатильности)
- MarketRegimeDetector (режим рынка)

Архитектурное решение:
- модуль работает на событиях (book ticker, trade)
- держит rolling-историю для percentile расчётов
- не блокирует hot path (лёгкие обновления)
- все пороги не жёстко зашиты, а считаются из данных
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from proscalper.core.events import BookTickerEvent, TradeEvent


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class SymbolMetricsConfig:
    """
    Конфигурация расчёта метрик инструмента.

    Все окна в секундах.
    """
    # Окна для расчёта trade rate
    trade_rate_window_sec: float = 5.0

    # Окна для волатильности
    range_window_sec: float = 60.0
    range_history_sec: float = 3_600.0    # 1 час — база для percentile

    # Окно для медианы trade rate (для adaptive порогов)
    trade_rate_history_sec: float = 300.0  # 5 минут

    # Ограничения по хранению
    max_trades: int = 20_000
    max_mid_prices: int = 10_000
    max_trade_rate_samples: int = 300


# ============================================================
# Снимок метрик
# ============================================================

@dataclass
class SymbolMetricsSnapshot:
    """
    Снимок метрик символа на момент времени.

    Все поля — числа, готовые для передачи в фильтры и журналы.
    """
    ts_ns: int
    symbol: str

    # --- Цена / спред ---
    last_price: float = 0.0
    mid_price: float = 0.0
    spread_ticks: int = 0
    spread_pct: float = 0.0

    # --- Глубина ---
    top_bid_qty: float = 0.0
    top_ask_qty: float = 0.0
    top_depth_notional: float = 0.0    # bid_price * qty + ask_price * qty

    # --- Лента ---
    trades_per_sec: float = 0.0
    trades_per_sec_median: float = 0.0

    # --- Волатильность ---
    range_current: float = 0.0          # диапазон за range_window_sec
    range_pct_current: float = 0.0      # диапазон в % от mid
    range_pct_percentile: float = 0.0   # percentile текущего range_pct за range_history_sec
    realized_vol_pct: float = 0.0       # stddev log-returns за range_window_sec, в %

    # --- Диагностика ---
    valid: bool = True                  # данные полные
    bookticker_age_ms: int = 0


# ============================================================
# Основной класс
# ============================================================

class SymbolMetrics:
    """
    Считает метрики для одного символа.

    Использование:
        metrics = SymbolMetrics("BTCUSDT", tick_size=0.1)

        # На каждом событии:
        metrics.on_book_ticker(event)
        metrics.on_trade(event)

        # Снимок метрик:
        snap = metrics.snapshot()
        if snap.spread_ticks > 3:
            ...

    Все обновления — O(1) амортизированно (с учётом prune).
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[SymbolMetricsConfig] = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size должен быть положительным")

        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self._inv_tick_size = 1.0 / tick_size
        self.config = config or SymbolMetricsConfig()

        # Последний bookTicker
        self._last_bookticker_ts_ns: int = 0
        self._last_bid: float = 0.0
        self._last_bid_qty: float = 0.0
        self._last_ask: float = 0.0
        self._last_ask_qty: float = 0.0

        # История цен (для волатильности)
        self._mid_prices: Deque[Tuple[int, float]] = deque()

        # История сделок (для trade rate)
        self._trade_ts: Deque[int] = deque(maxlen=self.config.max_trades)

        # Медиана trade rate (раз в секунду)
        self._trade_rate_samples: Deque[float] = deque(
            maxlen=self.config.max_trade_rate_samples
        )
        self._last_trade_rate_sample_ts: int = 0
        self._trades_this_second: int = 0

        # Percentile истории
        self._range_pct_history: Deque[Tuple[int, float]] = deque()

    # ============================================
    # События
    # ============================================

    def on_book_ticker(self, event: BookTickerEvent) -> None:
        """Обработка обновления лучших цен."""
        if event.symbol.upper() != self.symbol:
            return

        self._last_bookticker_ts_ns = event.ts_local_ns
        self._last_bid = event.bid_price
        self._last_bid_qty = event.bid_qty
        self._last_ask = event.ask_price
        self._last_ask_qty = event.ask_qty

        if event.bid_price > 0 and event.ask_price > 0:
            mid = (event.bid_price + event.ask_price) / 2.0
            self._mid_prices.append((event.ts_local_ns, mid))

    def on_trade(self, event: TradeEvent) -> None:
        """Обработка сделки для trade rate."""
        if event.symbol.upper() != self.symbol:
            return

        self._trade_ts.append(event.ts_ns or event.ts_local_ns)
        self._trades_this_second += 1

    # ============================================
    # Снимок метрик
    # ============================================

    def snapshot(self, now_ns: Optional[int] = None) -> SymbolMetricsSnapshot:
        """Возвращает снимок метрик символа."""
        if now_ns is None:
            now_ns = time.time_ns()

        self._prune(now_ns)
        self._update_trade_rate_sample(now_ns)

        snap = SymbolMetricsSnapshot(ts_ns=now_ns, symbol=self.symbol)

        # --- Цена / спред ---
        if self._last_bid > 0 and self._last_ask > 0:
            snap.mid_price = (self._last_bid + self._last_ask) / 2.0
            snap.last_price = snap.mid_price
            spread = self._last_ask - self._last_bid
            snap.spread_ticks = int(round(spread * self._inv_tick_size))
            if snap.mid_price > 0:
                snap.spread_pct = 100.0 * spread / snap.mid_price

            # --- Глубина ---
            snap.top_bid_qty = self._last_bid_qty
            snap.top_ask_qty = self._last_ask_qty
            snap.top_depth_notional = (
                self._last_bid * self._last_bid_qty
                + self._last_ask * self._last_ask_qty
            )
        else:
            snap.valid = False

        # --- Trade rate ---
        snap.trades_per_sec = self._compute_trade_rate(now_ns)
        snap.trades_per_sec_median = self._compute_trade_rate_median()

        # --- Волатильность ---
        self._compute_volatility(snap, now_ns)

        # --- Возраст bookTicker ---
        if self._last_bookticker_ts_ns > 0:
            snap.bookticker_age_ms = int(
                (now_ns - self._last_bookticker_ts_ns) / 1_000_000
            )

        return snap

    # ============================================
    # Внутренние вычисления
    # ============================================

    def _compute_trade_rate(self, now_ns: int) -> float:
        """Считает trades/sec в окне trade_rate_window_sec."""
        if not self._trade_ts:
            return 0.0

        window_ns = int(self.config.trade_rate_window_sec * 1_000_000_000)
        cutoff = now_ns - window_ns
        count = sum(1 for ts in self._trade_ts if ts >= cutoff)

        return count / max(self.config.trade_rate_window_sec, 1e-9)

    def _compute_trade_rate_median(self) -> float:
        """Медиана trade rate по сэмплам истории."""
        if not self._trade_rate_samples:
            return 0.0
        sorted_samples = sorted(self._trade_rate_samples)
        n = len(sorted_samples)
        return sorted_samples[n // 2]

    def _update_trade_rate_sample(self, now_ns: int) -> None:
        """
        Раз в секунду сохраняет сэмпл trade rate.
        Используется для adaptive порогов.
        """
        if self._last_trade_rate_sample_ts == 0:
            self._last_trade_rate_sample_ts = now_ns
            return

        elapsed_sec = (now_ns - self._last_trade_rate_sample_ts) / 1_000_000_000
        if elapsed_sec < 1.0:
            return

        # Сэмпл — количество сделок за прошедшую секунду
        sample = float(self._trades_this_second) / max(elapsed_sec, 1e-9)
        self._trade_rate_samples.append(sample)

        self._trades_this_second = 0
        self._last_trade_rate_sample_ts = now_ns

    def _compute_volatility(
        self,
        snap: SymbolMetricsSnapshot,
        now_ns: int,
    ) -> None:
        """Считает range, range_pct, percentile и realized vol."""
        if not self._mid_prices:
            return

        # --- Range за range_window_sec ---
        range_window_ns = int(self.config.range_window_sec * 1_000_000_000)
        range_cutoff = now_ns - range_window_ns

        highs: List[float] = []
        lows: List[float] = []
        prices_for_vol: List[float] = []

        for ts, mid in self._mid_prices:
            if ts >= range_cutoff:
                highs.append(mid)
                lows.append(mid)
                prices_for_vol.append(mid)

        if not highs:
            return

        range_current = max(highs) - min(lows)
        snap.range_current = range_current

        mid_for_pct = snap.mid_price if snap.mid_price > 0 else (
            (max(highs) + min(lows)) / 2.0
        )
        if mid_for_pct > 0:
            snap.range_pct_current = 100.0 * range_current / mid_for_pct

        # --- Realized vol ---
        snap.realized_vol_pct = self._realized_vol_pct(prices_for_vol)

        # --- Percentile range_pct ---
        if snap.range_pct_current > 0:
            self._range_pct_history.append((now_ns, snap.range_pct_current))

        snap.range_pct_percentile = self._range_percentile(
            snap.range_pct_current
        )

    def _realized_vol_pct(self, prices: List[float]) -> float:
        """stddev log-returns * 100."""
        if len(prices) < 3:
            return 0.0

        log_returns: List[float] = []
        for i in range(1, len(prices)):
            p_prev = prices[i - 1]
            p_cur = prices[i]
            if p_prev > 0 and p_cur > 0:
                log_returns.append(math.log(p_cur / p_prev))

        if len(log_returns) < 2:
            return 0.0

        mean = sum(log_returns) / len(log_returns)
        var = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        std = math.sqrt(var)

        return 100.0 * std

    def _range_percentile(self, current_range_pct: float) -> float:
        """Percentile текущего range_pct в истории."""
        if not self._range_pct_history or current_range_pct <= 0:
            return 50.0

        values = [v for _, v in self._range_pct_history]
        if not values:
            return 50.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        below = sum(1 for v in sorted_values if v <= current_range_pct)
        return 100.0 * below / max(n, 1)

    def _prune(self, now_ns: int) -> None:
        """Удаляет устаревшие записи."""
        # mid_prices
        mid_cutoff = now_ns - int(
            self.config.range_history_sec * 1_000_000_000
        )
        while self._mid_prices and self._mid_prices[0][0] < mid_cutoff:
            self._mid_prices.popleft()

        # trade_ts
        trade_cutoff = now_ns - int(
            self.config.trade_rate_history_sec * 1_000_000_000
        )
        while self._trade_ts and self._trade_ts[0] < trade_cutoff:
            self._trade_ts.popleft()

        # range_pct_history
        range_hist_cutoff = now_ns - int(
            self.config.range_history_sec * 1_000_000_000
        )
        while (
            self._range_pct_history
            and self._range_pct_history[0][0] < range_hist_cutoff
        ):
            self._range_pct_history.popleft()

    # ============================================
    # Сброс
    # ============================================

    def reset(self) -> None:
        """Полный сброс состояния."""
        self._mid_prices.clear()
        self._trade_ts.clear()
        self._trade_rate_samples.clear()
        self._range_pct_history.clear()
        self._last_bookticker_ts_ns = 0
        self._last_bid = 0.0
        self._last_bid_qty = 0.0
        self._last_ask = 0.0
        self._last_ask_qty = 0.0
        self._last_trade_rate_sample_ts = 0
        self._trades_this_second = 0


# ============================================================
# Менеджер
# ============================================================

class SymbolMetricsManager:
    """
    Менеджер SymbolMetrics по символам.

    Использование:
        manager = SymbolMetricsManager(tick_sizes={"BTCUSDT": 0.1})
        manager.on_book_ticker(event)
        manager.on_trade(event)
        snap = manager.snapshot("BTCUSDT")
    """

    def __init__(
        self,
        tick_sizes: Dict[str, float],
        config: Optional[SymbolMetricsConfig] = None,
    ) -> None:
        self._tick_sizes = {k.upper(): v for k, v in tick_sizes.items()}
        self._config = config or SymbolMetricsConfig()
        self._metrics: Dict[str, SymbolMetrics] = {}

    def get_or_create(self, symbol: str) -> SymbolMetrics:
        symbol = symbol.upper()
        if symbol not in self._metrics:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None or tick_size <= 0:
                raise ValueError(f"Не найден tick_size для {symbol}")
            self._metrics[symbol] = SymbolMetrics(
                symbol=symbol,
                tick_size=tick_size,
                config=self._config,
            )
        return self._metrics[symbol]

    def get(self, symbol: str) -> Optional[SymbolMetrics]:
        return self._metrics.get(symbol.upper())

    def on_book_ticker(self, event: BookTickerEvent) -> None:
        m = self._metrics.get(event.symbol.upper())
        if m is not None:
            m.on_book_ticker(event)

    def on_trade(self, event: TradeEvent) -> None:
        m = self._metrics.get(event.symbol.upper())
        if m is not None:
            m.on_trade(event)

    def snapshot(self, symbol: str, now_ns: Optional[int] = None) -> Optional[SymbolMetricsSnapshot]:
        m = self._metrics.get(symbol.upper())
        if m is None:
            return None
        return m.snapshot(now_ns)

    def reset_all(self) -> None:
        for m in self._metrics.values():
            m.reset()

    def all_symbols(self) -> List[str]:
        return list(self._metrics.keys())