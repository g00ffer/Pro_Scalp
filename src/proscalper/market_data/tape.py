"""
Анализатор ленты сделок (Tape Analyzer).

Предоставляет rolling-метрики для каждого символа:
- buy/sell объёмы за разные окна (1s, 5s, 30s)
- net delta
- trade rate (trades/sec)
- volume bursts
- absorption detection
- percentiles для адаптивных порогов

Работает в hot path, минимальные аллокации.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from proscalper.core.events import TradeEvent


@dataclass
class TapeMetrics:
    """Снимок метрик ленты в конкретный момент времени."""
    ts_ns: int
    symbol: str
    
    # Объёмы за rolling windows
    buy_volume_1s: float = 0.0
    sell_volume_1s: float = 0.0
    net_delta_1s: float = 0.0
    aggressor_volume_1s: float = 0.0  # объём в сторону текущего движения
    
    buy_volume_5s: float = 0.0
    sell_volume_5s: float = 0.0
    net_delta_5s: float = 0.0
    
    buy_volume_30s: float = 0.0
    sell_volume_30s: float = 0.0
    net_delta_30s: float = 0.0
    
    # Trade rate
    trades_per_sec_1s: float = 0.0
    trades_per_sec_5s: float = 0.0
    
    # Медианы (для адаптивных порогов)
    buy_volume_median_1s: float = 0.0
    sell_volume_median_1s: float = 0.0
    net_delta_median_1s: float = 0.0
    trades_per_sec_median: float = 0.0
    
    # Производные
    volume_imbalance_1s: float = 0.0  # (buy - sell) / (buy + sell)
    last_price: float = 0.0
    price_change_1s: float = 0.0
    
    @property
    def total_volume_1s(self) -> float:
        return self.buy_volume_1s + self.sell_volume_1s
    
    @property
    def total_volume_5s(self) -> float:
        return self.buy_volume_5s + self.sell_volume_5s


@dataclass
class _TradeRecord:
    """Внутренняя запись о сделке для rolling window."""
    ts_ns: int
    price: float
    quantity: float
    notional: float
    is_buy: bool  # True если buyer is taker (агрессор - покупатель)


class RollingWindow:
    """
    Circular buffer с автоматическим удалением устаревших записей.
    
    Используется для расчёта метрик за разные временные окна.
    """
    
    def __init__(self, window_ns: int, max_size: int = 100_000):
        self._window_ns = window_ns
        self._max_size = max_size
        self._records: Deque[_TradeRecord] = deque(maxlen=max_size)
    
    def add(self, record: _TradeRecord) -> None:
        self._records.append(record)
    
    def prune(self, now_ns: int) -> None:
        """Удаляет устаревшие записи."""
        cutoff = now_ns - self._window_ns
        while self._records and self._records[0].ts_ns < cutoff:
            self._records.popleft()
    
    def compute_volumes(self) -> tuple[float, float]:
        """Возвращает (buy_volume, sell_volume) за окно."""
        buy = 0.0
        sell = 0.0
        for r in self._records:
            if r.is_buy:
                buy += r.notional
            else:
                sell += r.notional
        return buy, sell
    
    def count(self) -> int:
        return len(self._records)
    
    def last_price(self) -> float:
        return self._records[-1].price if self._records else 0.0
    
    def first_price(self) -> float:
        return self._records[0].price if self._records else 0.0


class PercentileTracker:
    """
    Отслеживает percentiles метрик за длинное окно (например, 1 час).
    
    Используется для адаптивных порогов: "volume burst = 2.5x от медианы".
    Работает через bucketed histogram для O(1) обновления.
    """
    
    def __init__(
        self,
        window_ns: int = 3_600_000_000_000,  # 1 час
        bucket_size: float = 1000.0,
        max_buckets: int = 10000,
    ):
        self._window_ns = window_ns
        self._bucket_size = bucket_size
        self._buckets: Dict[int, int] = {}  # bucket_index -> count
        self._samples: Deque[tuple[int, int]] = deque()  # (ts_ns, bucket_index)
        self._total_count = 0
        self._max_buckets = max_buckets
    
    def add(self, ts_ns: int, value: float) -> None:
        """Добавляет новое значение."""
        bucket_idx = int(abs(value) / self._bucket_size)
        
        self._buckets[bucket_idx] = self._buckets.get(bucket_idx, 0) + 1
        self._samples.append((ts_ns, bucket_idx))
        self._total_count += 1
        
        # Prune старых значений
        cutoff = ts_ns - self._window_ns
        while self._samples and self._samples[0][0] < cutoff:
            _, old_idx = self._samples.popleft()
            self._buckets[old_idx] -= 1
            if self._buckets[old_idx] == 0:
                del self._buckets[old_idx]
            self._total_count -= 1
    
    def percentile(self, p: float) -> float:
        """
        Возвращает p-й percentile (0-100).
        Использует linear scan по bucket'ам.
        """
        if self._total_count == 0:
            return 0.0
        
        target = int(self._total_count * p / 100.0)
        cumulative = 0
        
        for bucket_idx in sorted(self._buckets.keys()):
            count = self._buckets[bucket_idx]
            if cumulative + count >= target:
                return (bucket_idx + 0.5) * self._bucket_size
            cumulative += count
        
        return 0.0
    
    def median(self) -> float:
        """Возвращает медиану (50-й percentile)."""
        return self.percentile(50.0)


class TapeAnalyzer:
    """
    Анализатор ленты сделок для одного символа.
    
    Используется для:
    - подтверждения пробоя (volume burst, delta)
    - определения импульса (trade rate acceleration)
    - детекции абсорбции (объём есть, цены нет)
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Rolling windows
        self._window_1s = RollingWindow(window_ns=1_000_000_000)
        self._window_5s = RollingWindow(window_ns=5_000_000_000)
        self._window_30s = RollingWindow(window_ns=30_000_000_000)
        
        # Percentile trackers для адаптивных порогов
        self._buy_volume_percentile = PercentileTracker(
            window_ns=3_600_000_000_000,
            bucket_size=500.0,
        )
        self._sell_volume_percentile = PercentileTracker(
            window_ns=3_600_000_000_000,
            bucket_size=500.0,
        )
        self._trade_rate_percentile = PercentileTracker(
            window_ns=3_600_000_000_000,
            bucket_size=5.0,
        )
        
        # Для подсчёта trade rate
        self._last_second_bucket_ns = 0
        self._trades_this_second = 0
        
        # Last known state
        self._last_trade_ts_ns = 0
        self._last_price = 0.0
        
        # Snapshot cache (чтобы не пересчитывать каждый раз)
        self._cached_metrics: Optional[TapeMetrics] = None
        self._cache_ts_ns = 0
        self._cache_ttl_ns = 10_000_000  # 10 ms
    
    def on_trade(self, event: TradeEvent) -> None:
        """
        Обработка новой сделки.
        
        В hot path - минимум аллокаций.
        """
        is_buy = not event.is_buyer_maker  # buyer is taker = aggressor is buyer
        
        record = _TradeRecord(
            ts_ns=event.ts_exchange_ns or event.ts_local_ns,
            price=event.price,
            quantity=event.quantity,
            notional=event.notional,
            is_buy=is_buy,
        )
        
        self._window_1s.add(record)
        self._window_5s.add(record)
        self._window_30s.add(record)
        
        # Обновляем percentile trackers (раз в секунду, не каждый trade)
        second_bucket = record.ts_ns // 1_000_000_000
        if second_bucket != self._last_second_bucket_ns:
            # Новая секунда - агрегируем предыдущую
            if self._last_second_bucket_ns > 0:
                buy_vol, sell_vol = self._window_1s.compute_volumes()
                self._buy_volume_percentile.add(
                    record.ts_ns, buy_vol
                )
                self._sell_volume_percentile.add(
                    record.ts_ns, sell_vol
                )
                self._trade_rate_percentile.add(
                    record.ts_ns, float(self._trades_this_second)
                )
            
            self._last_second_bucket_ns = second_bucket
            self._trades_this_second = 0
        
        self._trades_this_second += 1
        self._last_trade_ts_ns = record.ts_ns
        self._last_price = record.price
        
        # Инвалидируем кэш
        self._cached_metrics = None
    
    def snapshot(self, now_ns: Optional[int] = None) -> TapeMetrics:
        """
        Возвращает снимок текущих метрик.
        
        Кэшируется на 10 мс, чтобы избежать повторных вычислений
        при множественных проверках в SignalGenerator.
        """
        if now_ns is None:
            now_ns = time.time_ns()
        
        # Используем кэш если свежий
        if (
            self._cached_metrics is not None
            and now_ns - self._cache_ts_ns < self._cache_ttl_ns
        ):
            return self._cached_metrics
        
        # Prune старых записей
        self._window_1s.prune(now_ns)
        self._window_5s.prune(now_ns)
        self._window_30s.prune(now_ns)
        
        # Вычисляем объёмы
        buy_1s, sell_1s = self._window_1s.compute_volumes()
        buy_5s, sell_5s = self._window_5s.compute_volumes()
        buy_30s, sell_30s = self._window_30s.compute_volumes()
        
        # Trade rate
        count_1s = self._window_1s.count()
        count_5s = self._window_5s.count()
        trades_per_sec_1s = float(count_1s)
        trades_per_sec_5s = count_5s / 5.0
        
        # Дельты
        net_delta_1s = buy_1s - sell_1s
        net_delta_5s = buy_5s - sell_5s
        net_delta_30s = buy_30s - sell_30s
        
        # Агрессорный объём (в сторону текущего движения)
        aggressor_volume_1s = buy_1s if net_delta_1s > 0 else sell_1s
        
        # Volume imbalance: [-1, 1]
        total_1s = buy_1s + sell_1s
        volume_imbalance_1s = (
            net_delta_1s / total_1s if total_1s > 0 else 0.0
        )
        
        # Медианы для адаптивных порогов
        buy_median = self._buy_volume_percentile.median()
        sell_median = self._sell_volume_percentile.median()
        trades_median = self._trade_rate_percentile.median()
        
        # Цена
        last_price = self._last_price
        first_price_1s = self._window_1s.first_price()
        price_change_1s = (
            last_price - first_price_1s if first_price_1s > 0 else 0.0
        )
        
        metrics = TapeMetrics(
            ts_ns=now_ns,
            symbol=self.symbol,
            buy_volume_1s=buy_1s,
            sell_volume_1s=sell_1s,
            net_delta_1s=net_delta_1s,
            aggressor_volume_1s=aggressor_volume_1s,
            buy_volume_5s=buy_5s,
            sell_volume_5s=sell_5s,
            net_delta_5s=net_delta_5s,
            buy_volume_30s=buy_30s,
            sell_volume_30s=sell_30s,
            net_delta_30s=net_delta_30s,
            trades_per_sec_1s=trades_per_sec_1s,
            trades_per_sec_5s=trades_per_sec_5s,
            buy_volume_median_1s=buy_median,
            sell_volume_median_1s=sell_median,
            net_delta_median_1s=abs(net_delta_1s),  # примерно
            trades_per_sec_median=trades_median,
            volume_imbalance_1s=volume_imbalance_1s,
            last_price=last_price,
            price_change_1s=price_change_1s,
        )
        
        self._cached_metrics = metrics
        self._cache_ts_ns = now_ns
        
        return metrics
    
    def detect_volume_burst(
        self,
        multiplier: float = 2.5,
    ) -> tuple[bool, str]:
        """
        Обнаруживает всплеск объёма.
        
        Возвращает (is_burst, direction), где direction в {"BUY", "SELL", ""}.
        """
        metrics = self.snapshot()
        
        if metrics.buy_volume_median_1s <= 0:
            return False, ""
        
        buy_ratio = (
            metrics.buy_volume_1s / metrics.buy_volume_median_1s
            if metrics.buy_volume_median_1s > 0 else 0.0
        )
        sell_ratio = (
            metrics.sell_volume_1s / metrics.sell_volume_median_1s
            if metrics.sell_volume_median_1s > 0 else 0.0
        )
        
        if buy_ratio >= multiplier and buy_ratio > sell_ratio:
            return True, "BUY"
        
        if sell_ratio >= multiplier and sell_ratio > buy_ratio:
            return True, "SELL"
        
        return False, ""
    
    def detect_absorption(self, min_volume: float = 10000.0) -> bool:
        """
        Детектит абсорбцию: большой объём при слабом движении цены.
        
        Признак ложного пробоя или разворота.
        """
        metrics = self.snapshot()
        
        if metrics.total_volume_1s < min_volume:
            return False
        
        if metrics.last_price <= 0:
            return False
        
        # price_change как процент от цены
        price_move_pct = abs(metrics.price_change_1s) / metrics.last_price
        
        # Если объём большой, а движение < 0.05% - это абсорбция
        return price_move_pct < 0.0005
    
    def reset(self) -> None:
        """Сбрасывает состояние (используется при reconnect)."""
        self._window_1s = RollingWindow(window_ns=1_000_000_000)
        self._window_5s = RollingWindow(window_ns=5_000_000_000)
        self._window_30s = RollingWindow(window_ns=30_000_000_000)
        self._buy_volume_percentile = PercentileTracker()
        self._sell_volume_percentile = PercentileTracker()
        self._trade_rate_percentile = PercentileTracker()
        self._cached_metrics = None
        self._last_trade_ts_ns = 0
        self._last_price = 0.0


class TapeManager:
    """
    Менеджер TapeAnalyzer'ов для множества символов.
    """
    
    def __init__(self):
        self._analyzers: Dict[str, TapeAnalyzer] = {}
    
    def get_or_create(self, symbol: str) -> TapeAnalyzer:
        """Возвращает анализатор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        if symbol not in self._analyzers:
            self._analyzers[symbol] = TapeAnalyzer(symbol)
        return self._analyzers[symbol]
    
    def on_trade(self, event: TradeEvent) -> None:
        """Обработка сделки для нужного символа."""
        analyzer = self.get_or_create(event.symbol)
        analyzer.on_trade(event)
    
    def snapshot(self, symbol: str) -> Optional[TapeMetrics]:
        """Возвращает snapshot для символа."""
        analyzer = self._analyzers.get(symbol.upper())
        if analyzer is None:
            return None
        return analyzer.snapshot()
    
    def reset_all(self) -> None:
        """Сбрасывает все анализаторы."""
        for analyzer in self._analyzers.values():
            analyzer.reset()
    
    def all_symbols(self) -> List[str]:
        return list(self._analyzers.keys())