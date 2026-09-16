"""
Агрегатор баров (OHLCV) из потока сделок.

Строит бары заданной длительности (1s, 5s, 1m) из raw trades.
Используется как вход для LevelDetector и других feature-модулей.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from proscalper.core.events import TradeEvent


@dataclass
class Bar:
    """Один бар (свеча)."""
    ts_ns: int  # время открытия бара
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float  # сумма quantity
    notional: float  # сумма price * quantity
    trade_count: int
    buy_volume: float  # объём, где buyer is taker
    sell_volume: float  # объём, где seller is taker
    
    @property
    def net_delta(self) -> float:
        """Разница между buy и sell объёмами."""
        return self.buy_volume - self.sell_volume
    
    @property
    def duration_sec(self) -> float:
        """Длительность бара в секундах."""
        return 0.0  # определяется aggregator'ом
    
    def is_bullish(self) -> bool:
        return self.close >= self.open
    
    def is_bearish(self) -> bool:
        return self.close < self.open


class BarAggregator:
    """
    Агрегирует trades в бары заданной длительности.
    
    Поддерживает несколько таймфреймов одновременно (1s, 5s, 1m).
    Вызывает callback при закрытии каждого бара.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe_ns: int,
        on_bar_close: Optional[Callable[[Bar], None]] = None,
    ):
        """
        Args:
            symbol: символ инструмента
            timeframe_ns: длительность бара в наносекундах
            on_bar_close: callback при закрытии бара
        """
        self.symbol = symbol.upper()
        self.timeframe_ns = timeframe_ns
        self.timeframe_sec = timeframe_ns / 1_000_000_000
        self._on_bar_close = on_bar_close
        
        # Текущий открытый бар
        self._current_bar: Optional[Bar] = None
        self._current_bucket_ns: int = 0
    
    def on_trade(self, event: TradeEvent) -> None:
        """Обработка новой сделки."""
        ts_ns = event.ts_exchange_ns or event.ts_local_ns
        
        # Определяем bucket для этой сделки
        bucket_ns = (ts_ns // self.timeframe_ns) * self.timeframe_ns
        
        # Если это новый bucket - закрываем старый бар
        if bucket_ns != self._current_bucket_ns:
            # ВАЖНО: закрываем бар только если он валидный (есть данные)
            if (self._current_bar is not None 
                and self._on_bar_close is not None
                and self._current_bar.high > 0  # Фильтр нулевых баров
            ):
                self._on_bar_close(self._current_bar)
            
            # Начинаем новый бар
            self._current_bucket_ns = bucket_ns
            is_buy = not event.is_buyer_maker
            
            self._current_bar = Bar(
                ts_ns=bucket_ns,
                symbol=self.symbol,
                open=event.price,
                high=event.price,
                low=event.price,
                close=event.price,
                volume=event.quantity,
                notional=event.notional,
                trade_count=1,
                buy_volume=event.quantity if is_buy else 0.0,
                sell_volume=event.quantity if not is_buy else 0.0,
            )
        else:
            # Обновляем текущий бар
            if self._current_bar is not None:
                is_buy = not event.is_buyer_maker
                
                self._current_bar.high = max(self._current_bar.high, event.price)
                self._current_bar.low = min(self._current_bar.low, event.price)
                self._current_bar.close = event.price
                self._current_bar.volume += event.quantity
                self._current_bar.notional += event.notional
                self._current_bar.trade_count += 1
                
                if is_buy:
                    self._current_bar.buy_volume += event.quantity
                else:
                    self._current_bar.sell_volume += event.quantity
    
    def flush(self) -> None:
        """Принудительно закрывает текущий бар (используется при остановке)."""
        # ВАЖНО: закрываем только валидные бары
        if (self._current_bar is not None 
            and self._on_bar_close is not None
            and self._current_bar.high > 0  # Фильтр нулевых баров
        ):
            self._on_bar_close(self._current_bar)
            self._current_bar = None
            self._current_bucket_ns = 0
    
    def reset(self) -> None:
        """Сбрасывает состояние."""
        self._current_bar = None
        self._current_bucket_ns = 0


class MultiTimeframeAggregator:
    """
    Агрегатор для нескольких таймфреймов одновременно.
    
    Пример:
        aggregator = MultiTimeframeAggregator(
            symbol="BTCUSDT",
            timeframes_ns=[1_000_000_000, 5_000_000_000],  # 1s, 5s
            on_bar_close=handle_bar,
        )
    """
    
    def __init__(
        self,
        symbol: str,
        timeframes_ns: list[int],
        on_bar_close: Optional[Callable[[Bar], None]] = None,
    ):
        self.symbol = symbol.upper()
        self._aggregators: Dict[int, BarAggregator] = {}
        
        for tf_ns in timeframes_ns:
            self._aggregators[tf_ns] = BarAggregator(
                symbol=symbol,
                timeframe_ns=tf_ns,
                on_bar_close=on_bar_close,
            )
    
    def on_trade(self, event: TradeEvent) -> None:
        """Обработка сделки для всех таймфреймов."""
        for aggregator in self._aggregators.values():
            aggregator.on_trade(event)
    
    def flush(self) -> None:
        """Закрывает все открытые бары."""
        for aggregator in self._aggregators.values():
            aggregator.flush()
    
    def reset(self) -> None:
        """Сбрасывает все агрегаторы."""
        for aggregator in self._aggregators.values():
            aggregator.reset()
