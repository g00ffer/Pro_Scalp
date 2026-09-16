"""
Анализатор ленты сделок.

Отвечает за расчёт метрик из потока сделок:
- Дельта (покупки - продажи)
- Объём и номинал
- Скорость сделок
- Всплески объёма
- Скользящие метрики

Используется в связке с:
- ImpulseScore (оценка силы пробоя)
- ImpulseExhaustion (исчерпание импульса)
- CompressionDetector (консолидация)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from proscalper.core.types import TradeEvent


@dataclass
class TapeConfig:
    """
    Конфигурация анализатора ленты.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    # Окна анализа
    short_window_sec: int = 10          # короткое окно
    medium_window_sec: int = 30         # среднее окно
    long_window_sec: int = 60           # длинное окно
    
    # Всплески объёма
    volume_burst_multiplier: float = 2.0  # всплеск = 2× медианы
    
    # Скорость сделок
    trade_rate_window_sec: int = 10     # окно для скорости сделок
    
    # Хранение
    max_trades: int = 10000             # максимум сделок в памяти


@dataclass
class TapeMetrics:
    """
    Метрики ленты сделок.
    
    Содержит все расчётные метрики за разные окна.
    """
    ts_ns: int
    symbol: str
    
    # Объёмы за разные окна
    volume_short: float = 0.0           # объём за короткое окно
    volume_medium: float = 0.0          # объём за среднее окно
    volume_long: float = 0.0            # объём за длинное окно
    
    # Номиналы за разные окна
    notional_short: float = 0.0
    notional_medium: float = 0.0
    notional_long: float = 0.0
    
    # Дельта (покупки - продажи)
    net_delta_short: float = 0.0
    net_delta_medium: float = 0.0
    net_delta_long: float = 0.0
    
    # Скорость сделок
    trade_rate: float = 0.0             # сделок в секунду
    
    # Всплеск объёма
    volume_burst_ratio: float = 1.0     # текущий объём / медиана
    
    # Медианы для нормализации
    volume_median: float = 0.0
    
    # Производные
    has_volume_burst: bool = False      # есть ли всплеск объёма
    is_high_activity: bool = False      # высокая активность
    
    @property
    def delta_ratio(self) -> float:
        """Соотношение дельты к объёму [-1, 1]."""
        if self.volume_short <= 0:
            return 0.0
        return self.net_delta_short / self.volume_short
    
    @property
    def is_buy_dominant(self) -> bool:
        """Покупки доминируют."""
        return self.net_delta_short > 0
    
    @property
    def is_sell_dominant(self) -> bool:
        """Продажи доминируют."""
        return self.net_delta_short < 0


class _TradePoint:
    """Внутренняя структура для точки сделки."""
    __slots__ = ('ts_ns', 'price', 'quantity', 'notional', 'is_buy')
    
    def __init__(
        self,
        ts_ns: int,
        price: float,
        quantity: float,
        is_buy: bool,
    ):
        self.ts_ns = ts_ns
        self.price = price
        self.quantity = quantity
        self.notional = price * quantity
        self.is_buy = is_buy


class TapeAnalyzer:
    """
    Анализатор ленты сделок для одного символа.
    
    Работает на потоке сделок и рассчитывает метрики
    за разные временные окна.
    
    Использование:
        analyzer = TapeAnalyzer("BTCUSDT")
        
        # На каждой сделке
        analyzer.on_trade(trade_event)
        
        # Периодически получаем метрики
        metrics = analyzer.snapshot()
        
        if metrics.has_volume_burst:
            print("Всплеск объёма!")
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[TapeConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.config = config or TapeConfig()
        
        # История сделок
        self._trades: Deque[_TradePoint] = deque(maxlen=self.config.max_trades)
        
        # Последняя цена
        self._last_price: float = 0.0
        self._last_price_ts_ns: int = 0
        
        # Кэш для медианы
        self._volume_median_cache: float = 0.0
        self._volume_median_last_update_ns: int = 0
    
    def on_trade(self, trade: TradeEvent) -> None:
        """
        Обработка новой сделки.
        
        Вызывается на каждой сделке из потока.
        """
        if trade.price <= 0 or trade.quantity <= 0:
            return
        
        point = _TradePoint(
            ts_ns=trade.ts_ns,
            price=trade.price,
            quantity=trade.quantity,
            is_buy=(trade.side == "buy"),
        )
        
        self._trades.append(point)
        
        # Обновляем последнюю цену
        self._last_price = trade.price
        self._last_price_ts_ns = trade.ts_ns
    
    def snapshot(self) -> TapeMetrics:
        """
        Возвращает снимок метрик ленты.
        
        Вызывается периодически для получения текущих метрик.
        """
        now_ns = self._last_price_ts_ns or time.time_ns()
        
        # Рассчитываем объёмы за разные окна
        volume_short = self._calculate_volume(self.config.short_window_sec, now_ns)
        volume_medium = self._calculate_volume(self.config.medium_window_sec, now_ns)
        volume_long = self._calculate_volume(self.config.long_window_sec, now_ns)
        
        # Рассчитываем номиналы
        notional_short = self._calculate_notional(self.config.short_window_sec, now_ns)
        notional_medium = self._calculate_notional(self.config.medium_window_sec, now_ns)
        notional_long = self._calculate_notional(self.config.long_window_sec, now_ns)
        
        # Рассчитываем дельту
        net_delta_short = self._calculate_net_delta(self.config.short_window_sec, now_ns)
        net_delta_medium = self._calculate_net_delta(self.config.medium_window_sec, now_ns)
        net_delta_long = self._calculate_net_delta(self.config.long_window_sec, now_ns)
        
        # Рассчитываем скорость сделок
        trade_rate = self._calculate_trade_rate(now_ns)
        
        # Рассчитываем медиану объёма
        volume_median = self._calculate_volume_median(now_ns)
        
        # Рассчитываем всплеск объёма
        volume_burst_ratio = (
            volume_short / volume_median if volume_median > 0 else 1.0
        )
        
        # Определяем флаги
        has_volume_burst = volume_burst_ratio >= self.config.volume_burst_multiplier
        is_high_activity = trade_rate > 50.0  # больше 50 сделок в секунду
        
        return TapeMetrics(
            ts_ns=now_ns,
            symbol=self.symbol,
            volume_short=volume_short,
            volume_medium=volume_medium,
            volume_long=volume_long,
            notional_short=notional_short,
            notional_medium=notional_medium,
            notional_long=notional_long,
            net_delta_short=net_delta_short,
            net_delta_medium=net_delta_medium,
            net_delta_long=net_delta_long,
            trade_rate=trade_rate,
            volume_burst_ratio=volume_burst_ratio,
            volume_median=volume_median,
            has_volume_burst=has_volume_burst,
            is_high_activity=is_high_activity,
        )
    
    def get_last_price(self) -> float:
        """Возвращает последнюю цену."""
        return self._last_price
    
    def get_last_price_ts_ns(self) -> int:
        """Возвращает временную метку последней цены."""
        return self._last_price_ts_ns
    
    def reset(self) -> None:
        """Сбрасывает состояние анализатора."""
        self._trades.clear()
        self._last_price = 0.0
        self._last_price_ts_ns = 0
        self._volume_median_cache = 0.0
        self._volume_median_last_update_ns = 0
    
    def _calculate_volume(self, window_sec: int, now_ns: int) -> float:
        """Рассчитывает объём за указанное окно."""
        if not self._trades:
            return 0.0
        
        cutoff_ns = now_ns - window_sec * 1_000_000_000
        
        total_volume = 0.0
        for trade in self._trades:
            if trade.ts_ns >= cutoff_ns:
                total_volume += trade.quantity
        
        return total_volume
    
    def _calculate_notional(self, window_sec: int, now_ns: int) -> float:
        """Рассчитывает номинал за указанное окно."""
        if not self._trades:
            return 0.0
        
        cutoff_ns = now_ns - window_sec * 1_000_000_000
        
        total_notional = 0.0
        for trade in self._trades:
            if trade.ts_ns >= cutoff_ns:
                total_notional += trade.notional
        
        return total_notional
    
    def _calculate_net_delta(self, window_sec: int, now_ns: int) -> float:
        """
        Рассчитывает дельту (покупки - продажи) за указанное окно.
        """
        if not self._trades:
            return 0.0
        
        cutoff_ns = now_ns - window_sec * 1_000_000_000
        
        buy_volume = 0.0
        sell_volume = 0.0
        
        for trade in self._trades:
            if trade.ts_ns >= cutoff_ns:
                if trade.is_buy:
                    buy_volume += trade.quantity
                else:
                    sell_volume += trade.quantity
        
        return buy_volume - sell_volume
    
    def _calculate_trade_rate(self, now_ns: int) -> float:
        """
        Рассчитывает скорость сделок (сделок в секунду).
        """
        if not self._trades:
            return 0.0
        
        window_ns = self.config.trade_rate_window_sec * 1_000_000_000
        cutoff_ns = now_ns - window_ns
        
        count = 0
        for trade in self._trades:
            if trade.ts_ns >= cutoff_ns:
                count += 1
        
        if self.config.trade_rate_window_sec <= 0:
            return 0.0
        
        return count / self.config.trade_rate_window_sec
    
    def _calculate_volume_median(self, now_ns: int) -> float:
        """
        Рассчитывает медиану объёма за длинное окно.
        
        Кэшируется на 10 секунд для экономии ресурсов.
        """
        # Проверяем кэш
        cache_ttl_ns = 10_000_000_000  # 10 секунд
        if (now_ns - self._volume_median_last_update_ns) < cache_ttl_ns:
            return self._volume_median_cache
        
        if not self._trades:
            return 0.0
        
        # Собираем объёмы сделок за окно
        window_ns = self.config.long_window_sec * 1_000_000_000
        cutoff_ns = now_ns - window_ns
        
        volumes = []
        for trade in self._trades:
            if trade.ts_ns >= cutoff_ns and trade.quantity > 0:
                volumes.append(trade.quantity)
        
        if not volumes:
            self._volume_median_cache = 0.0
        else:
            volumes.sort()
            n = len(volumes)
            if n % 2 == 0:
                self._volume_median_cache = (volumes[n // 2 - 1] + volumes[n // 2]) / 2
            else:
                self._volume_median_cache = volumes[n // 2]
        
        self._volume_median_last_update_ns = now_ns
        return self._volume_median_cache


class TapeAnalyzerManager:
    """
    Менеджер TapeAnalyzer'ов для множества символов.
    
    Использование:
        manager = TapeAnalyzerManager()
        
        # Получаем анализатор для символа
        analyzer = manager.get_or_create("BTCUSDT")
        
        # Передаём сделки
        manager.on_trade(trade_event)
        
        # Получаем метрики
        metrics = manager.snapshot("BTCUSDT")
    """
    
    def __init__(self):
        self._analyzers: Dict[str, TapeAnalyzer] = {}
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[TapeConfig] = None,
    ) -> TapeAnalyzer:
        """Возвращает анализатор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._analyzers:
            self._analyzers[symbol] = TapeAnalyzer(
                symbol=symbol,
                config=config,
            )
        
        return self._analyzers[symbol]
    
    def on_trade(self, trade: TradeEvent) -> None:
        """Обработка новой сделки."""
        analyzer = self._analyzers.get(trade.symbol.upper())
        if analyzer is not None:
            analyzer.on_trade(trade)
    
    def snapshot(self, symbol: str) -> Optional[TapeMetrics]:
        """Возвращает метрики для символа."""
        analyzer = self._analyzers.get(symbol.upper())
        if analyzer is None:
            return None
        return analyzer.snapshot()
    
    def reset_all(self) -> None:
        """Сбрасывает все анализаторы."""
        for analyzer in self._analyzers.values():
            analyzer.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._analyzers.keys())