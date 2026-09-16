"""
Детектор режима рынка.

Определяет, торгуем ли мы вообще. Если режим не подходит
для стратегии пробоя — не торгуем.

Режимы:
- TRADING: условия для пробоя есть, торгуем
- RANGING: рынок в диапазоне без уровней, не торгуем
- TRENDING: трендовый рынок без консолидации, не торгуем
- VOLATILE: слишком волатильно, не торгуем
- LOW_VOLUME: низкий объём, не торгуем
- UNKNOWN: неизвестный режим

Используется в связке с:
- InstrumentSelector (отбор инструментов)
- RiskManager (фильтр режима рынка)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional


class MarketRegime(Enum):
    """Режим рынка."""
    TRADING = auto()        # Условия для пробоя есть, торгуем
    RANGING = auto()        # Рынок в диапазоне без уровней, не торгуем
    TRENDING = auto()       # Трендовый рынок без консолидации, не торгуем
    VOLATILE = auto()       # Слишком волатильно, не торгуем
    LOW_VOLUME = auto()     # Низкий объём, не торгуем
    UNKNOWN = auto()        # Неизвестный режим


@dataclass
class MarketRegimeConfig:
    """
    Конфигурация детектора режима рынка.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    # Волатильность
    min_volatility_pct: float = 0.3       # минимальная волатильность за час (%)
    max_volatility_pct: float = 5.0       # максимальная волатильность за час (%)
    
    # Объём
    min_volume_per_hour: float = 5_000_000  # минимальный объём за час ($)
    
    # Тренд
    trend_threshold: float = 0.6          # порог трендовости (0-1)
    
    # Консолидация
    min_consolidation_ratio: float = 0.3  # минимальное сжатие диапазона
    
    # Уровни
    min_level_strength: float = 3.0       # минимальная сила уровня


@dataclass
class MarketRegimeMetrics:
    """Метрики режима рынка."""
    ts_ns: int
    symbol: str
    
    # Волатильность
    volatility_pct: float = 0.0
    
    # Объём
    volume_per_hour: float = 0.0
    
    # Трендовость
    trend_strength: float = 0.0
    
    # Консолидация
    consolidation_ratio: float = 0.0
    
    # Уровни
    active_levels_count: int = 0
    max_level_strength: float = 0.0
    
    # Итоговый режим
    regime: MarketRegime = MarketRegime.UNKNOWN
    
    @property
    def is_trading_allowed(self) -> bool:
        """Торговля разрешена."""
        return self.regime == MarketRegime.TRADING


class MarketRegimeDetector:
    """
    Детектор режима рынка.
    
    Анализирует рыночные данные и определяет текущий режим.
    
    Использование:
        detector = MarketRegimeDetector("BTCUSDT")
        
        # Обновляем метрики
        detector.update(
            volatility_pct=1.5,
            volume_per_hour=50_000_000,
            trend_strength=0.3,
            consolidation_ratio=0.5,
            active_levels_count=5,
        )
        
        # Получаем режим
        regime = detector.get_regime()
        
        if regime == MarketRegime.TRADING:
            print("Торгуем!")
        else:
            print(f"Не торгуем: {regime.name}")
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[MarketRegimeConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.config = config or MarketRegimeConfig()
        
        # Текущие метрики
        self._metrics = MarketRegimeMetrics(
            ts_ns=time.time_ns(),
            symbol=symbol,
        )
    
    def update(
        self,
        volatility_pct: float = 0.0,
        volume_per_hour: float = 0.0,
        trend_strength: float = 0.0,
        consolidation_ratio: float = 0.0,
        active_levels_count: int = 0,
        max_level_strength: float = 0.0,
    ) -> MarketRegime:
        """
        Обновляет метрики и определяет режим.
        
        Вызывается периодически (например, каждую секунду).
        
        Возвращает определённый режим.
        """
        now_ns = time.time_ns()
        
        # Обновляем метрики
        self._metrics.ts_ns = now_ns
        self._metrics.volatility_pct = volatility_pct
        self._metrics.volume_per_hour = volume_per_hour
        self._metrics.trend_strength = trend_strength
        self._metrics.consolidation_ratio = consolidation_ratio
        self._metrics.active_levels_count = active_levels_count
        self._metrics.max_level_strength = max_level_strength
        
        # Определяем режим
        regime = self._determine_regime()
        self._metrics.regime = regime
        
        return regime
    
    def get_regime(self) -> MarketRegime:
        """Возвращает текущий режим."""
        return self._metrics.regime
    
    def get_metrics(self) -> MarketRegimeMetrics:
        """Возвращает текущие метрики."""
        return self._metrics
    
    def is_trading_allowed(self) -> bool:
        """Проверяет, разрешена ли торговля."""
        return self._metrics.regime == MarketRegime.TRADING
    
    def _determine_regime(self) -> MarketRegime:
        """Определяет режим на основе метрик."""
        # Проверка 1: Низкий объём
        if self._metrics.volume_per_hour < self.config.min_volume_per_hour:
            return MarketRegime.LOW_VOLUME
        
        # Проверка 2: Слишком высокая волатильность
        if self._metrics.volatility_pct > self.config.max_volatility_pct:
            return MarketRegime.VOLATILE
        
        # Проверка 3: Слишком низкая волатильность
        if self._metrics.volatility_pct < self.config.min_volatility_pct:
            return MarketRegime.RANGING
        
        # Проверка 4: Трендовый рынок
        if self._metrics.trend_strength > self.config.trend_threshold:
            return MarketRegime.TRENDING
        
        # Проверка 5: Нет уровней
        if self._metrics.active_levels_count == 0:
            return MarketRegime.RANGING
        
        # Проверка 6: Уровни слишком слабые
        if self._metrics.max_level_strength < self.config.min_level_strength:
            return MarketRegime.RANGING
        
        # Проверка 7: Нет консолидации
        if self._metrics.consolidation_ratio < self.config.min_consolidation_ratio:
            return MarketRegime.RANGING
        
        # Все проверки прошли — торгуем
        return MarketRegime.TRADING
    
    def reset(self) -> None:
        """Сбрасывает состояние детектора."""
        self._metrics = MarketRegimeMetrics(
            ts_ns=time.time_ns(),
            symbol=self.symbol,
        )


class MarketRegimeManager:
    """
    Менеджер детекторов режима рынка для множества символов.
    
    Использование:
        manager = MarketRegimeManager()
        
        # Создаём детектор для символа
        detector = manager.get_or_create("BTCUSDT")
        
        # Обновляем метрики
        manager.update("BTCUSDT", volatility_pct=1.5, volume_per_hour=50_000_000)
        
        # Проверяем, разрешена ли торговля
        if manager.is_trading_allowed("BTCUSDT"):
            print("Торгуем!")
    """
    
    def __init__(self):
        self._detectors: Dict[str, MarketRegimeDetector] = {}
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[MarketRegimeConfig] = None,
    ) -> MarketRegimeDetector:
        """Возвращает детектор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._detectors:
            self._detectors[symbol] = MarketRegimeDetector(
                symbol=symbol,
                config=config,
            )
        
        return self._detectors[symbol]
    
    def update(
        self,
        symbol: str,
        volatility_pct: float = 0.0,
        volume_per_hour: float = 0.0,
        trend_strength: float = 0.0,
        consolidation_ratio: float = 0.0,
        active_levels_count: int = 0,
        max_level_strength: float = 0.0,
    ) -> MarketRegime:
        """Обновляет метрики для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return MarketRegime.UNKNOWN
        
        return detector.update(
            volatility_pct=volatility_pct,
            volume_per_hour=volume_per_hour,
            trend_strength=trend_strength,
            consolidation_ratio=consolidation_ratio,
            active_levels_count=active_levels_count,
            max_level_strength=max_level_strength,
        )
    
    def is_trading_allowed(self, symbol: str) -> bool:
        """Проверяет, разрешена ли торговля для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return False
        return detector.is_trading_allowed()
    
    def get_regime(self, symbol: str) -> MarketRegime:
        """Возвращает режим для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return MarketRegime.UNKNOWN
        return detector.get_regime()
    
    def reset_all(self) -> None:
        """Сбрасывает все детекторы."""
        for detector in self._detectors.values():
            detector.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._detectors.keys())