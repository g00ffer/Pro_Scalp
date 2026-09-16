"""
Финальный генератор торговых сигналов.

Объединяет данные от всех детекторов:
- BreakoutDetector (попытки пробоя)
- ApproachDetector (подготовка к пробою)
- LevelDetector (уровни)
- CompressionDetector (консолидация)
- BookAnalyzer (ликвидность уровня)

Выдаёт финальный сигнал в ExecutionEngine через RiskManager.

Архитектура фильтров:
- Быстрые фильтры в SignalGenerator (до генерации)
- Полные проверки в RiskManager (после генерации)

Используется в связке с:
- RiskManager (фильтры рисков, размеры позиций)
- ExecutionEngine (исполнение сделок)
- DecisionJournal (логирование решений)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from proscalper.core.types import OrderSide
from proscalper.features.levels import Level
from proscalper.features.impulse_score import ImpulseScore, ImpulseStrength
from proscalper.features.compression import CompressionMetrics
from proscalper.features.book_analyzer import LevelLiquidity
from proscalper.signals.breakout import BreakoutSignal, BreakoutState
from proscalper.signals.approach import ApproachState, LevelApproach


class SignalType(Enum):
    """
    Тип торгового сигнала.
    
    В первой версии реализован только BREAKOUT.
    RETEST и FALSE_BREAKOUT будут добавлены позже.
    """
    BREAKOUT = auto()           # Пробой уровня
    RETEST = auto()             # Ретест уровня (после пробоя)
    FALSE_BREAKOUT = auto()     # Ложный пробой (контр-сигнал)


class SignalState(Enum):
    """
    Состояние сигнала.
    
    Жизненный цикл:
        PENDING → ACTIVE → EXECUTED / REJECTED / EXPIRED
    """
    PENDING = auto()        # Сигнал создан, ждёт проверки рисков
    ACTIVE = auto()         # Сигнал прошёл проверки, готов к исполнению
    EXECUTED = auto()       # Сигнал исполнен
    REJECTED = auto()       # Сигнал отклонён риск-менеджером
    EXPIRED = auto()        # Сигнал устарел


@dataclass
class SignalConfig:
    """
    Конфигурация генератора сигналов.
    
    Все пороги подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Ограничения
    max_active_signals: int = 1             # максимум активных сигналов
    signal_ttl_ms: int = 5000               # время жизни сигнала
    
    # Условия генерации
    require_approach_ready: bool = True     # требовать готовность подхода
    require_breakout_confirmed: bool = True # требовать подтверждение пробоя
    min_impulse_score: float = 0.4          # минимальный скор пробоя
    
    # Быстрые фильтры (до генерации сигнала)
    min_displacement_ticks: float = 3.0     # минимальное смещение за уровень
    min_volume_burst_ratio: float = 1.5     # минимальный всплеск объёма
    min_wall_consumption: float = 0.3       # минимальное проедание стены
    
    # Логирование
    include_full_context: bool = True       # включать полный контекст в сигнал


@dataclass
class TradingSignal:
    """
    Полная структура финального торгового сигнала.
    
    Содержит всю информацию для:
    - RiskManager (проверки рисков)
    - ExecutionEngine (исполнение сделки)
    - DecisionJournal (логирование решения)
    """
    # Идентификация
    signal_id: str
    signal_type: SignalType
    symbol: str
    
    # Уровень
    level_id: str
    level_center: float
    level_side: str  # "SUPPORT" или "RESISTANCE"
    
    # Направление и цена
    direction: OrderSide
    entry_price: float
    entry_ts_ns: int
    
    # Состояние сигнала
    state: SignalState = SignalState.PENDING
    
    # Оценки силы пробоя
    impulse_score: Optional[ImpulseScore] = None
    impulse_strength: ImpulseStrength = ImpulseStrength.WEAK
    
    # Состояние подхода
    approach_state: ApproachState = ApproachState.IDLE
    
    # Метрики консолидации
    compression_metrics: Optional[CompressionMetrics] = None
    
    # Ликвидность уровня
    level_liquidity: Optional[LevelLiquidity] = None
    
    # Метрики пробоя
    displacement_ticks: float = 0.0
    volume_burst_ratio: float = 1.0
    wall_consumption_ratio: float = 0.0
    imbalance: float = 0.0
    
    # Дополнительные метрики (для логирования)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_strong(self) -> bool:
        """Сильный пробой."""
        return self.impulse_strength == ImpulseStrength.STRONG
    
    @property
    def is_medium(self) -> bool:
        """Средний пробой."""
        return self.impulse_strength == ImpulseStrength.MEDIUM
    
    @property
    def is_weak(self) -> bool:
        """Слабый пробой."""
        return self.impulse_strength == ImpulseStrength.WEAK
    
    @property
    def is_buy(self) -> bool:
        """Сигнал на покупку."""
        return self.direction == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Сигнал на продажу."""
        return self.direction == OrderSide.SELL
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует сигнал в словарь для логирования.
        
        Используется в DecisionJournal для записи контекста решения.
        """
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.name,
            "symbol": self.symbol,
            "level_id": self.level_id,
            "level_center": self.level_center,
            "level_side": self.level_side,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "entry_ts_ns": self.entry_ts_ns,
            "state": self.state.name,
            "impulse_score": self.impulse_score.score if self.impulse_score else None,
            "impulse_strength": self.impulse_strength.name,
            "approach_state": self.approach_state.name,
            "displacement_ticks": self.displacement_ticks,
            "volume_burst_ratio": self.volume_burst_ratio,
            "wall_consumption_ratio": self.wall_consumption_ratio,
            "imbalance": self.imbalance,
            "metadata": self.metadata,
        }


class SignalGenerator:
    """
    Генератор торговых сигналов для одного символа.
    
    Работает на основе данных от:
    - BreakoutDetector (подтверждённые пробои)
    - ApproachDetector (состояния подхода)
    - Уровней (активные уровни)
    - Метрик консолидации и ликвидности
    
    Использование:
        generator = SignalGenerator("BTCUSDT", tick_size=0.1)
        
        # При получении подтверждённого пробоя
        signal = generator.generate_signal(
            breakout_signal=breakout,
            approach_state=approach,
            compression_metrics=compression,
            level_liquidity=liquidity,
        )
        
        if signal is not None:
            # Передаём сигнал в RiskManager
            risk_result = risk_manager.check_signal(signal)
            
            if risk_result.approved:
                # Передаём в ExecutionEngine
                execution_engine.execute(signal)
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[SignalConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or SignalConfig()
        
        # Активные сигналы
        self._active_signals: List[TradingSignal] = []
        
        # Счётчик сгенерированных сигналов (для статистики)
        self._total_generated: int = 0
        self._total_rejected: int = 0
    
    def generate_signal(
        self,
        breakout_signal: BreakoutSignal,
        approach_state: Optional[LevelApproach],
        compression_metrics: Optional[CompressionMetrics],
        level_liquidity: Optional[LevelLiquidity],
    ) -> Optional[TradingSignal]:
        """
        Генерирует финальный торговый сигнал.
        
        Вызывается при получении подтверждённого пробоя от BreakoutDetector.
        
        Args:
            breakout_signal: Подтверждённый сигнал пробоя
            approach_state: Состояние подхода к уровню
            compression_metrics: Метрики консолидации
            level_liquidity: Ликвидность уровня
            
        Returns:
            Финальный сигнал или None, если фильтры не прошли
        """
        now_ns = time.time_ns()
        
        # Проверяем лимит активных сигналов
        if len(self._active_signals) >= self.config.max_active_signals:
            self._total_rejected += 1
            return None
        
        # Проверяем быстрые фильтры
        if not self._check_fast_filters(breakout_signal, approach_state):
            self._total_rejected += 1
            return None
        
        # Определяем состояние подхода
        approach_state_enum = (
            approach_state.state if approach_state else ApproachState.IDLE
        )
        
        # Определяем силу пробоя
        impulse_strength = (
            breakout_signal.fast_score.strength
            if breakout_signal.fast_score else ImpulseStrength.WEAK
        )
        
        # Создаём финальный сигнал
        signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            signal_type=SignalType.BREAKOUT,
            symbol=self.symbol,
            level_id=breakout_signal.level_id,
            level_center=breakout_signal.level_center,
            level_side="RESISTANCE" if breakout_signal.direction == OrderSide.BUY else "SUPPORT",
            direction=breakout_signal.direction,
            entry_price=breakout_signal.entry_price,
            entry_ts_ns=now_ns,
            state=SignalState.PENDING,
            impulse_score=breakout_signal.fast_score,
            impulse_strength=impulse_strength,
            approach_state=approach_state_enum,
            compression_metrics=compression_metrics,
            level_liquidity=level_liquidity,
            displacement_ticks=breakout_signal.displacement_ticks,
            volume_burst_ratio=breakout_signal.volume_burst_ratio,
            wall_consumption_ratio=breakout_signal.wall_consumption_ratio,
            imbalance=(
                level_liquidity.imbalance if level_liquidity else 0.0
            ),
        )
        
        # Добавляем метаданные
        if self.config.include_full_context:
            signal.metadata = self._build_metadata(
                breakout_signal, approach_state, compression_metrics, level_liquidity
            )
        
        # Добавляем в активные сигналы
        self._active_signals.append(signal)
        self._total_generated += 1
        
        return signal
    
    def mark_signal_active(self, signal_id: str) -> None:
        """
        Помечает сигнал как активный (прошёл проверки рисков).
        
        Вызывается из RiskManager после успешной проверки.
        """
        for signal in self._active_signals:
            if signal.signal_id == signal_id:
                signal.state = SignalState.ACTIVE
                break
    
    def mark_signal_executed(self, signal_id: str) -> None:
        """
        Помечает сигнал как исполненный.
        
        Вызывается из ExecutionEngine после исполнения сделки.
        """
        for signal in self._active_signals:
            if signal.signal_id == signal_id:
                signal.state = SignalState.EXECUTED
                break
    
    def mark_signal_rejected(self, signal_id: str) -> None:
        """
        Помечает сигнал как отклонённый.
        
        Вызывается из RiskManager при отклонении сигнала.
        """
        for signal in self._active_signals:
            if signal.signal_id == signal_id:
                signal.state = SignalState.REJECTED
                break
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Возвращает все активные сигналы."""
        return [
            signal for signal in self._active_signals
            if signal.state in (SignalState.PENDING, SignalState.ACTIVE)
        ]
    
    def get_signal(self, signal_id: str) -> Optional[TradingSignal]:
        """Возвращает сигнал по ID."""
        for signal in self._active_signals:
            if signal.signal_id == signal_id:
                return signal
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику генератора."""
        return {
            "total_generated": self._total_generated,
            "total_rejected": self._total_rejected,
            "active_count": len(self.get_active_signals()),
        }
    
    def reset(self) -> None:
        """Сбрасывает состояние генератора."""
        self._active_signals.clear()
        self._total_generated = 0
        self._total_rejected = 0
    
    def _check_fast_filters(
        self,
        breakout_signal: BreakoutSignal,
        approach_state: Optional[LevelApproach],
    ) -> bool:
        """
        Быстрые фильтры перед генерацией сигнала.
        
        Это минимальные проверки, которые можно сделать быстро.
        Полные проверки делаются в RiskManager.
        
        Returns:
            True если все фильтры прошли
        """
        # Фильтр 1: Проверка минимального скора пробоя
        if breakout_signal.fast_score is not None:
            if breakout_signal.fast_score.score < self.config.min_impulse_score:
                return False
        
        # Фильтр 2: Проверка минимального смещения за уровень
        if breakout_signal.displacement_ticks < self.config.min_displacement_ticks:
            return False
        
        # Фильтр 3: Проверка минимального всплеска объёма
        if breakout_signal.volume_burst_ratio < self.config.min_volume_burst_ratio:
            return False
        
        # Фильтр 4: Проверка минимального проедания стены
        if breakout_signal.wall_consumption_ratio < self.config.min_wall_consumption:
            return False
        
        # Фильтр 5: Проверка готовности подхода (если требуется)
        if self.config.require_approach_ready:
            if approach_state is not None:
                if approach_state.state not in (
                    ApproachState.READY,
                    ApproachState.BREAKING,
                    ApproachState.CONFIRMED,
                ):
                    return False
        
        # Все фильтры прошли
        return True
    
    def _build_metadata(
        self,
        breakout_signal: BreakoutSignal,
        approach_state: Optional[LevelApproach],
        compression_metrics: Optional[CompressionMetrics],
        level_liquidity: Optional[LevelLiquidity],
    ) -> Dict[str, Any]:
        """
        Строит метаданные для логирования.
        
        Включает полный контекст решения для DecisionJournal.
        """
        metadata: Dict[str, Any] = {
            "breakout": {
                "level_center": breakout_signal.level_center,
                "displacement_ticks": breakout_signal.displacement_ticks,
                "volume_burst_ratio": breakout_signal.volume_burst_ratio,
                "wall_consumption_ratio": breakout_signal.wall_consumption_ratio,
            },
        }
        
        # Добавляем состояние подхода
        if approach_state is not None:
            metadata["approach"] = {
                "state": approach_state.state.name,
                "distance_to_level_ticks": approach_state.distance_to_level_ticks,
                "time_in_state_ms": approach_state.time_in_state_ms,
                "has_compression": approach_state.has_compression,
                "has_wall": approach_state.has_wall,
            }
        
        # Добавляем метрики консолидации
        if compression_metrics is not None:
            metadata["compression"] = {
                "compression_ratio": compression_metrics.compression_ratio,
                "range_short": compression_metrics.range_short,
                "range_long": compression_metrics.range_long,
                "volume_ratio": compression_metrics.volume_ratio,
                "is_compressing": compression_metrics.is_compressing,
            }
        
        # Добавляем ликвидность уровня
        if level_liquidity is not None:
            metadata["liquidity"] = {
                "support_score": level_liquidity.support_score,
                "pressure_score": level_liquidity.pressure_score,
                "imbalance": level_liquidity.imbalance,
                "walls_to_break_count": len(level_liquidity.walls_to_break),
            }
        
        return metadata
    
    def _prune_expired_signals(self, now_ns: int) -> None:
        """Удаляет устаревшие сигналы."""
        ttl_ns = self.config.signal_ttl_ms * 1_000_000
        
        self._active_signals = [
            signal for signal in self._active_signals
            if (now_ns - signal.entry_ts_ns) < ttl_ns
            and signal.state in (SignalState.PENDING, SignalState.ACTIVE, SignalState.EXECUTED)
        ]


class SignalGeneratorManager:
    """
    Менеджер SignalGenerator'ов для множества символов.
    
    Использование:
        manager = SignalGeneratorManager(tick_sizes={"BTCUSDT": 0.1})
        
        # Получаем генератор для символа
        generator = manager.get_or_create("BTCUSDT")
        
        # Генерируем сигнал при подтверждённом пробое
        signal = manager.generate_signal(
            "BTCUSDT", breakout, approach, compression, liquidity
        )
        
        if signal is not None:
            # Передаём в RiskManager
            risk_result = risk_manager.check_signal(signal)
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._generators: Dict[str, SignalGenerator] = {}
        self._tick_sizes = tick_sizes
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[SignalConfig] = None,
    ) -> SignalGenerator:
        """Возвращает генератор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._generators:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._generators[symbol] = SignalGenerator(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        
        return self._generators[symbol]
    
    def generate_signal(
        self,
        symbol: str,
        breakout_signal: BreakoutSignal,
        approach_state: Optional[LevelApproach],
        compression_metrics: Optional[CompressionMetrics],
        level_liquidity: Optional[LevelLiquidity],
    ) -> Optional[TradingSignal]:
        """Генерация сигнала для нужного символа."""
        generator = self._generators.get(symbol.upper())
        if generator is None:
            return None
        return generator.generate_signal(
            breakout_signal, approach_state, compression_metrics, level_liquidity
        )
    
    def get_active_signals(self, symbol: str) -> List[TradingSignal]:
        """Возвращает активные сигналы для символа."""
        generator = self._generators.get(symbol.upper())
        if generator is None:
            return []
        return generator.get_active_signals()
    
    def get_stats(self, symbol: str) -> Dict[str, int]:
        """Возвращает статистику генератора для символа."""
        generator = self._generators.get(symbol.upper())
        if generator is None:
            return {}
        return generator.get_stats()
    
    def reset_all(self) -> None:
        """Сбрасывает все генераторы."""
        for generator in self._generators.values():
            generator.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._generators.keys())