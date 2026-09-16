"""
Базовые классы для сигнальных движков.

Определяет единый интерфейс для всех типов сетапов:
- BreakoutEngine (пробой уровня)
- RetestEngine (ретест уровня после пробоя)
- FalseBreakoutEngine (ложный пробой / закол)

Архитектура:
- Каждый движок наследуется от BaseSignalEngine
- Все движки получают единый MarketContext
- Все движки возвращают единый EngineResult
- SignalGenerator агрегирует результаты всех движков

Разделение ответственности:
- base.py — абстракции и контракты
- breakout.py — логика пробоя
- retest.py — логика ретеста
- false_breakout.py — логика ложного пробоя
- filters.py — фильтры сигналов
- signal_generator.py — оркестрация движков
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.core.types import OrderSide, SignalRejectReason
from proscalper.features.levels import Level, LevelState


# ============================================================
# Типы сетапов
# ============================================================

class SetupType(Enum):
    """
    Тип торгового сетапа.
    
    Каждый сетап имеет отдельный движок:
    - BREAKOUT: пробой уровня после консолидации
    - RETEST: ретест уровня после пробоя
    - FALSE_BREAKOUT: ложный пробой (контр-сигнал)
    """
    BREAKOUT = auto()
    RETEST = auto()
    FALSE_BREAKOUT = auto()


class EngineState(Enum):
    """
    Состояние сигнального движка.
    
    Используется для диагностики и мониторинга.
    """
    IDLE = auto()           # Движок не имеет активных попыток
    TRACKING = auto()       # Движок отслеживает потенциал сетапа
    CONFIRMING = auto()     # Движок подтверждает сетап
    COOLDOWN = auto()       # Движок в кулдауне после сетапа
    DISABLED = auto()       # Движок отключён (например, из-за лимитов)


# ============================================================
# Контекст рынка (входные данные для движков)
# ============================================================

@dataclass
class MarketContext:
    """
    Агрегированный контекст рынка для сигнальных движков.
    
    Содержит все данные, необходимые для принятия решения.
    Формируется в SignalGenerator перед вызовом движков.
    
    Источники данных:
    - levels: LevelManager (активные уровни)
    - tape_metrics: TapeAnalyzer (дельта, объём, скорость)
    - book_liquidity: BookAnalyzer (ликвидность вокруг уровня)
    - compression: CompressionDetector (консолидация)
    - fast_state: BookTickerGuard (быстрые цены)
    """
    # Время и символ
    ts_ns: int
    symbol: str
    
    # Текущие цены (из bookTicker)
    last_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    spread_ticks: int = 0
    
    # Активные уровни
    active_levels: List[Level] = field(default_factory=list)
    
    # Ближайший уровень (для быстрых проверок)
    nearest_level: Optional[Level] = None
    distance_to_nearest_level_ticks: float = 0.0
    
    # Метрики ленты (из TapeAnalyzer)
    tape_metrics: Optional[object] = None
    
    # Ликвидность стакана (из BookAnalyzer)
    book_liquidity: Optional[object] = None
    
    # Метрики компрессии (из CompressionDetector)
    compression_metrics: Optional[object] = None
    
    # Метрики режима рынка (из MarketRegimeDetector)
    market_regime: Optional[object] = None
    
    # Возраст данных (для проверки актуальности)
    bookticker_age_ms: int = 0
    book_lag_ms: int = 0
    
    @property
    def mid_price(self) -> float:
        """Средняя цена."""
        if self.bid_price <= 0 or self.ask_price <= 0:
            return self.last_price
        return (self.bid_price + self.ask_price) / 2.0
    
    @property
    def data_fresh(self) -> bool:
        """Данные свежие (без больших лагов)."""
        return self.book_lag_ms < 300 and self.bookticker_age_ms < 80
    
    def get_levels_by_state(self, state: LevelState) -> List[Level]:
        """Возвращает уровни в указанном состоянии."""
        return [
            level for level in self.active_levels
            if level.state == state
        ]
    
    def get_levels_in_range(
        self,
        min_price: float,
        max_price: float,
    ) -> List[Level]:
        """Возвращает уровни в ценовом диапазоне."""
        return [
            level for level in self.active_levels
            if min_price <= level.center <= max_price
        ]


# ============================================================
# Результат работы движка
# ============================================================

@dataclass
class EngineResult:
    """
    Результат работы сигнального движка.
    
    Может содержать:
    - Сигнал (если сетап подтверждён)
    - Отклонение (если сетап не подтверждён)
    - Ничего (если движок не видит потенциала)
    
    Используется в SignalGenerator для агрегации результатов.
    """
    # Тип результата
    has_signal: bool = False
    has_rejection: bool = False
    
    # Параметры сигнала (если есть)
    setup_type: Optional[SetupType] = None
    side: Optional[OrderSide] = None
    level: Optional[Level] = None
    
    # Цены (если есть сигнал)
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    
    # Причины (для журнала решений)
    reasons: List[str] = field(default_factory=list)
    rejection_reason: Optional[SignalRejectReason] = None
    rejection_details: str = ""
    
    # Сила сетапа (для скоринга)
    confidence: float = 0.0
    
    # Снимок фичей (для журнала)
    features_snapshot: Dict = field(default_factory=dict)
    
    # Время создания
    created_ts_ns: int = 0
    
    @staticmethod
    def no_setup() -> "EngineResult":
        """Движок не видит потенциала сетапа."""
        return EngineResult(
            has_signal=False,
            has_rejection=False,
        )
    
    @staticmethod
    def signal(
        setup_type: SetupType,
        side: OrderSide,
        level: Level,
        entry_price: float,
        stop_price: float,
        reasons: List[str],
        confidence: float = 0.0,
        take_profit_price: float = 0.0,
        features_snapshot: Optional[Dict] = None,
        created_ts_ns: int = 0,
    ) -> "EngineResult":
        """Создаёт результат с сигналом."""
        return EngineResult(
            has_signal=True,
            has_rejection=False,
            setup_type=setup_type,
            side=side,
            level=level,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            reasons=reasons,
            confidence=confidence,
            features_snapshot=features_snapshot or {},
            created_ts_ns=created_ts_ns,
        )
    
    @staticmethod
    def rejection(
        reason: SignalRejectReason,
        details: str = "",
    ) -> "EngineResult":
        """Создаёт результат с отклонением."""
        return EngineResult(
            has_signal=False,
            has_rejection=True,
            rejection_reason=reason,
            rejection_details=details,
        )


# ============================================================
# Базовая конфигурация движка
# ============================================================

@dataclass
class BaseEngineConfig:
    """
    Базовая конфигурация сигнального движка.
    
    Конкретные движки (breakout, retest) наследуют
    и расширяют эту конфигурацию.
    """
    # Включение/отключение движка
    enabled: bool = True
    
    # Максимальное количество одновременных попыток
    max_concurrent_attempts: int = 3
    
    # Кулдаун после сетапа (секунды)
    cooldown_after_signal_sec: float = 30.0
    
    # Кулдаун после ложного сетапа (секунды)
    cooldown_after_false_sec: float = 60.0
    
    # Минимальный интервал между сигналами по одному уровню (секунды)
    min_interval_between_signals_sec: float = 120.0


# ============================================================
# Базовый класс сигнального движка
# ============================================================

class BaseSignalEngine(ABC):
    """
    Абстрактный базовый класс для всех сигнальных движков.
    
    Каждый движок отвечает за один тип сетапа:
    - BreakoutEngine: пробой уровня
    - RetestEngine: ретест уровня
    - FalseBreakoutEngine: ложный пробой
    
    Жизненный цикл движка:
    1. Получает MarketContext от SignalGenerator
    2. Проверяет, видит ли потенциал сетапа
    3. Если видит — отслеживает и подтверждает
    4. Возвращает EngineResult (сигнал, отклонение или ничего)
    
    Использование:
        engine = BreakoutEngine(config, tick_size)
        
        # На каждом обновлении рынка
        result = engine.evaluate(context)
        
        if result.has_signal:
            signal_generator.process_signal(result)
        elif result.has_rejection:
            journal.log_rejection(result)
    """
    
    def __init__(
        self,
        config: BaseEngineConfig,
        tick_size: float,
    ):
        self._config = config
        self._tick_size = tick_size
        
        # Состояние движка
        self._state = EngineState.IDLE
        
        # Время последнего сигнала (для кулдауна)
        self._last_signal_ts_ns: Dict[str, int] = {}
        
        # Статистика
        self._total_evaluations: int = 0
        self._total_signals: int = 0
        self._total_rejections: int = 0
    
    @property
    @abstractmethod
    def setup_type(self) -> SetupType:
        """Тип сетапа, за который отвечает движок."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Имя движка для логирования."""
        ...
    
    @property
    def state(self) -> EngineState:
        """Текущее состояние движка."""
        return self._state
    
    @property
    def is_enabled(self) -> bool:
        """Движок включён."""
        return self._config.enabled and self._state != EngineState.DISABLED
    
    @abstractmethod
    def evaluate(self, context: MarketContext) -> EngineResult:
        """
        Основная точка входа для движка.
        
        Вызывается на каждом обновлении рынка.
        Возвращает результат оценки (сигнал, отклонение или ничего).
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """
        Сбрасывает состояние движка.
        
        Вызывается при:
        - Переподключении к бирже
        - Аварийной остановке
        - Смене торговой сессии
        """
        ...
    
    def disable(self) -> None:
        """Отключает движок."""
        self._state = EngineState.DISABLED
    
    def enable(self) -> None:
        """Включает движок."""
        if self._state == EngineState.DISABLED:
            self._state = EngineState.IDLE
    
    # ============================================
    # Защищённые методы для наследников
    # ============================================
    
    def _can_emit_signal(
        self,
        level_id: str,
        now_ns: int,
    ) -> bool:
        """
        Проверяет, можно ли отправить сигнал для уровня.
        
        Учитывает:
        - Кулдаун после последнего сигнала
        - Минимальный интервал между сигналами
        """
        if not self.is_enabled:
            return False
        
        last_ts = self._last_signal_ts_ns.get(level_id, 0)
        if last_ts == 0:
            return True
        
        interval_sec = (now_ns - last_ts) / 1_000_000_000
        return interval_sec >= self._config.min_interval_between_signals_sec
    
    def _mark_signal_emitted(
        self,
        level_id: str,
        now_ns: int,
    ) -> None:
        """Помечает, что сигнал был отправлен."""
        self._last_signal_ts_ns[level_id] = now_ns
        self._total_signals += 1
    
    def _mark_rejection(self) -> None:
        """Помечает отклонение."""
        self._total_rejections += 1
    
    def _mark_evaluation(self) -> None:
        """Помечает оценку."""
        self._total_evaluations += 1
    
    def get_stats(self) -> Dict:
        """Возвращает статистику движка."""
        return {
            "name": self.name,
            "setup_type": self.setup_type.name,
            "state": self.state.name,
            "enabled": self.is_enabled,
            "total_evaluations": self._total_evaluations,
            "total_signals": self._total_signals,
            "total_rejections": self._total_rejections,
            "signal_rate": (
                self._total_signals / self._total_evaluations
                if self._total_evaluations > 0 else 0.0
            ),
        }


# ============================================================
# Менеджер сигнальных движков
# ============================================================

class SignalEngineRegistry:
    """
    Реестр сигнальных движков для одного символа.
    
    Хранит все движки (breakout, retest, false_breakout)
    и предоставляет единый интерфейс для SignalGenerator.
    
    Использование:
        registry = SignalEngineRegistry(symbol="BTCUSDT")
        
        # Регистрируем движки
        registry.register(breakout_engine)
        registry.register(retest_engine)
        registry.register(false_breakout_engine)
        
        # Оцениваем все движки
        results = registry.evaluate_all(context)
        
        # Находим лучший сигнал
        best = registry.get_best_signal(results)
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self._engines: Dict[SetupType, BaseSignalEngine] = {}
    
    def register(self, engine: BaseSignalEngine) -> None:
        """Регистрирует движок."""
        self._engines[engine.setup_type] = engine
    
    def unregister(self, setup_type: SetupType) -> None:
        """Удаляет движок."""
        self._engines.pop(setup_type, None)
    
    def get_engine(self, setup_type: SetupType) -> Optional[BaseSignalEngine]:
        """Возвращает движок по типу сетапа."""
        return self._engines.get(setup_type)
    
    def evaluate_all(self, context: MarketContext) -> List[EngineResult]:
        """
        Оценивает все зарегистрированные движки.
        
        Возвращает список результатов.
        """
        results: List[EngineResult] = []
        
        for engine in self._engines.values():
            if not engine.is_enabled:
                continue
            
            try:
                result = engine.evaluate(context)
                results.append(result)
            except Exception:
                # Ошибка в движке не должна ронять систему
                continue
        
        return results
    
    def get_best_signal(
        self,
        results: List[EngineResult],
    ) -> Optional[EngineResult]:
        """
        Возвращает лучший сигнал из списка результатов.
        
        Критерий: максимальная уверенность (confidence).
        """
        signals = [r for r in results if r.has_signal]
        
        if not signals:
            return None
        
        return max(signals, key=lambda r: r.confidence)
    
    def get_all_rejections(
        self,
        results: List[EngineResult],
    ) -> List[EngineResult]:
        """Возвращает все отклонения из списка результатов."""
        return [r for r in results if r.has_rejection]
    
    def reset_all(self) -> None:
        """Сбрасывает все движки."""
        for engine in self._engines.values():
            engine.reset()
    
    def get_stats(self) -> Dict:
        """Возвращает статистику всех движков."""
        return {
            "symbol": self.symbol,
            "engines": {
                setup_type.name: engine.get_stats()
                for setup_type, engine in self._engines.items()
            },
        }
    
    @property
    def engine_count(self) -> int:
        """Количество зарегистрированных движков."""
        return len(self._engines)
    
    @property
    def enabled_engine_count(self) -> int:
        """Количество включённых движков."""
        return sum(
            1 for engine in self._engines.values()
            if engine.is_enabled
        )