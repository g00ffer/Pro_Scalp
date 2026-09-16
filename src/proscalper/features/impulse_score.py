"""
Оценка силы импульса / пробоя уровня.

Критичный модуль для двухэтапной оценки пробоя:
- Быстрая оценка (0-300мс) → решение о ВХОДЕ
- Полная оценка (1-3сек) → судьба УРОВНЯ (BROKEN / RETESTABLE / FAILED_BREAK)

Метрики берутся из трёх источников:
- Лента: всплеск объёма, дельта, скорость сделок
- Стакан: проедание стен, дисбаланс
- Движение цены: смещение за уровень, удержание, возврат

Используется в связке с:
- ApproachDetector (подготовка к пробою)
- BreakoutDetector (сигнал пробоя)
- LevelTracker (судьба уровня после пробоя)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class ImpulseStrength(Enum):
    """
    Градация силы пробоя.
    
    Используется для определения судьбы уровня:
    - WEAK → FAILED_BREAK (ложный пробой / закол)
    - MEDIUM → RETESTABLE (ждём ретест)
    - STRONG → BROKEN (уровень сломан)
    """
    WEAK = auto()       # Слабый пробой (ложный пробой / закол)
    MEDIUM = auto()     # Средний пробой (ретест)
    STRONG = auto()     # Сильный пробой (уровень сломан)


@dataclass
class ImpulseMetrics:
    """
    Сырые метрики импульса из трёх источников данных.
    
    Заполняются из:
    - TapeAnalyzer (лента)
    - BookAnalyzer (стакан)
    - Текущей цены (движение)
    """
    ts_ns: int
    symbol: str
    
    # === Метрики из ленты ===
    
    # Всплеск объёма (текущий объём / медиана)
    # 1.0 = нет всплеска, 2.0 = двойной всплеск
    volume_burst_ratio: float = 1.0
    
    # Дельта за окно (покупки - продажи)
    net_delta: float = 0.0
    
    # Скорость сделок (сделок в секунду)
    trade_rate: float = 0.0
    
    # === Метрики из стакана ===
    
    # Доля проедания стены (0 = не проедена, 1 = полностью)
    wall_consumption_ratio: float = 0.0
    
    # Дисбаланс в зоне уровня [-1, 1]
    # +1 = давление вверх, -1 = давление вниз
    imbalance: float = 0.0
    
    # === Метрики движения цены ===
    
    # Смещение цены за уровень (в тиках)
    price_displacement_ticks: float = 0.0
    
    # Время удержания за уровнем (мс)
    hold_duration_ms: int = 0
    
    # Скорость возврата к уровню (тиков/сек)
    # 0 = нет возврата, >0 = цена возвращается
    reversion_speed: float = 0.0
    
    # === Производные ===
    
    @property
    def has_volume_burst(self) -> bool:
        """Есть ли всплеск объёма."""
        return self.volume_burst_ratio > 1.5
    
    @property
    def has_wall_consumed(self) -> bool:
        """Проедена ли стена."""
        return self.wall_consumption_ratio > 0.5
    
    @property
    def has_displacement(self) -> bool:
        """Есть ли смещение за уровень."""
        return abs(self.price_displacement_ticks) > 3
    
    @property
    def is_reverting(self) -> bool:
        """Цена возвращается к уровню (ложный пробой)."""
        return self.reversion_speed > 0.5


@dataclass
class ImpulseScoreConfig:
    """
    Конфигурация расчёта силы импульса.
    
    Все пороги подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Временные окна
    fast_window_ms: int = 300       # окно быстрого режима (для входа)
    full_window_ms: int = 2000      # окно полного режима (для судьбы уровня)
    
    # Пороги градаций
    medium_threshold: float = 0.4   # score >= 0.4 → MEDIUM
    strong_threshold: float = 0.7   # score >= 0.7 → STRONG
    
    # Веса метрик (сумма должна быть ~1.0)
    # Лента
    w_volume_burst: float = 0.25
    w_net_delta: float = 0.15
    w_trade_rate: float = 0.10
    
    # Стакан
    w_wall_consumption: float = 0.20
    w_imbalance: float = 0.10
    
    # Движение цены
    w_price_displacement: float = 0.15
    w_hold_duration: float = 0.05
    
    # Нормализация метрик (мин/макс для приведения к [0,1])
    volume_burst_max: float = 5.0       # всплеск 5x = максимум
    net_delta_max: float = 100000.0     # дельта 100k = максимум
    trade_rate_max: float = 500.0       # 500 сделок/сек = максимум
    displacement_max_ticks: float = 20.0  # 20 тиков = максимум
    hold_duration_max_ms: int = 1000    # 1 секунда удержания = максимум


@dataclass
class ImpulseScore:
    """
    Результат оценки силы импульса.
    
    Содержит итоговый скор [0, 1] и градацию силы.
    """
    ts_ns: int
    symbol: str
    
    # Итоговый скор [0, 1]
    score: float = 0.0
    
    # Градация силы
    strength: ImpulseStrength = ImpulseStrength.WEAK
    
    # Временной режим
    is_fast_mode: bool = True
    
    # Детализация по метрикам (для логирования)
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_strong(self) -> bool:
        """Сильный пробой."""
        return self.strength == ImpulseStrength.STRONG
    
    @property
    def is_medium(self) -> bool:
        """Средний пробой."""
        return self.strength == ImpulseStrength.MEDIUM
    
    @property
    def is_weak(self) -> bool:
        """Слабый пробой (ложный)."""
        return self.strength == ImpulseStrength.WEAK


class ImpulseScoreCalculator:
    """
    Калькулятор силы импульса.
    
    Реализует двухэтапную оценку пробоя:
    1. Быстрый режим (0-300мс) — для решения о входе
    2. Полный режим (1-3сек) — для судьбы уровня
    
    Использование:
        calculator = ImpulseScoreCalculator("BTCUSDT")
        
        # Быстрая оценка (для входа)
        fast_score = calculator.calculate_fast(metrics)
        
        if fast_score.score >= 0.4:
            print("Входим в пробой!")
        
        # Полная оценка (для судьбы уровня)
        full_score = calculator.calculate_full(metrics)
        
        if full_score.is_strong:
            level.state = BROKEN
        elif full_score.is_medium:
            level.state = RETESTABLE
        else:
            level.state = FAILED_BREAK
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[ImpulseScoreConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.config = config or ImpulseScoreConfig()
    
    def calculate_fast(self, metrics: ImpulseMetrics) -> ImpulseScore:
        """
        Быстрая оценка силы импульса (0-300мс).
        
        Используется для принятия решения о ВХОДЕ.
        Фокус на метриках ленты и стакана.
        """
        now_ns = metrics.ts_ns or time.time_ns()
        
        # Нормализуем метрики к [0, 1]
        volume_burst_norm = self._normalize(
            metrics.volume_burst_ratio, 1.0, self.config.volume_burst_max
        )
        net_delta_norm = self._normalize(
            abs(metrics.net_delta), 0.0, self.config.net_delta_max
        )
        trade_rate_norm = self._normalize(
            metrics.trade_rate, 0.0, self.config.trade_rate_max
        )
        wall_consumption_norm = self._normalize(
            metrics.wall_consumption_ratio, 0.0, 1.0
        )
        imbalance_norm = self._normalize(
            abs(metrics.imbalance), 0.0, 1.0
        )
        
        # В быстром режиме не используем метрики движения цены
        # (они ещё не сформировались)
        displacement_norm = 0.0
        hold_duration_norm = 0.0
        
        # Рассчитываем итоговый скор
        score = (
            self.config.w_volume_burst * volume_burst_norm
            + self.config.w_net_delta * net_delta_norm
            + self.config.w_trade_rate * trade_rate_norm
            + self.config.w_wall_consumption * wall_consumption_norm
            + self.config.w_imbalance * imbalance_norm
            + self.config.w_price_displacement * displacement_norm
            + self.config.w_hold_duration * hold_duration_norm
        )
        
        # Нормализуем на сумму использованных весов
        used_weights_sum = (
            self.config.w_volume_burst
            + self.config.w_net_delta
            + self.config.w_trade_rate
            + self.config.w_wall_consumption
            + self.config.w_imbalance
        )
        if used_weights_sum > 0:
            score = score / used_weights_sum
        
        # Определяем градацию
        strength = self._determine_strength(score)
        
        return ImpulseScore(
            ts_ns=now_ns,
            symbol=self.symbol,
            score=score,
            strength=strength,
            is_fast_mode=True,
            component_scores={
                "volume_burst": volume_burst_norm,
                "net_delta": net_delta_norm,
                "trade_rate": trade_rate_norm,
                "wall_consumption": wall_consumption_norm,
                "imbalance": imbalance_norm,
            },
        )
    
    def calculate_full(self, metrics: ImpulseMetrics) -> ImpulseScore:
        """
        Полная оценка силы импульса (1-3сек).
        
        Используется для определения СУДЬБЫ УРОВНЯ.
        Учитывает все метрики, включая движение цены.
        """
        now_ns = metrics.ts_ns or time.time_ns()
        
        # Нормализуем все метрики к [0, 1]
        volume_burst_norm = self._normalize(
            metrics.volume_burst_ratio, 1.0, self.config.volume_burst_max
        )
        net_delta_norm = self._normalize(
            abs(metrics.net_delta), 0.0, self.config.net_delta_max
        )
        trade_rate_norm = self._normalize(
            metrics.trade_rate, 0.0, self.config.trade_rate_max
        )
        wall_consumption_norm = self._normalize(
            metrics.wall_consumption_ratio, 0.0, 1.0
        )
        imbalance_norm = self._normalize(
            abs(metrics.imbalance), 0.0, 1.0
        )
        displacement_norm = self._normalize(
            abs(metrics.price_displacement_ticks),
            0.0,
            self.config.displacement_max_ticks,
        )
        hold_duration_norm = self._normalize(
            metrics.hold_duration_ms, 0.0, self.config.hold_duration_max_ms
        )
        
        # Рассчитываем итоговый скор
        score = (
            self.config.w_volume_burst * volume_burst_norm
            + self.config.w_net_delta * net_delta_norm
            + self.config.w_trade_rate * trade_rate_norm
            + self.config.w_wall_consumption * wall_consumption_norm
            + self.config.w_imbalance * imbalance_norm
            + self.config.w_price_displacement * displacement_norm
            + self.config.w_hold_duration * hold_duration_norm
        )
        
        # Штраф за возврат цены (ложный пробой)
        if metrics.is_reverting:
            reversion_penalty = self._normalize(
                metrics.reversion_speed, 0.0, 2.0
            )
            score = score * (1.0 - 0.5 * reversion_penalty)
        
        # Определяем градацию
        strength = self._determine_strength(score)
        
        return ImpulseScore(
            ts_ns=now_ns,
            symbol=self.symbol,
            score=score,
            strength=strength,
            is_fast_mode=False,
            component_scores={
                "volume_burst": volume_burst_norm,
                "net_delta": net_delta_norm,
                "trade_rate": trade_rate_norm,
                "wall_consumption": wall_consumption_norm,
                "imbalance": imbalance_norm,
                "displacement": displacement_norm,
                "hold_duration": hold_duration_norm,
            },
        )
    
    def _normalize(
        self,
        value: float,
        min_value: float,
        max_value: float,
    ) -> float:
        """
        Нормализует значение к диапазону [0, 1].
        
        Если значение вне диапазона - обрезается.
        """
        if max_value <= min_value:
            return 0.0
        
        normalized = (value - min_value) / (max_value - min_value)
        return max(0.0, min(1.0, normalized))
    
    def _determine_strength(self, score: float) -> ImpulseStrength:
        """Определяет градацию силы по скору."""
        if score >= self.config.strong_threshold:
            return ImpulseStrength.STRONG
        elif score >= self.config.medium_threshold:
            return ImpulseStrength.MEDIUM
        else:
            return ImpulseStrength.WEAK


class ImpulseScoreManager:
    """
    Менеджер ImpulseScoreCalculator'ов для множества символов.
    
    Использование:
        manager = ImpulseScoreManager()
        
        # Получаем калькулятор для символа
        calculator = manager.get_or_create("BTCUSDT")
        
        # Быстрая оценка (для входа)
        fast_score = manager.calculate_fast("BTCUSDT", metrics)
        
        # Полная оценка (для судьбы уровня)
        full_score = manager.calculate_full("BTCUSDT", metrics)
    """
    
    def __init__(self):
        self._calculators: Dict[str, ImpulseScoreCalculator] = {}
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[ImpulseScoreConfig] = None,
    ) -> ImpulseScoreCalculator:
        """Возвращает калькулятор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._calculators:
            self._calculators[symbol] = ImpulseScoreCalculator(
                symbol=symbol,
                config=config,
            )
        
        return self._calculators[symbol]
    
    def calculate_fast(
        self,
        symbol: str,
        metrics: ImpulseMetrics,
    ) -> Optional[ImpulseScore]:
        """Быстрая оценка силы импульса."""
        calculator = self._calculators.get(symbol.upper())
        if calculator is None:
            return None
        return calculator.calculate_fast(metrics)
    
    def calculate_full(
        self,
        symbol: str,
        metrics: ImpulseMetrics,
    ) -> Optional[ImpulseScore]:
        """Полная оценка силы импульса."""
        calculator = self._calculators.get(symbol.upper())
        if calculator is None:
            return None
        return calculator.calculate_full(metrics)
    
    def reset_all(self) -> None:
        """Сбрасывает все калькуляторы."""
        self._calculators.clear()
    
    def all_symbols(self) -> list:
        """Возвращает список всех символов."""
        return list(self._calculators.keys())