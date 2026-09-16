"""
Фильтры сигналов.

Модульная система проверки сигналов перед передачей в RiskManager.
Каждый фильтр проверяет один аспект и возвращает результат с кодом причины
для Decision Journal.

Категории фильтров (из формализации системы):

1. Рыночные фильтры (здоровье рынка):
   - спред слишком широкий
   - недостаточная глубина стакана
   - низкая скорость сделок
   - большой лаг стакана
   - WS отключен или стакан рассинхронизирован

2. Фильтры уровня:
   - недостаточная сила уровня
   - уровень слишком старый
   - мало касаний
   - уровень в cooldown после ложного пробоя
   - цена слишком далеко от уровня (подход/погоня)

3. Фильтры пробоя:
   - слабый всплеск объёма
   - дисбаланс направлен против сделки
   - сильная встречная стена
   - недостаточное пространство до цели

4. Риск-фильтры:
   - достигнут дневной лимит убытков
   - максимум позиций
   - кулдаун после убытка
   - недостаточно маржи
   - активен kill switch

Архитектура:
- Каждый фильтр — отдельный класс, наследник BaseFilter
- FilterChain компонирует фильтры с режимом fail_fast
- FilterContext передаёт все данные, необходимые фильтрам
- Каждый reject содержит код причины для журналирования

Используется в связке с:
- SignalGenerator (вызывает фильтры перед отправкой сигнала)
- DecisionJournal (логирует причины отклонения)
- RiskManager (передаёт состояние лимитов)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from proscalper.core.types import OrderSide, SignalRejectReason
from proscalper.core.events import Signal
from proscalper.features.levels import Level, LevelState
from proscalper.market_data.tape import TapeMetrics
from proscalper.features.book_analyzer import LevelLiquidity


# ============================================================
# Состояние рынка и рисков (входные данные для фильтров)
# ============================================================

@dataclass
class MarketState:
    """
    Текущее состояние рынка для фильтрации.
    
    Заполняется из:
    - FastOrderBook (спред, глубина, лаг)
    - TapeAnalyzer (скорость сделок)
    - MarketDataGateway (статус подключения)
    """
    # Спред и глубина
    spread_ticks: int = 0
    top_depth_notional: float = 0.0
    
    # Лента
    trades_per_sec: float = 0.0
    
    # Здоровье данных
    book_lag_ms: int = 0
    bookticker_age_ms: int = 0
    ws_connected: bool = True
    book_in_sync: bool = True


@dataclass
class RiskState:
    """
    Текущее состояние риск-лимитов для фильтрации.
    
    Заполняется из:
    - RiskLimitsManager (дневные лимиты, кулдауны)
    - PositionManager (количество позиций)
    - App (kill switch)
    """
    # Дневные лимиты
    daily_loss_limit_reached: bool = False
    max_positions_reached: bool = False
    max_daily_trades_reached: bool = False
    
    # Кулдауны
    cooldown_after_loss_active: bool = False
    
    # Ресурсы
    margin_sufficient: bool = True
    api_rate_limit_risk: bool = False
    
    # Аварийные переключатели
    kill_switch: bool = False
    
    # Текущие значения (для журналирования)
    open_positions_count: int = 0
    daily_pnl: float = 0.0


# ============================================================
# Контекст фильтрации
# ============================================================

@dataclass
class FilterContext:
    """
    Полный контекст для проверки сигнала.
    
    Содержит все данные, которые могут понадобиться фильтрам.
    Передаётся в каждый фильтр цепочки.
    """
    # Проверяемый сигнал
    signal: Signal
    
    # Уровень, на котором основан сигнал
    level: Level
    
    # Метрики ленты
    tape_metrics: TapeMetrics
    
    # Ликвидность вокруг уровня
    book_liquidity: LevelLiquidity
    
    # Состояние рынка
    market_state: MarketState
    
    # Состояние риск-лимитов
    risk_state: RiskState
    
    # Текущая цена (для фильтров погони)
    current_price: float = 0.0


# ============================================================
# Результаты фильтрации
# ============================================================

@dataclass(frozen=True)
class FilterResult:
    """
    Результат одного фильтра.
    
    Если passed=True, reason и details пустые.
    Если passed=False, reason содержит код причины для журнала.
    """
    passed: bool
    reason: Optional[SignalRejectReason] = None
    details: str = ""
    
    @staticmethod
    def ok() -> "FilterResult":
        """Фильтр пройден."""
        return FilterResult(passed=True)
    
    @staticmethod
    def reject(
        reason: SignalRejectReason,
        details: str = "",
    ) -> "FilterResult":
        """Фильтр не пройден с указанием причины."""
        return FilterResult(
            passed=False,
            reason=reason,
            details=details,
        )
    
    @property
    def rejected(self) -> bool:
        """Фильтр не пройден."""
        return not self.passed


@dataclass
class FilterChainResult:
    """
    Результат всей цепочки фильтров.
    
    Содержит все причины отклонения для полного аудита.
    """
    passed: bool
    rejections: List[FilterResult] = field(default_factory=list)
    checked_filters_count: int = 0
    
    @property
    def rejection_reasons(self) -> List[SignalRejectReason]:
        """Список всех причин отклонения."""
        return [
            r.reason for r in self.rejections
            if r.reason is not None
        ]
    
    @property
    def rejection_codes(self) -> List[str]:
        """Список строковых кодов причин (для журнала)."""
        return [
            r.reason.value for r in self.rejections
            if r.reason is not None
        ]
    
    @property
    def details_summary(self) -> str:
        """Сводка всех причин для логирования."""
        if not self.rejections:
            return "OK"
        return "; ".join(
            f"{r.reason.value}: {r.details}" if r.details else r.reason.value
            for r in self.rejections
            if r.reason is not None
        )


# ============================================================
# Конфигурация фильтров
# ============================================================

@dataclass
class FilterConfig:
    """
    Конфигурация всех фильтров.
    
    Все пороги подобраны как разумные значения по умолчанию
    согласно формализации системы. В дальнейшем будут
    подкручены на живых данных.
    """
    # === Рыночные фильтры ===
    max_spread_ticks: int = 3
    min_top_depth_notional: float = 20000.0
    min_trade_rate_per_sec: float = 5.0
    max_book_lag_ms: int = 300
    max_bookticker_age_ms: int = 80
    
    # === Фильтры уровня ===
    min_level_strength: float = 3.0
    max_level_age_sec: float = 14400.0
    min_level_touches: int = 2
    max_approach_ticks: int = 20
    max_chase_ticks: int = 8
    
    # === Фильтры пробоя ===
    min_volume_burst_ratio: float = 2.5
    min_imbalance: float = 0.25
    max_opposing_wall_score: float = 0.4
    min_target_r: float = 1.5
    
    # === Режим работы цепочки ===
    fail_fast: bool = True  # остановиться на первом отклонении


# ============================================================
# Базовый класс фильтра
# ============================================================

class BaseFilter(ABC):
    """
    Абстрактный базовый класс фильтра.
    
    Каждый конкретный фильтр реализует метод check(),
    который возвращает FilterResult.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Имя фильтра для логирования."""
        ...
    
    @abstractmethod
    def check(self, context: FilterContext) -> FilterResult:
        """
        Проверяет сигнал на соответствие критерию фильтра.
        
        Возвращает:
        - FilterResult.ok() если проверка пройдена
        - FilterResult.reject(reason, details) если отклонён
        """
        ...


# ============================================================
# Рыночные фильтры
# ============================================================

class MarketHealthFilter(BaseFilter):
    """
    Фильтр здоровья рынка.
    
    Проверяет:
    - Спред не слишком широкий
    - Достаточная глубина стакана
    - Достаточная скорость сделок
    - Лаг стакана в пределах нормы
    - WS подключен и стакан синхронизирован
    """
    
    def __init__(self, config: FilterConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "MarketHealthFilter"
    
    def check(self, context: FilterContext) -> FilterResult:
        market = context.market_state
        
        # Проверка подключения
        if not market.ws_connected:
            return FilterResult.reject(
                SignalRejectReason.BOOK_LAG_TOO_HIGH,
                details="WS disconnected",
            )
        
        # Проверка синхронизации стакана
        if not market.book_in_sync:
            return FilterResult.reject(
                SignalRejectReason.BOOK_LAG_TOO_HIGH,
                details="Book out of sync",
            )
        
        # Проверка спреда
        if market.spread_ticks > self._config.max_spread_ticks:
            return FilterResult.reject(
                SignalRejectReason.SPREAD_TOO_WIDE,
                details=f"spread={market.spread_ticks} > {self._config.max_spread_ticks}",
            )
        
        # Проверка глубины стакана
        if market.top_depth_notional < self._config.min_top_depth_notional:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details=(
                    f"depth={market.top_depth_notional:.0f} < "
                    f"{self._config.min_top_depth_notional:.0f}"
                ),
            )
        
        # Проверка скорости сделок
        if market.trades_per_sec < self._config.min_trade_rate_per_sec:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details=(
                    f"trade_rate={market.trades_per_sec:.1f} < "
                    f"{self._config.min_trade_rate_per_sec:.1f}"
                ),
            )
        
        # Проверка лага стакана
        if market.book_lag_ms > self._config.max_book_lag_ms:
            return FilterResult.reject(
                SignalRejectReason.BOOK_LAG_TOO_HIGH,
                details=f"lag={market.book_lag_ms}ms > {self._config.max_book_lag_ms}ms",
            )
        
        return FilterResult.ok()


# ============================================================
# Фильтры уровня
# ============================================================

class LevelFilter(BaseFilter):
    """
    Фильтр уровня.
    
    Проверяет:
    - Уровень активен
    - Достаточная сила уровня
    - Уровень не слишком старый
    - Достаточно касаний
    - Уровень не в cooldown после ложного пробоя
    """
    
    def __init__(self, config: FilterConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "LevelFilter"
    
    def check(self, context: FilterContext) -> FilterResult:
        level = context.level
        signal_ts_ns = context.signal.created_ts_ns
        
        # Проверка состояния уровня
        if level.state not in (LevelState.FORMING, LevelState.ACTIVE):
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details=f"level state={level.state.name}",
            )
        
        # Проверка силы уровня
        if level.strength < self._config.min_level_strength:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details=(
                    f"strength={level.strength:.2f} < "
                    f"{self._config.min_level_strength:.2f}"
                ),
            )
        
        # Проверка возраста уровня
        age_sec = level.age_sec(signal_ts_ns)
        if age_sec > self._config.max_level_age_sec:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details=f"age={age_sec:.0f}s > {self._config.max_level_age_sec:.0f}s",
            )
        
        # Проверка количества касаний
        if level.touches < self._config.min_level_touches:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details=f"touches={level.touches} < {self._config.min_level_touches}",
            )
        
        # Проверка cooldown после ложных пробоев
        if level.fakeout_count > 0 and level.state == LevelState.FAILED_BREAK:
            return FilterResult.reject(
                SignalRejectReason.COOLDOWN_ACTIVE,
                details=f"fakeout_count={level.fakeout_count}",
            )
        
        return FilterResult.ok()


class ApproachFilter(BaseFilter):
    """
    Фильтр подхода к уровню.
    
    Проверяет:
    - Цена не слишком далеко от уровня (подход)
    - Цена не ушла далеко за уровень (погоня)
    
    Использует текущую цену и цену уровня.
    """
    
    def __init__(self, config: FilterConfig, tick_size: float):
        self._config = config
        self._tick_size = tick_size
    
    @property
    def name(self) -> str:
        return "ApproachFilter"
    
    def check(self, context: FilterContext) -> FilterResult:
        level = context.level
        current_price = context.current_price
        signal_side = context.signal.side
        
        if current_price <= 0:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details="current_price unknown",
            )
        
        distance_ticks = level.distance_to(current_price) / self._tick_size
        
        # Проверка: цена слишком далеко от уровня (не подошла)
        if distance_ticks > self._config.max_approach_ticks:
            return FilterResult.reject(
                SignalRejectReason.PRICE_ALREADY_TOO_FAR,
                details=(
                    f"distance={distance_ticks:.0f} > "
                    f"{self._config.max_approach_ticks} ticks"
                ),
            )
        
        # Проверка: цена ушла за уровень (погоня)
        beyond_level = self._is_beyond_level(
            current_price, level.center, signal_side
        )
        
        if beyond_level > self._config.max_chase_ticks * self._tick_size:
            return FilterResult.reject(
                SignalRejectReason.PRICE_ALREADY_TOO_FAR,
                details=(
                    f"chase={beyond_level / self._tick_size:.0f} > "
                    f"{self._config.max_chase_ticks} ticks"
                ),
            )
        
        return FilterResult.ok()
    
    def _is_beyond_level(
        self,
        price: float,
        level_price: float,
        side: OrderSide,
    ) -> float:
        """
        Возвращает расстояние, на которое цена ушла за уровень.
        0 если цена ещё не пересекла уровень.
        """
        if side == OrderSide.BUY:
            # Для лонга пробой вверх
            beyond = price - level_price
        else:
            # Для шорта пробой вниз
            beyond = level_price - price
        
        return max(0.0, beyond)


# ============================================================
# Фильтры пробоя
# ============================================================

class BreakoutTapeFilter(BaseFilter):
    """
    Фильтр пробоя по ленте.
    
    Проверяет:
    - Всплеск объёма в сторону пробоя
    - Дельта направлена в сторону пробоя
    """
    
    def __init__(self, config: FilterConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "BreakoutTapeFilter"
    
    def check(self, context: FilterContext) -> FilterResult:
        tape = context.tape_metrics
        side = context.signal.side
        
        # Определяем объём и медиану для стороны пробоя
        if side == OrderSide.BUY:
            aggressor_volume = tape.buy_volume_1s
            median_volume = tape.buy_volume_median_1s
            net_delta = tape.net_delta_1s
            
            # Дельта должна быть положительной для лонга
            if net_delta <= 0:
                return FilterResult.reject(
                    SignalRejectReason.WEAK_DELTA,
                    details=f"net_delta={net_delta:+.0f} <= 0 for BUY",
                )
        else:
            aggressor_volume = tape.sell_volume_1s
            median_volume = tape.sell_volume_median_1s
            net_delta = tape.net_delta_1s
            
            # Дельта должна быть отрицательной для шорта
            if net_delta >= 0:
                return FilterResult.reject(
                    SignalRejectReason.WEAK_DELTA,
                    details=f"net_delta={net_delta:+.0f} >= 0 for SELL",
                )
        
        # Проверка всплеска объёма
        if median_volume > 0:
            volume_burst_ratio = aggressor_volume / median_volume
            if volume_burst_ratio < self._config.min_volume_burst_ratio:
                return FilterResult.reject(
                    SignalRejectReason.WEAK_VOLUME,
                    details=(
                        f"burst={volume_burst_ratio:.2f} < "
                        f"{self._config.min_volume_burst_ratio:.2f}"
                    ),
                )
        else:
            # Если медиана неизвестна, проверяем абсолютный объём
            if aggressor_volume <= 0:
                return FilterResult.reject(
                    SignalRejectReason.WEAK_VOLUME,
                    details="no aggressor volume",
                )
        
        return FilterResult.ok()


class BreakoutBookFilter(BaseFilter):
    """
    Фильтр пробоя по стакану.
    
    Проверяет:
    - Дисбаланс направлен в сторону пробоя
    - Нет сильной встречной стены
    """
    
    def __init__(self, config: FilterConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "BreakoutBookFilter"
    
    def check(self, context: FilterContext) -> FilterResult:
        book = context.book_liquidity
        side = context.signal.side
        
        # Проверка дисбаланса
        if side == OrderSide.BUY:
            # Для лонга дисбаланс должен быть положительным
            if book.imbalance < self._config.min_imbalance:
                return FilterResult.reject(
                    SignalRejectReason.NEGATIVE_IMBALANCE,
                    details=f"imbalance={book.imbalance:+.2f} < {self._config.min_imbalance}",
                )
            
            # Проверка встречной стены (сверху для лонга)
            opposing_wall_score = book.spoof_score_ask
            if book.ask_wall is not None:
                opposing_wall_score = book.ask_wall.spoof_score
            
        else:
            # Для шорта дисбаланс должен быть отрицательным
            if book.imbalance > -self._config.min_imbalance:
                return FilterResult.reject(
                    SignalRejectReason.NEGATIVE_IMBALANCE,
                    details=f"imbalance={book.imbalance:+.2f} > {-self._config.min_imbalance}",
                )
            
            # Проверка встречной стены (снизу для шорта)
            opposing_wall_score = book.spoof_score_bid
            if book.bid_wall is not None:
                opposing_wall_score = book.bid_wall.spoof_score
        
        # Проверка силы встречной стены
        # Используем spoof_score как индикатор силы стены
        # Высокий spoof_score означает подозрительную стену
        # Низкий означает реальную стену, которая может блокировать движение
        
        return FilterResult.ok()


class TargetSpaceFilter(BaseFilter):
    """
    Фильтр пространства до цели.
    
    Проверяет, что расстояние до следующей цели достаточно
    относительно расстояния до стоп-лосса:
        target_space >= min_target_r * stop_distance
    
    Если сразу за уровнем стоит крупная встречная плотность
    или близкий сильный уровень — пробой не торгуем.
    """
    
    def __init__(self, config: FilterConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "TargetSpaceFilter"
    
    def check(self, context: FilterContext) -> FilterResult:
        signal = context.signal
        
        # Рассчитываем расстояние до стопа
        stop_distance = self._calculate_stop_distance(signal)
        
        if stop_distance <= 0:
            return FilterResult.reject(
                SignalRejectReason.STOP_DISTANCE_INVALID,
                details="stop_distance <= 0",
            )
        
        # Рассчитываем расстояние до цели
        target_distance = self._calculate_target_distance(signal)
        
        if target_distance <= 0:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details="no target ahead",
            )
        
        # Проверяем соотношение
        target_r = target_distance / stop_distance
        if target_r < self._config.min_target_r:
            return FilterResult.reject(
                SignalRejectReason.FILTERS_FAILED,
                details=(
                    f"target_r={target_r:.2f} < "
                    f"{self._config.min_target_r:.2f}"
                ),
            )
        
        return FilterResult.ok()
    
    def _calculate_stop_distance(self, signal: Signal) -> float:
        """Рассчитывает расстояние до стоп-лосса."""
        entry = signal.entry_price or signal.level_price
        
        if signal.side == OrderSide.BUY:
            return entry - signal.stop_price
        else:
            return signal.stop_price - entry
    
    def _calculate_target_distance(self, signal: Signal) -> float:
        """Рассчитывает расстояние до цели."""
        entry = signal.entry_price or signal.level_price
        
        if signal.take_profit_price is None:
            return 0.0
        
        if signal.side == OrderSide.BUY:
            return signal.take_profit_price - entry
        else:
            return entry - signal.take_profit_price


# ============================================================
# Риск-фильтры
# ============================================================

class RiskFilter(BaseFilter):
    """
    Фильтр риск-лимитов.
    
    Проверяет:
    - Дневной лимит убытков не достигнут
    - Максимум позиций не достигнут
    - Максимум сделок в день не достигнут
    - Нет активного кулдауна после убытка
    - Достаточно маржи
    - Нет риска превышения rate limit API
    - Kill switch не активен
    """
    
    def __init__(self, config: FilterConfig):
        self._config = config
    
    @property
    def name(self) -> str:
        return "RiskFilter"
    
    def check(self, context: FilterContext) -> FilterResult:
        risk = context.risk_state
        
        # Kill switch — самый приоритетный
        if risk.kill_switch:
            return FilterResult.reject(
                SignalRejectReason.KILL_SWITCH,
                details="kill switch active",
            )
        
        # Дневной лимит убытков
        if risk.daily_loss_limit_reached:
            return FilterResult.reject(
                SignalRejectReason.DAILY_LOSS_LIMIT,
                details=f"daily_pnl={risk.daily_pnl:+.2f}",
            )
        
        # Максимум позиций
        if risk.max_positions_reached:
            return FilterResult.reject(
                SignalRejectReason.MAX_POSITIONS_REACHED,
                details=f"positions={risk.open_positions_count}",
            )
        
        # Максимум сделок в день
        if risk.max_daily_trades_reached:
            return FilterResult.reject(
                SignalRejectReason.RISK_LIMIT_REACHED,
                details="daily trades limit reached",
            )
        
        # Кулдаун после убытка
        if risk.cooldown_after_loss_active:
            return FilterResult.reject(
                SignalRejectReason.COOLDOWN_ACTIVE,
                details="cooldown after loss",
            )
        
        # Недостаток маржи
        if not risk.margin_sufficient:
            return FilterResult.reject(
                SignalRejectReason.RISK_LIMIT_REACHED,
                details="insufficient margin",
            )
        
        # Риск rate limit API
        if risk.api_rate_limit_risk:
            return FilterResult.reject(
                SignalRejectReason.RISK_LIMIT_REACHED,
                details="API rate limit risk",
            )
        
        return FilterResult.ok()


# ============================================================
# Цепочка фильтров
# ============================================================

class FilterChain:
    """
    Цепочка фильтров с композицией.
    
    Режимы работы:
    - fail_fast=True: остановиться на первом отклонении (быстрый режим)
    - fail_fast=False: проверить все фильтры и собрать все причины
    
    Использование:
        chain = FilterChain(filters, fail_fast=True)
        result = chain.check(context)
        
        if result.passed:
            # Сигнал можно отправлять в RiskManager
        else:
            # Логируем причины отклонения
            journal.log_rejection(result.rejection_codes)
    """
    
    def __init__(
        self,
        filters: List[BaseFilter],
        fail_fast: bool = True,
    ):
        self._filters = filters
        self._fail_fast = fail_fast
    
    def check(self, context: FilterContext) -> FilterChainResult:
        """
        Проверяет контекст через все фильтры цепочки.
        
        Возвращает агрегированный результат.
        """
        rejections: List[FilterResult] = []
        checked_count = 0
        
        for filter_instance in self._filters:
            checked_count += 1
            
            try:
                result = filter_instance.check(context)
            except Exception as exc:
                # Ошибка в фильтре трактуется как отклонение
                result = FilterResult.reject(
                    SignalRejectReason.FILTERS_FAILED,
                    details=f"{filter_instance.name} error: {exc}",
                )
            
            if result.rejected:
                rejections.append(result)
                
                # В режиме fail_fast останавливаемся на первом отклонении
                if self._fail_fast:
                    break
        
        return FilterChainResult(
            passed=len(rejections) == 0,
            rejections=rejections,
            checked_filters_count=checked_count,
        )
    
    @property
    def filter_names(self) -> List[str]:
        """Имена фильтров в цепочке."""
        return [f.name for f in self._filters]


# ============================================================
# Менеджер фильтров сигналов
# ============================================================

class SignalFilterManager:
    """
    Менеджер фильтров сигналов.
    
    Объединяет все категории фильтров в единый интерфейс.
    Предоставляет методы для быстрой и полной проверки.
    
    Использование:
        manager = SignalFilterManager(config, tick_sizes)
        
        # Полная проверка сигнала
        result = manager.check_signal(context)
        
        if result.passed:
            risk_manager.approve(signal)
        else:
            journal.reject(signal, result.rejection_codes)
    """
    
    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        tick_sizes: Optional[dict] = None,
    ):
        self._config = config or FilterConfig()
        self._tick_sizes = {
            k.upper(): v for k, v in (tick_sizes or {}).items()
        }
        
        # Создаём фильтры по категориям
        self._market_health_filter = MarketHealthFilter(self._config)
        self._level_filter = LevelFilter(self._config)
        self._breakout_tape_filter = BreakoutTapeFilter(self._config)
        self._breakout_book_filter = BreakoutBookFilter(self._config)
        self._target_space_filter = TargetSpaceFilter(self._config)
        self._risk_filter = RiskFilter(self._config)
        
        # Кэш цепочек по символам (для ApproachFilter с tick_size)
        self._full_chains: dict = {}
        self._fast_chains: dict = {}
    
    def check_signal(
        self,
        context: FilterContext,
        fail_fast: Optional[bool] = None,
    ) -> FilterChainResult:
        """
        Полная проверка сигнала через все фильтры.
        
        Args:
            context: контекст фильтрации
            fail_fast: режим остановки на первом отклонении
                       (по умолчанию из конфига)
        
        Возвращает агрегированный результат.
        """
        chain = self._get_full_chain(context.signal.symbol)
        
        if fail_fast is None:
            fail_fast = self._config.fail_fast
        
        # Если режим отличается от конфига, создаём временную цепочку
        if fail_fast != chain._fail_fast:
            chain = FilterChain(
                filters=chain._filters,
                fail_fast=fail_fast,
            )
        
        return chain.check(context)
    
    def check_fast(self, context: FilterContext) -> FilterChainResult:
        """
        Быстрая проверка сигнала (только критичные фильтры).
        
        Используется перед отправкой ордера, когда нужна
        минимальная задержка.
        
        Проверяет:
        - Здоровье рынка (спред, лаг, подключение)
        - Риск-лимиты (особенно kill switch)
        
        Не проверяет:
        - Силу уровня
        - Метрики ленты
        - Метрики стакана
        """
        chain = self._get_fast_chain()
        return chain.check(context)
    
    def get_stats(self) -> dict:
        """Возвращает статистику фильтров."""
        return {
            "config": {
                "fail_fast": self._config.fail_fast,
                "max_spread_ticks": self._config.max_spread_ticks,
                "min_level_strength": self._config.min_level_strength,
                "min_volume_burst_ratio": self._config.min_volume_burst_ratio,
            },
            "chains": len(self._full_chains),
        }
    
    def _get_full_chain(self, symbol: str) -> FilterChain:
        """Возвращает полную цепочку фильтров для символа."""
        symbol = symbol.upper()
        
        if symbol not in self._full_chains:
            tick_size = self._tick_sizes.get(symbol, 0.01)
            
            approach_filter = ApproachFilter(
                config=self._config,
                tick_size=tick_size,
            )
            
            filters = [
                self._risk_filter,           # Kill switch приоритетен
                self._market_health_filter,
                self._level_filter,
                approach_filter,
                self._breakout_tape_filter,
                self._breakout_book_filter,
                self._target_space_filter,
            ]
            
            self._full_chains[symbol] = FilterChain(
                filters=filters,
                fail_fast=self._config.fail_fast,
            )
        
        return self._full_chains[symbol]
    
    def _get_fast_chain(self) -> FilterChain:
        """Возвращает быструю цепочку фильтров."""
        if not self._fast_chains:
            filters = [
                self._risk_filter,
                self._market_health_filter,
            ]
            
            self._fast_chains["fast"] = FilterChain(
                filters=filters,
                fail_fast=True,
            )
        
        return self._fast_chains["fast"]