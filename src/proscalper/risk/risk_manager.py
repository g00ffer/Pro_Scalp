"""
Общий менеджер рисков.

Объединяет все компоненты риск-менеджмента:
- RiskLimitsManager (лимиты: просадка, дневной убыток, позиции)
- PositionSizerManager (расчёт размеров позиций)
- PositionManager (управление позициями)

Логика работы:
1. Получаем сигнал от SignalGenerator
2. Проверяем лимиты (дёшево, быстро)
3. Рассчитываем размер позиции
4. Проверяем лимиты позиции
5. Если всё ок → в очередь на исполнение
6. Если не ок → отклоняем с причиной

Интеграция с ExecutionEngine через асинхронную очередь.
Это даёт разделение ответственности и асинхронность.

Используется в связке с:
- SignalGenerator (входные сигналы)
- ExecutionEngine (исполнение одобренных сигналов)
- DecisionJournal (логирование решений)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from proscalper.core.types import OrderSide
from proscalper.signals.signal_generator import TradingSignal
from proscalper.risk.sizing import (
    PositionSize,
    PositionSizerManager,
    SizingConfig,
)
from proscalper.risk.limits import (
    RiskLimitsManager,
    LimitCheckResult,
    LimitConfig,
)
from proscalper.risk.position_manager import (
    PositionManager,
    ManagedPosition,
    PositionConfig,
)


class RiskDecision(Enum):
    """
    Решение риск-менеджера по сигналу.
    """
    APPROVED = auto()       # Сигнал одобрен, в очередь на исполнение
    REJECTED = auto()       # Сигнал отклонён
    DELAYED = auto()        # Сигнал отложен (например, ждём освобождения слота)


@dataclass
class RiskCheckResult:
    """
    Результат проверки сигнала риск-менеджером.
    
    Содержит всю информацию для:
    - ExecutionEngine (если одобрен)
    - DecisionJournal (логирование)
    """
    # Решение
    decision: RiskDecision
    reason: str = ""
    
    # Исходный сигнал
    signal: Optional[TradingSignal] = None
    
    # Рассчитанный размер позиции
    position_size: Optional[PositionSize] = None
    
    # Нарушенные лимиты
    violated_limits: List[str] = field(default_factory=list)
    
    # Время проверки
    checked_ts_ns: int = 0
    
    # Дополнительные данные для логирования
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_approved(self) -> bool:
        """Сигнал одобрен."""
        return self.decision == RiskDecision.APPROVED
    
    @property
    def is_rejected(self) -> bool:
        """Сигнал отклонён."""
        return self.decision == RiskDecision.REJECTED


@dataclass
class RiskManagerConfig:
    """
    Конфигурация риск-менеджера.
    
    Все параметры подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Очередь исполнения
    queue_size: int = 10                    # размер очереди сигналов
    
    # Проверки
    require_valid_stop: bool = True         # требовать валидный стоп-лосс
    min_risk_reward_ratio: float = 1.5      # минимальное соотношение риск/прибыль
    require_breakeven_feasible: bool = True # требовать достижимости безубытка
    
    # Действия при нарушениях
    stop_trading_on_drawdown: bool = True   # останавливать торговлю при просадке
    skip_signal_on_position_limit: bool = True  # пропускать сигнал при лимите позиций
    
    # Таймауты
    signal_ttl_ms: int = 5000               # время жизни сигнала в очереди


@dataclass
class QueuedSignal:
    """
    Сигнал в очереди на исполнение.
    
    Оборачивает TradingSignal с дополнительной информацией
    для ExecutionEngine.
    """
    queue_id: str
    signal: TradingSignal
    position_size: PositionSize
    queued_ts_ns: int
    target_levels: List[float] = field(default_factory=list)
    
    @property
    def age_ms(self) -> float:
        """Возраст сигнала в очереди (мс)."""
        return (time.time_ns() - self.queued_ts_ns) / 1_000_000


class RiskManager:
    """
    Общий менеджер рисков.
    
    Основная логика:
    1. Получаем сигнал от SignalGenerator
    2. Проверяем общие лимиты (просадка, дневной убыток)
    3. Рассчитываем размер позиции
    4. Проверяем лимиты позиции
    5. Проверяем соотношение риск/прибыль
    6. Если всё ок → в очередь на исполнение
    7. Если не ок → отклоняем с причиной
    
    Использование:
        risk_manager = RiskManager(
            limits_manager=limits,
            sizer_manager=sizer,
            position_manager=position_mgr,
            config=RiskManagerConfig(),
        )
        
        # При получении сигнала
        result = await risk_manager.check_signal(
            signal=trading_signal,
            next_levels=[76200, 76500],
        )
        
        if result.is_approved:
            print(f"Сигнал одобрен, размер: {result.position_size.quantity}")
        else:
            print(f"Сигнал отклонён: {result.reason}")
        
        # Исполнитель читает из очереди
        async for queued in risk_manager.approved_signals():
            execution_engine.execute(queued)
    """
    
    def __init__(
        self,
        limits_manager: RiskLimitsManager,
        sizer_manager: PositionSizerManager,
        position_manager: PositionManager,
        config: Optional[RiskManagerConfig] = None,
    ):
        self.limits_manager = limits_manager
        self.sizer_manager = sizer_manager
        self.position_manager = position_manager
        self.config = config or RiskManagerConfig()
        
        # Очередь одобренных сигналов
        self._approved_queue: asyncio.Queue[QueuedSignal] = asyncio.Queue(
            maxsize=self.config.queue_size
        )
        
        # Статистика
        self._total_checked: int = 0
        self._total_approved: int = 0
        self._total_rejected: int = 0
        self._rejection_reasons: Dict[str, int] = {}
    
    async def check_signal(
        self,
        signal: TradingSignal,
        next_levels: List[float],
    ) -> RiskCheckResult:
        """
        Проверяет торговый сигнал.
        
        Это основной метод, вызываемый при получении сигнала.
        
        Порядок проверок:
        1. Общие лимиты (просадка, дневной убыток)
        2. Расчёт размера позиции
        3. Лимиты позиции
        4. Соотношение риск/прибыль
        5. Помещение в очередь
        
        Args:
            signal: Торговый сигнал от SignalGenerator
            next_levels: Следующие уровни (для тейк-профитов)
            
        Returns:
            Результат проверки
        """
        now_ns = time.time_ns()
        self._total_checked += 1
        
        # ============================================
        # Шаг 1: Проверяем общие лимиты
        # ============================================
        general_check = self.limits_manager.check_trading_allowed()
        
        if general_check.is_blocked:
            return self._reject_signal(
                signal=signal,
                reason=f"Общие лимиты нарушены: {', '.join(general_check.violated_limits)}",
                violated_limits=general_check.violated_limits,
                now_ns=now_ns,
            )
        
        # ============================================
        # Шаг 2: Рассчитываем размер позиции
        # ============================================
        # Определяем стоп-лосс
        stop_loss_price = self._calculate_stop_loss(signal)
        
        # Получаем сайзер для символа
        sizer = self.sizer_manager.get_sizer(signal.symbol)
        if sizer is None:
            return self._reject_signal(
                signal=signal,
                reason=f"Нет сайзера для символа {signal.symbol}",
                now_ns=now_ns,
            )
        
        # Рассчитываем размер
        position_size = sizer.calculate_size(
            deposit=self.limits_manager.current_deposit,
            entry_price=signal.entry_price,
            stop_loss_price=stop_loss_price,
            direction=signal.direction,
        )
        
        if not position_size.is_valid:
            return self._reject_signal(
                signal=signal,
                reason=f"Невалидный размер позиции: {', '.join(position_size.validation_errors)}",
                violated_limits=position_size.validation_errors,
                now_ns=now_ns,
            )
        
        # ============================================
        # Шаг 3: Проверяем лимиты позиции
        # ============================================
        position_check = self.limits_manager.check_position(
            position_notional=position_size.notional,
        )
        
        if position_check.is_blocked:
            return self._reject_signal(
                signal=signal,
                reason=f"Лимиты позиции нарушены: {', '.join(position_check.violated_limits)}",
                violated_limits=position_check.violated_limits,
                now_ns=now_ns,
            )
        
        # ============================================
        # Шаг 4: Проверяем соотношение риск/прибыль
        # ============================================
        if self.config.min_risk_reward_ratio > 0:
            risk_reward = self._calculate_risk_reward(
                entry_price=signal.entry_price,
                stop_loss_price=stop_loss_price,
                next_levels=next_levels,
                direction=signal.direction,
            )
            
            if risk_reward < self.config.min_risk_reward_ratio:
                return self._reject_signal(
                    signal=signal,
                    reason=(
                        f"Соотношение риск/прибыль {risk_reward:.2f} "
                        f"меньше минимального {self.config.min_risk_reward_ratio}"
                    ),
                    metadata={"risk_reward": risk_reward},
                    now_ns=now_ns,
                )
        
        # ============================================
        # Шаг 5: Проверяем достаточность уровней для тейков
        # ============================================
        required_levels = self.position_manager.config.num_take_profit_levels
        if len(next_levels) < required_levels:
            return self._reject_signal(
                signal=signal,
                reason=(
                    f"Недостаточно уровней для тейков: "
                    f"нужно {required_levels}, есть {len(next_levels)}"
                ),
                now_ns=now_ns,
            )
        
        # ============================================
        # Шаг 6: Одобряем сигнал и помещаем в очередь
        # ============================================
        return await self._approve_signal(
            signal=signal,
            position_size=position_size,
            next_levels=next_levels,
            now_ns=now_ns,
        )
    
    async def approved_signals(self):
        """
        Асинхронный генератор одобренных сигналов.
        
        Используется в ExecutionEngine для получения сигналов на исполнение.
        
        Использование:
            async for queued in risk_manager.approved_signals():
                execution_engine.execute(queued.signal, queued.position_size)
        """
        while True:
            try:
                queued = await self._approved_queue.get()
                
                # Проверяем, не устарел ли сигнал
                if queued.age_ms > self.config.signal_ttl_ms:
                    continue
                
                yield queued
                
                self._approved_queue.task_done()
            
            except asyncio.CancelledError:
                break
    
    def on_trade_closed(self, pnl: float) -> None:
        """
        Обработка закрытой сделки.
        
        Вызывается из ExecutionEngine после закрытия позиции.
        Обновляет лимиты и статистику.
        """
        self.limits_manager.record_trade_pnl(pnl)
        self.limits_manager.close_position()
    
    def on_position_opened(self) -> None:
        """
        Обработка открытой позиции.
        
        Вызывается из ExecutionEngine после открытия позиции.
        """
        self.limits_manager.open_position()
    
    def update_equity(self, equity: float) -> None:
        """
        Обновляет текущий капитал.
        
        Вызывается периодически или после каждой сделки.
        """
        self.limits_manager.update_equity(equity)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику риск-менеджера."""
        return {
            "total_checked": self._total_checked,
            "total_approved": self._total_approved,
            "total_rejected": self._total_rejected,
            "approval_rate": (
                self._total_approved / self._total_checked
                if self._total_checked > 0 else 0.0
            ),
            "rejection_reasons": self._rejection_reasons.copy(),
            "queue_size": self._approved_queue.qsize(),
            "limits": self.limits_manager.get_stats(),
        }
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику."""
        self._total_checked = 0
        self._total_approved = 0
        self._total_rejected = 0
        self._rejection_reasons.clear()
    
    async def _approve_signal(
        self,
        signal: TradingSignal,
        position_size: PositionSize,
        next_levels: List[float],
        now_ns: int,
    ) -> RiskCheckResult:
        """Одобряет сигнал и помещает в очередь."""
        queued = QueuedSignal(
            queue_id=str(uuid.uuid4()),
            signal=signal,
            position_size=position_size,
            queued_ts_ns=now_ns,
            target_levels=next_levels,
        )
        
        try:
            self._approved_queue.put_nowait(queued)
            self._total_approved += 1
            
            return RiskCheckResult(
                decision=RiskDecision.APPROVED,
                reason="Сигнал одобрен",
                signal=signal,
                position_size=position_size,
                checked_ts_ns=now_ns,
                metadata={
                    "queue_id": queued.queue_id,
                    "next_levels": next_levels,
                },
            )
        
        except asyncio.QueueFull:
            return self._reject_signal(
                signal=signal,
                reason="Очередь исполнения переполнена",
                now_ns=now_ns,
            )
    
    def _reject_signal(
        self,
        signal: TradingSignal,
        reason: str,
        violated_limits: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        now_ns: int = 0,
    ) -> RiskCheckResult:
        """Отклоняет сигнал с причиной."""
        self._total_rejected += 1
        
        # Записываем причину для статистики
        reason_key = reason.split(":")[0].strip()
        self._rejection_reasons[reason_key] = (
            self._rejection_reasons.get(reason_key, 0) + 1
        )
        
        return RiskCheckResult(
            decision=RiskDecision.REJECTED,
            reason=reason,
            signal=signal,
            violated_limits=violated_limits or [],
            checked_ts_ns=now_ns or time.time_ns(),
            metadata=metadata or {},
        )
    
    def _calculate_stop_loss(self, signal: TradingSignal) -> float:
        """
        Рассчитывает цену стоп-лосса для сигнала.
        
        Стоп ставится за уровнем с буфером.
        """
        # Получаем тик-сайз
        sizer = self.sizer_manager.get_sizer(signal.symbol)
        if sizer is None:
            tick_size = 0.01
        else:
            tick_size = sizer.tick_size
        
        buffer = self.position_manager.config.stop_loss_buffer_ticks * tick_size
        
        if signal.direction == OrderSide.BUY:
            # Для лонга стоп ниже уровня
            return signal.level_center - buffer
        else:
            # Для шорта стоп выше уровня
            return signal.level_center + buffer
    
    def _calculate_risk_reward(
        self,
        entry_price: float,
        stop_loss_price: float,
        next_levels: List[float],
        direction: OrderSide,
    ) -> float:
        """
        Рассчитывает соотношение риск/прибыль.
        
        Риск = расстояние до стоп-лосса
        Прибыль = расстояние до первой цели
        
        Возвращает отношение прибыль/риск.
        """
        if not next_levels:
            return 0.0
        
        first_target = next_levels[0]
        
        if direction == OrderSide.BUY:
            risk = entry_price - stop_loss_price
            reward = first_target - entry_price
        else:
            risk = stop_loss_price - entry_price
            reward = entry_price - first_target
        
        if risk <= 0:
            return 0.0
        
        return reward / risk