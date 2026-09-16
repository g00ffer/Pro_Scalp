"""
Менеджер ордеров: координатор слоя исполнения.

Связывает все модули исполнения в единую систему:
- OrderBuilder: построение ордеров
- BracketExecutor: безопасная отправка (batch/native/fill-driven)
- ProtectionWatchdog: мониторинг защиты позиций
- OrderTracker: отслеживание активных ордеров

Предоставляет единый интерфейс для верхних слоёв:
- RiskManager: проверка и одобрение сделок
- SignalGenerator: исполнение сигналов
- PositionManager: управление позициями

Использование:
    manager = OrderManager(
        order_builder=builder,
        bracket_executor=bracket,
        protection_watchdog=watchdog,
        order_tracker=tracker,
    )
    
    # Открытие позиции
    result = await manager.open_position(signal, sizing)
    
    # Закрытие позиции
    await manager.close_position(position_id, reason="take_profit")
    
    # Отмена всех ордеров
    await manager.cancel_all_orders(symbol)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol

from proscalper.core.events import Signal
from proscalper.core.types import (
    ExecutionStatus,
    OrderSide,
    OrderStatus,
    PositionSide,
)
from proscalper.execution.order_builder import OrderBuilder, OrderRequest
from proscalper.execution.bracket import (
    BaseBracketExecutor,
    BracketResult,
    PositionProtectionState,
)
from proscalper.execution.protection_watchdog import ProtectionWatchdog


# ============================================================
# Состояния позиции
# ============================================================

class PositionState(str, Enum):
    """Состояние позиции."""
    PENDING_OPEN = "PENDING_OPEN"           # Открытие в процессе
    OPEN = "OPEN"                           # Позиция открыта
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"   # Частично закрыта
    CLOSING = "CLOSING"                     # Закрытие в процессе
    CLOSED = "CLOSED"                       # Позиция закрыта
    FAILED = "FAILED"                       # Ошибка открытия/закрытия


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class OrderManagerConfig:
    """
    Конфигурация менеджера ордеров.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    # Таймауты
    position_open_timeout_ms: int = 5000    # максимум времени открытия
    position_close_timeout_ms: int = 3000   # максимум времени закрытия
    
    # Ограничения
    max_open_positions: int = 1             # максимум открытых позиций (для скальпинга)
    max_concurrent_operations: int = 5      # максимум одновременных операций
    
    # Поведение
    auto_register_in_tracker: bool = True   # автоматическая регистрация в трекере
    notify_on_state_change: bool = True     # уведомления об изменениях


# ============================================================
# Интерфейсы для внешних зависимостей
# ============================================================

class OrderTrackerClient(Protocol):
    """Протокол для работы с трекером ордеров."""
    
    def track_order(self, order: OrderRequest) -> None:
        """Регистрирует ордер для отслеживания."""
        ...
    
    def untrack_order(self, client_order_id: str) -> None:
        """Удаляет ордер из отслеживания."""
        ...
    
    def get_order_status(self, client_order_id: str) -> Optional[OrderStatus]:
        """Возвращает статус ордера."""
        ...


class IncidentReporter(Protocol):
    """Протокол для логирования инцидентов."""
    
    def report_incident(
        self,
        incident_type: str,
        symbol: str,
        details: Dict[str, Any],
        severity: str = "WARNING",
    ) -> None:
        """Логирует инцидент."""
        ...


# ============================================================
# Результат операции
# ============================================================

@dataclass
class PositionOperationResult:
    """
    Результат операции с позицией.
    
    Содержит статус, идентификаторы и метаданные.
    """
    # Идентификация
    position_id: str
    signal_id: str
    symbol: str
    
    # Статус
    success: bool
    status: ExecutionStatus = ExecutionStatus.PENDING
    position_state: PositionState = PositionState.PENDING_OPEN
    
    # Детали исполнения
    entry_price: float = 0.0
    filled_quantity: float = 0.0
    stop_price: float = 0.0
    
    # Защита
    is_protected: bool = False
    protection_state: PositionProtectionState = PositionProtectionState.PENDING_ENTRY
    protected_after_ms: float = 0.0
    
    # Время
    started_ts_ns: int = 0
    completed_ts_ns: int = 0
    
    # Ошибки и инциденты
    error_message: str = ""
    incidents: List[str] = field(default_factory=list)
    
    @property
    def latency_ms(self) -> float:
        """Задержка операции в миллисекундах."""
        if self.completed_ts_ns <= 0 or self.started_ts_ns <= 0:
            return 0.0
        return (self.completed_ts_ns - self.started_ts_ns) / 1_000_000


# ============================================================
# Позиция
# ============================================================

@dataclass
class ManagedPosition:
    """
    Управляемая позиция.
    
    Хранит полное состояние позиции и связанные ордера.
    """
    # Идентификация
    position_id: str
    signal_id: str
    symbol: str
    side: OrderSide
    
    # Параметры
    entry_price: float
    quantity: float
    stop_price: float
    
    # Состояние
    state: PositionState = PositionState.OPEN
    filled_quantity: float = 0.0
    current_quantity: float = 0.0
    
    # Защита
    is_protected: bool = False
    stop_order_id: Optional[str] = None
    protection_state: PositionProtectionState = PositionProtectionState.PENDING_ENTRY
    
    # Время
    opened_ts_ns: int = 0
    closed_ts_ns: int = 0
    
    # Связанные ордера
    entry_order_id: Optional[str] = None
    close_order_ids: List[str] = field(default_factory=list)
    
    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        """Позиция в лонг."""
        return self.side == OrderSide.BUY
    
    @property
    def is_short(self) -> bool:
        """Позиция в шорт."""
        return self.side == OrderSide.SELL
    
    @property
    def is_open(self) -> bool:
        """Позиция открыта."""
        return self.state == PositionState.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Позиция закрыта."""
        return self.state == PositionState.CLOSED


# ============================================================
# Менеджер ордеров
# ============================================================

class OrderManager:
    """
    Координатор слоя исполнения.
    
    Управляет жизненным циклом позиций:
    1. Открытие (через BracketExecutor)
    2. Мониторинг защиты (через ProtectionWatchdog)
    3. Закрытие (по сигналу или аварийно)
    4. Отмена ордеров
    
    Использование:
        manager = OrderManager(
            order_builder=builder,
            bracket_executor=bracket,
            protection_watchdog=watchdog,
            order_tracker=tracker,
            config=OrderManagerConfig(),
        )
        
        # Открытие позиции
        result = await manager.open_position(
            signal=signal,
            quantity=0.001,
            stop_price=75990.0,
        )
        
        if result.success:
            print(f"Позиция открыта: {result.position_id}")
        
        # Закрытие позиции
        await manager.close_position(
            position_id=result.position_id,
            reason="take_profit",
        )
    """
    
    def __init__(
        self,
        order_builder: OrderBuilder,
        bracket_executor: BaseBracketExecutor,
        protection_watchdog: ProtectionWatchdog,
        order_tracker: Optional[OrderTrackerClient] = None,
        incident_reporter: Optional[IncidentReporter] = None,
        config: Optional[OrderManagerConfig] = None,
    ):
        self._builder = order_builder
        self._bracket = bracket_executor
        self._watchdog = protection_watchdog
        self._tracker = order_tracker
        self._incident_reporter = incident_reporter
        self._config = config or OrderManagerConfig()
        
        # Активные позиции: position_id -> ManagedPosition
        self._positions: Dict[str, ManagedPosition] = {}
        
        # Счётчик для генерации ID
        self._position_counter: int = 0
        
        # Обработчики событий
        self._state_change_handlers: List[Callable[[ManagedPosition, PositionState], None]] = []
        
        # Статистика
        self._total_open_attempts: int = 0
        self._total_open_success: int = 0
        self._total_close_attempts: int = 0
        self._total_close_success: int = 0
    
    # ============================================
    # Открытие позиций
    # ============================================
    
    async def open_position(
        self,
        signal: Signal,
        quantity: float,
        stop_price: float,
    ) -> PositionOperationResult:
        """
        Открывает новую позицию.
        
        Логика:
        1. Проверяет лимит открытых позиций
        2. Вызывает BracketExecutor для безопасного открытия
        3. Активирует ProtectionWatchdog для мониторинга
        4. Регистрирует позицию в менеджере
        5. Возвращает результат
        
        Args:
            signal: торговый сигнал
            quantity: размер позиции
            stop_price: цена защитного стопа
            
        Returns:
            Результат операции открытия
        """
        started_ts = time.time_ns()
        self._total_open_attempts += 1
        
        # Проверяем лимит открытых позиций
        open_count = sum(1 for p in self._positions.values() if p.is_open)
        if open_count >= self._config.max_open_positions:
            result = PositionOperationResult(
                position_id="",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                success=False,
                status=ExecutionStatus.REJECTED,
                error_message="Max open positions reached",
                started_ts_ns=started_ts,
                completed_ts_ns=time.time_ns(),
            )
            result.incidents.append("MAX_POSITIONS_REACHED")
            return result
        
        # Генерируем position_id
        self._position_counter += 1
        position_id = f"pos_{self._position_counter}"
        
        # Создаём результат
        result = PositionOperationResult(
            position_id=position_id,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            success=False,
            started_ts_ns=started_ts,
        )
        
        try:
            # Вызываем BracketExecutor
            bracket_result = await self._bracket.open_with_protection(
                signal=signal,
                quantity=quantity,
                stop_price=stop_price,
            )
            
            # Обрабатываем результат bracket
            result.status = bracket_result.status
            result.protection_state = bracket_result.protection_state
            result.incidents.extend(bracket_result.incidents)
            
            if bracket_result.entry_leg is not None:
                result.filled_quantity = bracket_result.entry_leg.filled_qty
                result.entry_price = bracket_result.entry_leg.avg_fill_price
            
            if bracket_result.stop_leg is not None:
                result.stop_price = stop_price
            
            result.is_protected = bracket_result.is_protected
            result.protected_after_ms = bracket_result.protected_after_ms
            
            # Если вход успешен
            if bracket_result.entry_leg is not None and bracket_result.entry_leg.accepted:
                result.success = True
                result.position_state = PositionState.OPEN
                self._total_open_success += 1
                
                # Создаём ManagedPosition
                position = ManagedPosition(
                    position_id=position_id,
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    side=signal.side,
                    entry_price=result.entry_price,
                    quantity=quantity,
                    stop_price=stop_price,
                    state=PositionState.OPEN,
                    filled_quantity=result.filled_quantity,
                    current_quantity=result.filled_quantity,
                    is_protected=result.is_protected,
                    stop_order_id=bracket_result.stop_order_id,
                    protection_state=result.protection_state,
                    opened_ts_ns=time.time_ns(),
                    entry_order_id=bracket_result.entry_leg.client_order_id,
                )
                
                self._positions[position_id] = position
                
                # Активируем ProtectionWatchdog
                await self._watchdog.arm(
                    signal=signal,
                    expected_quantity=quantity,
                    stop_price=stop_price,
                )
                
                # Уведомляем об открытии
                self._notify_state_change(position, PositionState.OPEN)
            else:
                # Вход не удался
                result.success = False
                result.position_state = PositionState.FAILED
                result.error_message = "Entry failed"
        
        except Exception as exc:
            result.success = False
            result.status = ExecutionStatus.ERROR
            result.error_message = str(exc)
            result.incidents.append(f"OPEN_POSITION_ERROR: {exc}")
            
            self._report_incident(
                "OPEN_POSITION_ERROR",
                signal.symbol,
                {
                    "signal_id": signal.signal_id,
                    "error": str(exc),
                },
                severity="ERROR",
            )
        
        result.completed_ts_ns = time.time_ns()
        return result
    
    # ============================================
    # Закрытие позиций
    # ============================================
    
    async def close_position(
        self,
        position_id: str,
        reason: str = "manual",
        close_price: Optional[float] = None,
    ) -> PositionOperationResult:
        """
        Закрывает позицию.
        
        Логика:
        1. Находит позицию по ID
        2. Вызывает BracketExecutor для закрытия
        3. Отменяет защитный стоп
        4. Обновляет состояние позиции
        5. Возвращает результат
        
        Args:
            position_id: идентификатор позиции
            reason: причина закрытия (для логирования)
            close_price: цена закрытия (для расчёта PnL)
            
        Returns:
            Результат операции закрытия
        """
        started_ts = time.time_ns()
        self._total_close_attempts += 1
        
        # Находим позицию
        position = self._positions.get(position_id)
        if position is None:
            return PositionOperationResult(
                position_id=position_id,
                signal_id="",
                symbol="",
                success=False,
                status=ExecutionStatus.REJECTED,
                error_message="Position not found",
                started_ts_ns=started_ts,
                completed_ts_ns=time.time_ns(),
            )
        
        result = PositionOperationResult(
            position_id=position_id,
            signal_id=position.signal_id,
            symbol=position.symbol,
            success=False,
            started_ts_ns=started_ts,
        )
        
        try:
            # Обновляем состояние
            position.state = PositionState.CLOSING
            self._notify_state_change(position, PositionState.CLOSING)
            
            # Отключаем ProtectionWatchdog
            self._watchdog.disarm(position.signal_id)
            
            # Вызываем BracketExecutor для закрытия
            close_leg = await self._bracket.close_position(
                signal_id=position.signal_id,
                reason=reason,
            )
            
            if close_leg is not None:
                result.success = True
                result.filled_quantity = close_leg.filled_qty
                result.entry_price = close_leg.avg_fill_price
                
                # Рассчитываем PnL
                if close_price is not None and position.entry_price > 0:
                    if position.is_long:
                        result.realized_pnl = (
                            (close_price - position.entry_price)
                            * position.current_quantity
                        )
                    else:
                        result.realized_pnl = (
                            (position.entry_price - close_price)
                            * position.current_quantity
                        )
                
                position.realized_pnl = result.realized_pnl
                position.close_order_ids.append(close_leg.client_order_id)
            
            # Обновляем состояние позиции
            position.state = PositionState.CLOSED
            position.closed_ts_ns = time.time_ns()
            position.current_quantity = 0.0
            
            result.success = True
            result.position_state = PositionState.CLOSED
            result.status = ExecutionStatus.COMPLETED
            self._total_close_success += 1
            
            # Уведомляем о закрытии
            self._notify_state_change(position, PositionState.CLOSED)
        
        except Exception as exc:
            result.success = False
            result.status = ExecutionStatus.ERROR
            result.error_message = str(exc)
            result.incidents.append(f"CLOSE_POSITION_ERROR: {exc}")
            
            self._report_incident(
                "CLOSE_POSITION_ERROR",
                position.symbol,
                {
                    "position_id": position_id,
                    "reason": reason,
                    "error": str(exc),
                },
                severity="ERROR",
            )
        
        result.completed_ts_ns = time.time_ns()
        return result
    
    async def cancel_all_orders(self, symbol: str) -> int:
        """
        Отменяет все активные ордера для символа.
        
        Используется при аварийной остановке или смене режима.
        
        Returns:
            Количество отменённых ордеров
        """
        cancelled_count = 0
        
        # Находим все позиции для символа
        for position in self._positions.values():
            if position.symbol != symbol:
                continue
            
            if position.state in (PositionState.OPEN, PositionState.CLOSING):
                # Отменяем стоп-ордер
                if position.stop_order_id is not None:
                    try:
                        await self._bracket._exchange.cancel_order(
                            symbol=symbol,
                            client_order_id=position.stop_order_id,
                        )
                        cancelled_count += 1
                    except Exception:
                        pass
            
            # Отключаем ProtectionWatchdog
            self._watchdog.disarm(position.signal_id)
        
        return cancelled_count
    
    # ============================================
    # Запросы состояния
    # ============================================
    
    def get_position(self, position_id: str) -> Optional[ManagedPosition]:
        """Возвращает позицию по ID."""
        return self._positions.get(position_id)
    
    def get_open_positions(self) -> List[ManagedPosition]:
        """Возвращает все открытые позиции."""
        return [
            pos for pos in self._positions.values()
            if pos.is_open
        ]
    
    def get_positions_by_symbol(self, symbol: str) -> List[ManagedPosition]:
        """Возвращает все позиции для символа."""
        return [
            pos for pos in self._positions.values()
            if pos.symbol == symbol.upper()
        ]
    
    def get_position_count(self) -> int:
        """Возвращает количество открытых позиций."""
        return sum(1 for pos in self._positions.values() if pos.is_open)
    
    def is_position_open(self, position_id: str) -> bool:
        """Проверяет, открыта ли позиция."""
        pos = self._positions.get(position_id)
        return pos is not None and pos.is_open
    
    # ============================================
    # Уведомления
    # ============================================
    
    def on_state_change(
        self,
        handler: Callable[[ManagedPosition, PositionState], None],
    ) -> None:
        """
        Регистрирует обработчик изменения состояния позиции.
        
        Вызывается при каждом изменении состояния.
        """
        self._state_change_handlers.append(handler)
    
    def _notify_state_change(
        self,
        position: ManagedPosition,
        new_state: PositionState,
    ) -> None:
        """Уведомляет подписчиков об изменении состояния."""
        if not self._config.notify_on_state_change:
            return
        
        for handler in self._state_change_handlers:
            try:
                handler(position, new_state)
            except Exception:
                pass  # Ошибка в обработчике не должна ронять менеджер
    
    # ============================================
    # Статистика
    # ============================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику менеджера."""
        return {
            "total_open_attempts": self._total_open_attempts,
            "total_open_success": self._total_open_success,
            "total_close_attempts": self._total_close_attempts,
            "total_close_success": self._total_close_success,
            "open_positions": len(self.get_open_positions()),
            "total_positions": len(self._positions),
            "open_success_rate": (
                self._total_open_success / self._total_open_attempts
                if self._total_open_attempts > 0 else 0.0
            ),
        }
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    def _report_incident(
        self,
        incident_type: str,
        symbol: str,
        details: Dict[str, Any],
        severity: str = "WARNING",
    ) -> None:
        """Логирует инцидент."""
        if self._incident_reporter is not None:
            self._incident_reporter.report_incident(
                incident_type=incident_type,
                symbol=symbol,
                details=details,
                severity=severity,
            )
        else:
            # Fallback: печать в консоль
            print(
                f"[ORDER_MANAGER] {severity}: {incident_type} "
                f"symbol={symbol} {details}"
            )