"""
Protection Watchdog: защита от незащищённых позиций.

Решает критическую проблему скальпинга — окно незащищённого риска
между исполнением входа и постановкой защитного стопа.

Логика:
1. При получении сигнала на вход вызывается arm()
2. Watchdog ждёт исполнения входа (заполнение ордера)
3. Как только есть исполненный объём — ставит защитный стоп
4. Если стоп не поставлен за лимит — аварийное закрытие позиции
5. Если позиция не исполнилась за лимит — отмена

Критичные состояния позиции:
- PENDING_ENTRY → PARTIALLY_FILLED_UNPROTECTED → PROTECTED (норма)
- FILLED_UNPROTECTED → PROTECTION_FAILED → ERROR_FLATTEN (авария)

Самое опасное состояние: FILLED_UNPROTECTED.
Оно должно иметь максимальный приоритет обработки.

Используется в связке с:
- BracketExecutor (инициирует защиту)
- OrderBuilder (построение стоп-ордеров)
- IncidentJournal (логирование инцидентов)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol

from proscalper.core.events import Signal
from proscalper.core.types import (
    ExecutionStatus,
    OrderSide,
    OrderStatus,
    Priority,
)


# ============================================================
# Состояния защиты
# ============================================================

class ProtectionState(str, Enum):
    """Состояние защиты позиции."""
    PENDING_ENTRY = "PENDING_ENTRY"
    PARTIALLY_FILLED_UNPROTECTED = "PARTIALLY_FILLED_UNPROTECTED"
    FILLED_UNPROTECTED = "FILLED_UNPROTECTED"
    PROTECTED = "PROTECTED"
    PROTECTION_FAILED = "PROTECTION_FAILED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR_FLATTEN = "ERROR_FLATTEN"


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class ProtectionConfig:
    """
    Конфигурация Protection Watchdog.
    
    Параметры из формализации системы (архитектура v2):
    - max_unprotected_ms: 120 (максимум времени без защиты)
    - stop_retry_interval_ms: 20 (интервал повторных попыток)
    - max_stop_retries: 5 (максимум попыток)
    - emergency_flatten_on_protection_failure: true
    """
    # Таймауты
    max_unprotected_ms: int = 120       # максимум времени без защиты
    max_entry_wait_ms: int = 5000       # максимум ожидания исполнения входа
    poll_interval_ms: int = 5           # интервал опроса состояния
    
    # Повторные попытки стопа
    stop_retry_interval_ms: int = 20    # интервал между попытками
    max_stop_retries: int = 5           # максимум попыток
    
    # Аварийное поведение
    emergency_flatten_on_failure: bool = True
    disable_symbol_on_repeated_failure: bool = True
    max_consecutive_failures: int = 3   # максимум последовательных отказов
    
    # Ограничения
    max_concurrent_monitors: int = 10   # максимум одновременных мониторингов


# ============================================================
# Интерфейсы для внешних зависимостей
# ============================================================

class OrderPlacementClient(Protocol):
    """Протокол для отправки ордеров."""
    
    async def place_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        client_order_id: str,
    ) -> "StopOrderResponse":
        """Отправляет защитный стоп-ордер."""
        ...
    
    async def place_market_close(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        client_order_id: str,
    ) -> "CloseOrderResponse":
        """Отправляет рыночное закрытие позиции."""
        ...
    
    async def cancel_order(
        self,
        symbol: str,
        client_order_id: str,
    ) -> bool:
        """Отменяет ордер."""
        ...


class PositionStateProvider(Protocol):
    """Протокол для получения состояния позиции."""
    
    def filled_quantity(self, symbol: str) -> float:
        """Возвращает исполненный объём позиции."""
        ...
    
    def entry_price(self, symbol: str) -> float:
        """Возвращает среднюю цену входа."""
        ...
    
    def mark_protected(
        self,
        symbol: str,
        stop_order_id: str,
        protected_after_ms: float,
    ) -> None:
        """Помечает позицию как защищённую."""
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


@dataclass
class StopOrderResponse:
    """Ответ на отправку стоп-ордера."""
    accepted: bool
    order_id: str = ""
    error_code: int = 0
    error_message: str = ""


@dataclass
class CloseOrderResponse:
    """Ответ на отправку рыночного закрытия."""
    accepted: bool
    order_id: str = ""
    filled_qty: float = 0.0
    error_message: str = ""


# ============================================================
# Задача мониторинга защиты
# ============================================================

@dataclass
class ProtectionTask:
    """
    Задача мониторинга защиты для одной позиции.
    
    Хранит состояние и историю попыток защиты.
    """
    # Идентификация
    signal_id: str
    symbol: str
    side: OrderSide
    
    # Ожидаемые параметры
    expected_quantity: float
    stop_price: float
    
    # Время
    armed_ts_ns: int = 0
    first_fill_ts_ns: int = 0
    protected_ts_ns: int = 0
    
    # Приоритет
    priority: Priority = Priority.NORMAL
    
    # Состояние
    state: ProtectionState = ProtectionState.PENDING_ENTRY
    attempts: int = 0
    last_attempt_ts_ns: int = 0
    
    # Результат
    stop_order_id: Optional[str] = None
    protected_after_ms: float = 0.0
    
    # Задача мониторинга
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    
    @property
    def is_active(self) -> bool:
        """Задача мониторинга активна."""
        return self._task is not None and not self._task.done()
    
    @property
    def unprotected_ms(self) -> float:
        """Время без защиты в миллисекундах."""
        if self.first_fill_ts_ns == 0:
            return 0.0
        now_ns = time.time_ns()
        return (now_ns - self.first_fill_ts_ns) / 1_000_000
    
    @property
    def is_overdue(self) -> bool:
        """Время защиты превышено."""
        return self.unprotected_ms > 0  # проверяется с конфигом


# ============================================================
# Основной класс
# ============================================================

class ProtectionWatchdog:
    """
    Watchdog для защиты позиций.
    
    Гарантирует, что каждая открытая позиция будет защищена
    стоп-ордером в лимитированное время. Если защита не
    поставлена — аварийное закрытие позиции.
    
    Использование:
        watchdog = ProtectionWatchdog(
            order_client=order_client,
            position_provider=position_provider,
            incident_reporter=incident_reporter,
            config=ProtectionConfig(),
        )
        
        # При открытии позиции
        await watchdog.arm(
            signal=signal,
            expected_quantity=quantity,
            stop_price=stop_price,
            priority=Priority.CRITICAL,
        )
        
        # При успешной постановке стопа (вызывается извне)
        watchdog.disarm(signal_id)
    """
    
    def __init__(
        self,
        order_client: OrderPlacementClient,
        position_provider: PositionStateProvider,
        incident_reporter: Optional[IncidentReporter] = None,
        config: Optional[ProtectionConfig] = None,
    ) -> None:
        self._order_client = order_client
        self._position_provider = position_provider
        self._incident_reporter = incident_reporter
        self._config = config or ProtectionConfig()
        
        # Активные задачи мониторинга
        self._tasks: Dict[str, ProtectionTask] = {}
        
        # Счётчики отказов по символам
        self._failure_counts: Dict[str, int] = {}
        
        # Отключённые символы
        self._disabled_symbols: set = set()
        
        # Статистика
        self._total_armed: int = 0
        self._total_protected: int = 0
        self._total_emergency_flattened: int = 0
        self._total_timeouts: int = 0
    
    # ============================================
    # Публичный интерфейс
    # ============================================
    
    async def arm(
        self,
        signal: Signal,
        expected_quantity: float,
        stop_price: float,
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """
        Начинает мониторинг защиты для сигнала.
        
        Вызывается из BracketExecutor после отправки входа.
        
        Возвращает True, если мониторинг запущен.
        """
        # Проверяем лимит одновременных мониторингов
        active_count = sum(
            1 for t in self._tasks.values() if t.is_active
        )
        if active_count >= self._config.max_concurrent_monitors:
            self._report_incident(
                "WATCHDOG_OVERLOADED",
                signal.symbol,
                {"active_count": active_count},
                severity="ERROR",
            )
            return False
        
        # Проверяем, не отключён ли символ
        if signal.symbol in self._disabled_symbols:
            self._report_incident(
                "SYMBOL_DISABLED",
                signal.symbol,
                {"reason": "repeated_protection_failures"},
                severity="WARNING",
            )
            return False
        
        # Создаём задачу
        task = ProtectionTask(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            expected_quantity=expected_quantity,
            stop_price=stop_price,
            armed_ts_ns=time.time_ns(),
            priority=priority,
        )
        
        self._tasks[signal.signal_id] = task
        self._total_armed += 1
        
        # Запускаем асинхронный мониторинг
        task._task = asyncio.create_task(
            self._protect_until_success(task)
        )
        
        return True
    
    def disarm(self, signal_id: str) -> None:
        """
        Прекращает мониторинг (защита поставлена успешно).
        
        Вызывается извне, когда стоп-ордер подтверждён.
        """
        task = self._tasks.get(signal_id)
        if task is not None:
            task.state = ProtectionState.PROTECTED
            task.protected_ts_ns = time.time_ns()
            
            # Отменяем задачу мониторинга
            if task._task is not None and not task._task.done():
                task._task.cancel()
            
            # Очищаем
            del self._tasks[signal_id]
    
    def get_unprotected_positions(self) -> List[ProtectionTask]:
        """Возвращает все незащищённые позиции."""
        return [
            task for task in self._tasks.values()
            if task.state in (
                ProtectionState.PARTIALLY_FILLED_UNPROTECTED,
                ProtectionState.FILLED_UNPROTECTED,
            )
        ]
    
    def get_task(self, signal_id: str) -> Optional[ProtectionTask]:
        """Возвращает задачу по signal_id."""
        return self._tasks.get(signal_id)
    
    def is_symbol_disabled(self, symbol: str) -> bool:
        """Проверяет, отключён ли символ."""
        return symbol in self._disabled_symbols
    
    def enable_symbol(self, symbol: str) -> None:
        """Включает символ после отключения."""
        self._disabled_symbols.discard(symbol)
        self._failure_counts.pop(symbol, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику watchdog."""
        return {
            "total_armed": self._total_armed,
            "total_protected": self._total_protected,
            "total_emergency_flattened": self._total_emergency_flattened,
            "total_timeouts": self._total_timeouts,
            "active_monitors": len([
                t for t in self._tasks.values() if t.is_active
            ]),
            "unprotected_positions": len(self.get_unprotected_positions()),
            "disabled_symbols": list(self._disabled_symbols),
            "protection_rate": (
                self._total_protected / self._total_armed
                if self._total_armed > 0 else 0.0
            ),
        }
    
    # ============================================
    # Внутренняя логика мониторинга
    # ============================================
    
    async def _protect_until_success(self, task: ProtectionTask) -> None:
        """
        Основной цикл мониторинга защиты.
        
        Логика:
        1. Ждём исполнения входа
        2. Как только есть исполненный объём — ставим стоп
        3. Если стоп отклонён — повторяем с интервалом
        4. Если превышен лимит — аварийное закрытие
        """
        started_ms = time.time_ns() / 1_000_000
        
        try:
            while True:
                # Получаем текущий исполненный объём
                filled_qty = self._position_provider.filled_quantity(task.symbol)
                
                # Если объём ещё не исполнен
                if filled_qty <= 0:
                    elapsed_ms = time.time_ns() / 1_000_000 - started_ms
                    
                    # Проверяем таймаут ожидания входа
                    if elapsed_ms > self._config.max_entry_wait_ms:
                        task.state = ProtectionState.PROTECTION_FAILED
                        self._total_timeouts += 1
                        
                        self._report_incident(
                            "ENTRY_NO_FILL_TIMEOUT",
                            task.symbol,
                            {
                                "signal_id": task.signal_id,
                                "waited_ms": elapsed_ms,
                            },
                            severity="WARNING",
                        )
                        
                        # Отменяем входной ордер
                        await self._cancel_entry(task)
                        return
                    
                    await asyncio.sleep(self._config.poll_interval_ms / 1000)
                    continue
                
                # Объём исполнен — фиксируем время первого исполнения
                if task.first_fill_ts_ns == 0:
                    task.first_fill_ts_ns = time.time_ns()
                    task.state = ProtectionState.FILLED_UNPROTECTED
                
                # Проверяем, не превышен ли лимит времени без защиты
                unprotected_ms = task.unprotected_ms
                if unprotected_ms > self._config.max_unprotected_ms:
                    # Аварийное закрытие
                    task.state = ProtectionState.PROTECTION_FAILED
                    
                    if self._config.emergency_flatten_on_failure:
                        await self._emergency_flatten(
                            task,
                            reason="PROTECTION_TIMEOUT",
                        )
                    
                    self._record_failure(task.symbol)
                    return
                
                # Пытаемся поставить защитный стоп
                stop_response = await self._attempt_place_stop(task, filled_qty)
                
                if stop_response.accepted:
                    # Успех!
                    task.state = ProtectionState.PROTECTED
                    task.stop_order_id = stop_response.order_id
                    task.protected_ts_ns = time.time_ns()
                    task.protected_after_ms = unprotected_ms
                    
                    # Помечаем позицию как защищённую
                    self._position_provider.mark_protected(
                        symbol=task.symbol,
                        stop_order_id=stop_response.order_id,
                        protected_after_ms=unprotected_ms,
                    )
                    
                    self._total_protected += 1
                    self._clear_failure_count(task.symbol)
                    
                    # Удаляем задачу
                    self._tasks.pop(task.signal_id, None)
                    return
                
                # Стоп отклонён — логируем и повторяем
                task.attempts += 1
                task.last_attempt_ts_ns = time.time_ns()
                
                self._report_incident(
                    "STOP_PLACE_FAILED",
                    task.symbol,
                    {
                        "signal_id": task.signal_id,
                        "attempt": task.attempts,
                        "error_code": stop_response.error_code,
                        "error_message": stop_response.error_message,
                    },
                    severity="WARNING",
                )
                
                # Проверяем лимит попыток
                if task.attempts >= self._config.max_stop_retries:
                    task.state = ProtectionState.PROTECTION_FAILED
                    
                    if self._config.emergency_flatten_on_failure:
                        await self._emergency_flatten(
                            task,
                            reason="MAX_STOP_RETRIES_EXCEEDED",
                        )
                    
                    self._record_failure(task.symbol)
                    return
                
                # Ждём перед следующей попыткой
                await asyncio.sleep(self._config.stop_retry_interval_ms / 1000)
        
        except asyncio.CancelledError:
            # Задача отменена (например, извне вызван disarm)
            return
        except Exception as exc:
            # Непредвиденная ошибка
            task.state = ProtectionState.PROTECTION_FAILED
            
            self._report_incident(
                "WATCHDOG_UNEXPECTED_ERROR",
                task.symbol,
                {
                    "signal_id": task.signal_id,
                    "error": str(exc),
                },
                severity="ERROR",
            )
    
    async def _attempt_place_stop(
        self,
        task: ProtectionTask,
        filled_qty: float,
    ) -> StopOrderResponse:
        """Пытается поставить защитный стоп-ордер."""
        try:
            # Сторона ордера противоположна позиции
            close_side = (
                OrderSide.SELL if task.side == OrderSide.BUY
                else OrderSide.BUY
            )
            
            # Генерируем уникальный client_order_id
            client_order_id = f"{task.signal_id}:stop:{task.attempts}"
            
            response = await self._order_client.place_stop_order(
                symbol=task.symbol,
                side=close_side,
                quantity=filled_qty,
                stop_price=task.stop_price,
                client_order_id=client_order_id,
            )
            
            return response
        
        except Exception as exc:
            return StopOrderResponse(
                accepted=False,
                error_message=str(exc),
            )
    
    async def _emergency_flatten(
        self,
        task: ProtectionTask,
        reason: str,
    ) -> None:
        """
        Аварийное закрытие позиции.
        
        Вызывается когда защита не может быть поставлена.
        """
        task.state = ProtectionState.ERROR_FLATTEN
        self._total_emergency_flattened += 1
        
        self._report_incident(
            "EMERGENCY_FLATTEN",
            task.symbol,
            {
                "signal_id": task.signal_id,
                "reason": reason,
                "attempts": task.attempts,
                "unprotected_ms": task.unprotected_ms,
            },
            severity="CRITICAL",
        )
        
        try:
            # Получаем текущий объём
            filled_qty = self._position_provider.filled_quantity(task.symbol)
            
            if filled_qty > 0:
                # Сторона ордера противоположна позиции
                close_side = (
                    OrderSide.SELL if task.side == OrderSide.BUY
                    else OrderSide.BUY
                )
                
                client_order_id = f"{task.signal_id}:emergency_close"
                
                await self._order_client.place_market_close(
                    symbol=task.symbol,
                    side=close_side,
                    quantity=filled_qty,
                    client_order_id=client_order_id,
                )
        
        except Exception as exc:
            self._report_incident(
                "EMERGENCY_FLATTEN_FAILED",
                task.symbol,
                {
                    "signal_id": task.signal_id,
                    "error": str(exc),
                },
                severity="CRITICAL",
            )
    
    async def _cancel_entry(self, task: ProtectionTask) -> None:
        """Отменяет входной ордер при таймауте."""
        try:
            client_order_id = f"{task.signal_id}:entry"
            await self._order_client.cancel_order(
                symbol=task.symbol,
                client_order_id=client_order_id,
            )
        except Exception:
            pass  # Ошибка отмены не критична
    
    # ============================================
    # Управление отказами
    # ============================================
    
    def _record_failure(self, symbol: str) -> None:
        """Записывает отказ для символа."""
        self._failure_counts[symbol] = self._failure_counts.get(symbol, 0) + 1
        
        # Проверяем, нужно ли отключить символ
        if self._config.disable_symbol_on_repeated_failure:
            if self._failure_counts[symbol] >= self._config.max_consecutive_failures:
                self._disabled_symbols.add(symbol)
                
                self._report_incident(
                    "SYMBOL_DISABLED",
                    symbol,
                    {
                        "consecutive_failures": self._failure_counts[symbol],
                        "reason": "repeated_protection_failures",
                    },
                    severity="ERROR",
                )
    
    def _clear_failure_count(self, symbol: str) -> None:
        """Сбрасывает счётчик отказов при успешной защите."""
        self._failure_counts.pop(symbol, None)
    
    # ============================================
    # Логирование
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
                f"[WATCHDOG] {severity}: {incident_type} "
                f"symbol={symbol} {details}"
            )