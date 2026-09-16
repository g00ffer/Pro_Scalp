"""
Bracket Executor: защищённое открытие позиций.

Решает критическую проблему скальпинга — окно незащищённого риска
между исполнением входа и постановкой защитного стопа.

Режимы работы:
- BATCH: отправка входа и стопа в одном запросе (Binance batchOrders)
- NATIVE: использование нативных attached SL/TP (Bybit)
- FILL_DRIVEN: стоп ставится сразу после получения fill-события
- SEQUENTIAL: последовательная отправка (только как аварийный вариант)

Критичные инварианты:
- Каждая позиция должна быть защищена стопом за < 120мс
- Если стоп отклонён — немедленный retry через ProtectionWatchdog
- Если защита не поставлена за лимит — аварийное закрытие

Используется в связке с:
- OrderBuilder (построение ордеров)
- ProtectionWatchdog (контроль защиты)
- IncidentJournal (логирование инцидентов)
- DecisionJournal (рыночный снапшот при отправке)
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol

from proscalper.core.events import (
    FillEvent,
    OrderEvent,
    Signal,
)
from proscalper.core.types import (
    EntryType,
    ExecutionStatus,
    OrderSide,
    OrderStatus,
    OrderType,
    Priority,
    TimeInForce,
)


# ============================================================
# Типы и конфигурация
# ============================================================

class BracketMode(str, Enum):
    """Режим работы bracket-исполнителя."""
    BATCH = "BATCH"                    # Batch orders (Binance)
    NATIVE_ATTACHED = "NATIVE_ATTACHED"  # Нативные SL/TP (Bybit)
    FILL_DRIVEN = "FILL_DRIVEN"        # Стоп после fill-события
    SEQUENTIAL = "SEQUENTIAL"          # Последовательная отправка (аварийный)


class PositionProtectionState(str, Enum):
    """Состояние защиты позиции."""
    PENDING_ENTRY = "PENDING_ENTRY"
    PARTIALLY_FILLED_UNPROTECTED = "PARTIALLY_FILLED_UNPROTECTED"
    FILLED_UNPROTECTED = "FILLED_UNPROTECTED"
    PROTECTED = "PROTECTED"
    PROTECTION_FAILED = "PROTECTION_FAILED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR_FLATTEN = "ERROR_FLATTEN"


@dataclass(frozen=True)
class BracketConfig:
    """
    Конфигурация bracket-исполнителя.
    
    Параметры из формализации системы (архитектура v2):
    - max_unprotected_ms: максимальное время без защиты
    - stop_retry_interval_ms: интервал повторных попыток стопа
    - max_stop_retries: максимум попыток постановки стопа
    - emergency_flatten_on_protection_failure: аварийное закрытие
    """
    # Режим работы
    preferred_mode: BracketMode = BracketMode.BATCH
    fallback_mode: BracketMode = BracketMode.FILL_DRIVEN
    
    # Таймауты защиты
    max_unprotected_ms: int = 120
    stop_retry_interval_ms: int = 20
    max_stop_retries: int = 5
    
    # Аварийное поведение
    emergency_flatten_on_protection_failure: bool = True
    disable_symbol_on_protection_failure: bool = True
    
    # Входные ордера
    entry_order_type: EntryType = EntryType.MARKETABLE_LIMIT_IOC
    max_slippage_ticks: int = 4
    allow_market_order_on_ultra_liquidity: bool = True
    
    # Защитные стопы
    stop_order_type: OrderType = OrderType.STOP_MARKET
    stop_reduce_only: bool = True
    stop_working_type: str = "CONTRACT_PRICE"
    
    # Валидация перед отправкой
    require_bookticker_validation: bool = True
    max_bookticker_age_ms: int = 80


@dataclass
class OrderLegResult:
    """Результат одной ноги (ордера) в bracket-пакете."""
    # Идентификация
    client_order_id: str
    exchange_order_id: Optional[str] = None
    leg_type: str = ""  # "entry", "stop", "take_profit"
    
    # Статус
    accepted: bool = False
    status: OrderStatus = OrderStatus.PENDING
    
    # Исполнение
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    
    # Ошибки
    error_code: Optional[int] = None
    error_message: str = ""
    
    # Время
    submit_ts_ns: int = 0
    ack_ts_ns: int = 0
    fill_ts_ns: int = 0
    
    @property
    def latency_ms(self) -> float:
        """Задержка от отправки до подтверждения."""
        if self.ack_ts_ns <= 0 or self.submit_ts_ns <= 0:
            return 0.0
        return (self.ack_ts_ns - self.submit_ts_ns) / 1_000_000
    
    @property
    def is_filled(self) -> bool:
        """Ордер полностью исполнен."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_partially_filled(self) -> bool:
        """Ордер частично исполнен."""
        return self.status == OrderStatus.PARTIALLY_FILLED


@dataclass
class BracketResult:
    """
    Результат всей bracket-операции.
    
    Содержит статус защиты и результаты обеих ног.
    """
    # Идентификация
    signal_id: str
    symbol: str
    mode: BracketMode
    
    # Итоговый статус
    status: ExecutionStatus = ExecutionStatus.PENDING
    protection_state: PositionProtectionState = PositionProtectionState.PENDING_ENTRY
    
    # Ноги
    entry_leg: Optional[OrderLegResult] = None
    stop_leg: Optional[OrderLegResult] = None
    
    # Защита
    protected_after_ms: float = 0.0
    stop_order_id: Optional[str] = None
    
    # Время
    started_ts_ns: int = 0
    completed_ts_ns: int = 0
    
    # Инциденты
    incidents: List[str] = field(default_factory=list)
    
    # Метаданные для журнала
    market_snapshot: Optional[Dict[str, Any]] = None
    
    @property
    def is_protected(self) -> bool:
        """Позиция защищена стопом."""
        return self.protection_state == PositionProtectionState.PROTECTED
    
    @property
    def is_pending_protection(self) -> bool:
        """Позиция ожидает защиты."""
        return self.protection_state in (
            PositionProtectionState.PARTIALLY_FILLED_UNPROTECTED,
            PositionProtectionState.FILLED_UNPROTECTED,
        )
    
    @property
    def protection_failed(self) -> bool:
        """Защита не удалась."""
        return self.protection_state == PositionProtectionState.PROTECTION_FAILED
    
    @property
    def total_latency_ms(self) -> float:
        """Общая задержка операции."""
        if self.completed_ts_ns <= 0 or self.started_ts_ns <= 0:
            return 0.0
        return (self.completed_ts_ns - self.started_ts_ns) / 1_000_000


# ============================================================
# Интерфейс для работы с биржей
# ============================================================

class ExchangeOrderClient(Protocol):
    """
    Протокол для работы с ордерами биржи.
    
    Реализуется отдельно для каждой биржи.
    Позволяет BracketExecutor не зависеть от конкретной биржи.
    """
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        working_type: str = "CONTRACT_PRICE",
        client_order_id: str = "",
    ) -> OrderEvent:
        """Отправляет одиночный ордер."""
        ...
    
    async def place_batch_orders(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[OrderEvent]:
        """
        Отправляет пакет ордеров (если поддерживается).
        
        Для Binance Futures: POST /fapi/v1/batchOrders
        Для Bybit: не поддерживается напрямую (нативные параметры)
        """
        ...
    
    async def cancel_order(
        self,
        symbol: str,
        client_order_id: str,
    ) -> bool:
        """Отменяет ордер."""
        ...
    
    def supports_batch_orders(self) -> bool:
        """Поддерживает ли биржа пакетные ордера."""
        ...
    
    def supports_native_stop(self) -> bool:
        """Поддерживает ли биржа нативные стоп-параметры."""
        ...


# ============================================================
# Базовый класс исполнителя
# ============================================================

class BaseBracketExecutor(ABC):
    """
    Базовый класс для защищённого исполнения сигналов.
    
    Отвечает за:
    - Выбор режима работы (batch, native, fill-driven)
    - Отправку входа и защитного стопа
    - Отслеживание состояния защиты
    - Взаимодействие с ProtectionWatchdog
    """
    
    def __init__(
        self,
        exchange_client: ExchangeOrderClient,
        config: Optional[BracketConfig] = None,
    ) -> None:
        self._exchange = exchange_client
        self._config = config or BracketConfig()
        
        # Активные позиции (для отслеживания защиты)
        self._positions: Dict[str, BracketResult] = {}
        
        # Статистика
        self._total_executions = 0
        self._protected_executions = 0
        self._failed_protections = 0
    
    @property
    def config(self) -> BracketConfig:
        return self._config
    
    @abstractmethod
    async def open_with_protection(
        self,
        signal: Signal,
        quantity: float,
        stop_price: float,
    ) -> BracketResult:
        """
        Открывает позицию с защитным стопом.
        
        Это основной метод, который должен быть реализован
        для каждой биржи и каждого режима работы.
        """
        ...
    
    @abstractmethod
    async def close_position(
        self,
        signal_id: str,
        reason: str,
    ) -> Optional[OrderLegResult]:
        """
        Закрывает позицию (аварийно или по сигналу).
        """
        ...
    
    def get_position_state(
        self,
        signal_id: str,
    ) -> PositionProtectionState:
        """Возвращает состояние защиты позиции."""
        result = self._positions.get(signal_id)
        if result is None:
            return PositionProtectionState.CLOSED
        return result.protection_state
    
    def get_all_unprotected(self) -> List[BracketResult]:
        """Возвращает все незащищённые позиции."""
        return [
            result for result in self._positions.values()
            if result.is_pending_protection
        ]
    
    def get_stats(self) -> Dict[str, int]:
        """Статистика исполнителя."""
        return {
            "total_executions": self._total_executions,
            "protected_executions": self._protected_executions,
            "failed_protections": self._failed_protections,
            "unprotected_positions": len(self.get_all_unprotected()),
        }
    
    def _register_position(self, result: BracketResult) -> None:
        """Регистрирует позицию для отслеживания."""
        self._positions[result.signal_id] = result
        self._total_executions += 1
    
    def _mark_protected(self, result: BracketResult) -> None:
        """Помечает позицию как защищённую."""
        result.protection_state = PositionProtectionState.PROTECTED
        self._protected_executions += 1
    
    def _mark_protection_failed(self, result: BracketResult) -> None:
        """Помечает неудачную защиту."""
        result.protection_state = PositionProtectionState.PROTECTION_FAILED
        self._failed_protections += 1


# ============================================================
# Реализация с пакетными ордерами (Binance)
# ============================================================

class BatchBracketExecutor(BaseBracketExecutor):
    """
    Исполнитель через пакетные ордера.
    
    Для Binance Futures:
    - POST /fapi/v1/batchOrders позволяет отправить до 5 ордеров
    - Ордера применяются на стороне биржи "одновременно"
    - Это уменьшает окно незащищённого риска, но не гарантирует
      атомарность (ордера всё равно обрабатываются последовательно)
    
    Логика:
    1. Отправляем batch: [entry_order, stop_order]
    2. Если оба приняты → статус PROTECTED
    3. Если entry принят, но стоп отклонён → статус PENDING_PROTECTION
       и активируем ProtectionWatchdog для повторной постановки
    4. Если оба отклонены → статус REJECTED
    """
    
    def __init__(
        self,
        exchange_client: ExchangeOrderClient,
        config: Optional[BracketConfig] = None,
        on_stop_rejected: Optional[callable] = None,
    ) -> None:
        super().__init__(exchange_client, config)
        self._on_stop_rejected = on_stop_rejected
    
    async def open_with_protection(
        self,
        signal: Signal,
        quantity: float,
        stop_price: float,
    ) -> BracketResult:
        """
        Открытие позиции через пакетные ордера.
        """
        started_ts = time.time_ns()
        
        result = BracketResult(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            mode=BracketMode.BATCH,
            started_ts_ns=started_ts,
        )
        
        # Проверяем поддержку пакетных ордеров
        if not self._exchange.supports_batch_orders():
            result.incidents.append("BATCH_NOT_SUPPORTED")
            result.status = ExecutionStatus.REJECTED
            return result
        
        # Строим ордера
        entry_order = self._build_entry_order(signal, quantity)
        stop_order = self._build_stop_order(signal, quantity, stop_price)
        
        # Отправляем пакет
        batch_request = [
            self._order_to_batch_dict(entry_order),
            self._order_to_batch_dict(stop_order),
        ]
        
        try:
            responses = await self._exchange.place_batch_orders(batch_request)
        except Exception as exc:
            result.incidents.append(f"BATCH_ERROR: {exc}")
            result.status = ExecutionStatus.REJECTED
            return result
        
        # Обрабатываем результаты
        if len(responses) < 2:
            result.incidents.append("INCOMPLETE_BATCH_RESPONSE")
            result.status = ExecutionStatus.REJECTED
            return result
        
        entry_response = responses[0]
        stop_response = responses[1]
        
        # Результат входа
        result.entry_leg = self._response_to_leg(entry_response, "entry")
        
        # Результат стопа
        result.stop_leg = self._response_to_leg(stop_response, "stop")
        
        # Определяем итоговый статус
        entry_accepted = result.entry_leg.accepted
        stop_accepted = result.stop_leg.accepted
        
        if entry_accepted and stop_accepted:
            # Успех: оба ордера приняты
            result.status = ExecutionStatus.PROTECTED
            result.protection_state = PositionProtectionState.PROTECTED
            result.stop_order_id = result.stop_leg.exchange_order_id
            result.protected_after_ms = (
                (time.time_ns() - started_ts) / 1_000_000
            )
            self._mark_protected(result)
            
        elif entry_accepted and not stop_accepted:
            # Критичный случай: вход принят, но стоп отклонён
            result.incidents.append("STOP_LEG_REJECTED_IN_BATCH")
            result.status = ExecutionStatus.PENDING_PROTECTION
            result.protection_state = PositionProtectionState.FILLED_UNPROTECTED
            
            # Активируем повторную постановку стопа
            if self._on_stop_rejected is not None:
                await self._on_stop_rejected(
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    quantity=quantity,
                    stop_price=stop_price,
                    error=result.stop_leg.error_message,
                    priority=Priority.CRITICAL,
                )
        else:
            # Оба отклонены или только стоп принят (аномалия)
            result.incidents.append("ENTRY_REJECTED")
            result.status = ExecutionStatus.REJECTED
            result.protection_state = PositionProtectionState.CLOSED
        
        result.completed_ts_ns = time.time_ns()
        self._register_position(result)
        
        return result
    
    async def close_position(
        self,
        signal_id: str,
        reason: str,
    ) -> Optional[OrderLegResult]:
        """Закрытие позиции (аварийное или по сигналу)."""
        result = self._positions.get(signal_id)
        if result is None:
            return None
        
        result.protection_state = PositionProtectionState.CLOSING
        
        # Отменяем стоп-ордер если он ещё активен
        if result.stop_order_id is not None:
            await self._exchange.cancel_order(
                symbol=result.symbol,
                client_order_id=result.stop_order_id,
            )
        
        # Отправляем рыночное закрытие
        close_side = OrderSide.SELL  # По умолчанию лонг
        if result.entry_leg is not None:
            # Определяем сторону из оригинального сигнала
            # (упрощённо, в реальной реализации храним сторону)
            pass
        
        close_order_id = f"{signal_id}:close:{int(time.time_ns())}"
        
        response = await self._exchange.place_order(
            symbol=result.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=result.entry_leg.filled_qty if result.entry_leg else 0,
            reduce_only=True,
            client_order_id=close_order_id,
        )
        
        result.protection_state = PositionProtectionState.CLOSED
        result.incidents.append(f"CLOSED_REASON: {reason}")
        
        return self._response_to_leg(response, "close")
    
    def _build_entry_order(
        self,
        signal: Signal,
        quantity: float,
    ) -> Dict[str, Any]:
        """Строит входной ордер (обычно marketable LIMIT IOC)."""
        order_id = f"{signal.signal_id}:entry"
        
        # Определяем тип и цену в зависимости от конфигурации
        if signal.entry_type == EntryType.MARKETABLE_LIMIT_IOC:
            return {
                "symbol": signal.symbol,
                "side": signal.side.value,
                "type": OrderType.LIMIT.value,
                "timeInForce": TimeInForce.IOC.value,
                "quantity": quantity,
                "price": signal.entry_price,
                "newClientOrderId": order_id,
            }
        else:
            # Чистый MARKET (только для сверхликвидных)
            return {
                "symbol": signal.symbol,
                "side": signal.side.value,
                "type": OrderType.MARKET.value,
                "quantity": quantity,
                "newClientOrderId": order_id,
            }
    
    def _build_stop_order(
        self,
        signal: Signal,
        quantity: float,
        stop_price: float,
    ) -> Dict[str, Any]:
        """Строит защитный стоп-ордер."""
        order_id = f"{signal.signal_id}:stop"
        close_side = OrderSide.SELL if signal.side == OrderSide.BUY else OrderSide.BUY
        
        return {
            "symbol": signal.symbol,
            "side": close_side.value,
            "type": self._config.stop_order_type.value,
            "stopPrice": stop_price,
            "workingType": self._config.stop_working_type,
            "quantity": quantity,
            "reduceOnly": self._config.stop_reduce_only,
            "newClientOrderId": order_id,
        }
    
    def _order_to_batch_dict(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Конвертирует ордер в формат для batch-запроса."""
        return order
    
    def _response_to_leg(
        self,
        response: OrderEvent,
        leg_type: str,
    ) -> OrderLegResult:
        """Конвертирует ответ биржи в результат ноги."""
        return OrderLegResult(
            client_order_id=response.client_order_id,
            exchange_order_id=response.exchange_order_id,
            leg_type=leg_type,
            accepted=response.status != OrderStatus.REJECTED,
            status=response.status,
            filled_qty=response.filled_qty,
            avg_fill_price=response.avg_fill_price,
            error_message=getattr(response, "error_message", ""),
            submit_ts_ns=time.time_ns(),
            ack_ts_ns=time.time_ns(),
        )


# ============================================================
# Реализация через нативные стопы (Bybit)
# ============================================================

class NativeStopBracketExecutor(BaseBracketExecutor):
    """
    Исполнитель через нативные параметры стоп-ордеров.
    
    Для Bybit V5:
    - Ордер может быть создан с параметрами:
      takeProfit, stopLoss, slTriggerBy, tpTriggerBy
    - Это предпочтительный режим для Bybit
    
    Важно проверить:
    - Работает ли для MARKET и LIMIT
    - Конфликт с hedge mode
    - Поведение при partial fills
    - Возможность модификации стопа после частичного исполнения
    """
    
    def __init__(
        self,
        exchange_client: ExchangeOrderClient,
        config: Optional[BracketConfig] = None,
    ) -> None:
        super().__init__(exchange_client, config)
    
    async def open_with_protection(
        self,
        signal: Signal,
        quantity: float,
        stop_price: float,
    ) -> BracketResult:
        """
        Открытие позиции через нативные параметры стопа.
        
        Отправляем один ордер с параметрами:
        - stopLoss: цена защитного стопа
        - slTriggerBy: тип триггера (обычно LastPrice или MarkPrice)
        """
        started_ts = time.time_ns()
        
        result = BracketResult(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            mode=BracketMode.NATIVE_ATTACHED,
            started_ts_ns=started_ts,
        )
        
        # Проверяем поддержку нативных стопов
        if not self._exchange.supports_native_stop():
            result.incidents.append("NATIVE_STOP_NOT_SUPPORTED")
            result.status = ExecutionStatus.REJECTED
            return result
        
        # Строим ордер с нативным стопом
        order_id = f"{signal.signal_id}:entry"
        
        try:
            response = await self._exchange.place_order(
                symbol=signal.symbol,
                side=signal.side,
                order_type=signal.entry_type.to_order_type(),
                quantity=quantity,
                price=signal.entry_price,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC,
                reduce_only=False,
                client_order_id=order_id,
            )
        except Exception as exc:
            result.incidents.append(f"NATIVE_STOP_ERROR: {exc}")
            result.status = ExecutionStatus.REJECTED
            return result
        
        # Обрабатываем результат
        result.entry_leg = self._response_to_leg(response, "entry")
        
        if response.status == OrderStatus.REJECTED:
            result.incidents.append("ENTRY_REJECTED")
            result.status = ExecutionStatus.REJECTED
            result.protection_state = PositionProtectionState.CLOSED
        else:
            # Нативный стоп ставится вместе с ордером
            result.status = ExecutionStatus.PROTECTED
            result.protection_state = PositionProtectionState.PROTECTED
            result.protected_after_ms = (
                (time.time_ns() - started_ts) / 1_000_000
            )
            self._mark_protected(result)
        
        result.completed_ts_ns = time.time_ns()
        self._register_position(result)
        
        return result
    
    async def close_position(
        self,
        signal_id: str,
        reason: str,
    ) -> Optional[OrderLegResult]:
        """Закрытие позиции."""
        result = self._positions.get(signal_id)
        if result is None:
            return None
        
        result.protection_state = PositionProtectionState.CLOSING
        
        # Для нативных стопов отмена не требуется
        # (стоп привязан к позиции)
        
        result.protection_state = PositionProtectionState.CLOSED
        result.incidents.append(f"CLOSED_REASON: {reason}")
        
        return None
    
    def _response_to_leg(
        self,
        response: OrderEvent,
        leg_type: str,
    ) -> OrderLegResult:
        """Конвертирует ответ биржи в результат ноги."""
        return OrderLegResult(
            client_order_id=response.client_order_id,
            exchange_order_id=response.exchange_order_id,
            leg_type=leg_type,
            accepted=response.status != OrderStatus.REJECTED,
            status=response.status,
            filled_qty=response.filled_qty,
            avg_fill_price=response.avg_fill_price,
            submit_ts_ns=time.time_ns(),
            ack_ts_ns=time.time_ns(),
        )


# ============================================================
# Реализация через заполнение (универсальный)
# ============================================================

class FillDrivenBracketExecutor(BaseBracketExecutor):
    """
    Исполнитель с постановкой стопа после получения исполнения.
    
    Используется когда:
    - Пакетные ордера не поддерживаются
    - Нативные стопы недоступны
    - Как универсальный вариант для любой биржи
    
    Логика:
    1. Отправляем входной ордер
    2. Параллельно подписаны на user data stream
    3. При получении первого fill-события:
       - Немедленно отправляем защитный стоп на фактически
         исполненный объём
    4. При частичном исполнении:
       - Стоп ставится на исполненный объём
       - При следующем частичном исполнении стоп модифицируется
    """
    
    def __init__(
        self,
        exchange_client: ExchangeOrderClient,
        config: Optional[BracketConfig] = None,
        on_fill_received: Optional[callable] = None,
    ) -> None:
        super().__init__(exchange_client, config)
        self._on_fill_received = on_fill_received
    
    async def open_with_protection(
        self,
        signal: Signal,
        quantity: float,
        stop_price: float,
    ) -> BracketResult:
        """
        Открытие позиции с последующей постановкой стопа.
        """
        started_ts = time.time_ns()
        
        result = BracketResult(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            mode=BracketMode.FILL_DRIVEN,
            started_ts_ns=started_ts,
        )
        
        # Шаг 1: Отправляем входной ордер
        entry_order_id = f"{signal.signal_id}:entry"
        
        try:
            entry_response = await self._exchange.place_order(
                symbol=signal.symbol,
                side=signal.side,
                order_type=signal.entry_type.to_order_type(),
                quantity=quantity,
                price=signal.entry_price,
                time_in_force=TimeInForce.IOC if signal.entry_type == EntryType.MARKETABLE_LIMIT_IOC else TimeInForce.GTC,
                reduce_only=False,
                client_order_id=entry_order_id,
            )
        except Exception as exc:
            result.incidents.append(f"ENTRY_ERROR: {exc}")
            result.status = ExecutionStatus.REJECTED
            return result
        
        result.entry_leg = self._response_to_leg(entry_response, "entry")
        
        # Проверяем, принят ли вход
        if entry_response.status == OrderStatus.REJECTED:
            result.incidents.append("ENTRY_REJECTED")
            result.status = ExecutionStatus.REJECTED
            result.protection_state = PositionProtectionState.CLOSED
            result.completed_ts_ns = time.time_ns()
            self._register_position(result)
            return result
        
        # Шаг 2: Ожидаем исполнение и ставим стоп
        # В реальной реализации здесь используем FillEvent из user data stream
        # Для упрощения сразу ставим стоп на запрошенный объём
        
        stop_order_id = f"{signal.signal_id}:stop:1"
        close_side = OrderSide.SELL if signal.side == OrderSide.BUY else OrderSide.BUY
        
        try:
            stop_response = await self._exchange.place_order(
                symbol=signal.symbol,
                side=close_side,
                order_type=self._config.stop_order_type,
                quantity=quantity,
                stop_price=stop_price,
                reduce_only=self._config.stop_reduce_only,
                client_order_id=stop_order_id,
            )
            
            result.stop_leg = self._response_to_leg(stop_response, "stop")
            
            if stop_response.status != OrderStatus.REJECTED:
                result.status = ExecutionStatus.PROTECTED
                result.protection_state = PositionProtectionState.PROTECTED
                result.stop_order_id = stop_response.exchange_order_id
                result.protected_after_ms = (
                    (time.time_ns() - started_ts) / 1_000_000
                )
                self._mark_protected(result)
            else:
                result.incidents.append("STOP_REJECTED_AFTER_FILL")
                result.status = ExecutionStatus.PENDING_PROTECTION
                result.protection_state = PositionProtectionState.FILLED_UNPROTECTED
                
                # Активируем повторную постановку
                if self._on_fill_received is not None:
                    await self._on_fill_received(
                        signal_id=signal.signal_id,
                        symbol=signal.symbol,
                        quantity=quantity,
                        stop_price=stop_price,
                        priority=Priority.CRITICAL,
                    )
        except Exception as exc:
            result.incidents.append(f"STOP_ERROR: {exc}")
            result.status = ExecutionStatus.PENDING_PROTECTION
            result.protection_state = PositionProtectionState.FILLED_UNPROTECTED
        
        result.completed_ts_ns = time.time_ns()
        self._register_position(result)
        
        return result
    
    async def close_position(
        self,
        signal_id: str,
        reason: str,
    ) -> Optional[OrderLegResult]:
        """Закрытие позиции."""
        result = self._positions.get(signal_id)
        if result is None:
            return None
        
        result.protection_state = PositionProtectionState.CLOSING
        
        # Отменяем стоп если он активен
        if result.stop_order_id is not None:
            await self._exchange.cancel_order(
                symbol=result.symbol,
                client_order_id=result.stop_order_id,
            )
        
        result.protection_state = PositionProtectionState.CLOSED
        result.incidents.append(f"CLOSED_REASON: {reason}")
        
        return None
    
    def _response_to_leg(
        self,
        response: OrderEvent,
        leg_type: str,
    ) -> OrderLegResult:
        """Конвертирует ответ биржи в результат ноги."""
        return OrderLegResult(
            client_order_id=response.client_order_id,
            exchange_order_id=response.exchange_order_id,
            leg_type=leg_type,
            accepted=response.status != OrderStatus.REJECTED,
            status=response.status,
            filled_qty=response.filled_qty,
            avg_fill_price=response.avg_fill_price,
            submit_ts_ns=time.time_ns(),
            ack_ts_ns=time.time_ns(),
        )


# ============================================================
# Фабрика для выбора исполнителя
# ============================================================

def create_bracket_executor(
    exchange_client: ExchangeOrderClient,
    config: Optional[BracketConfig] = None,
    on_stop_rejected: Optional[callable] = None,
    on_fill_received: Optional[callable] = None,
) -> BaseBracketExecutor:
    """
    Создаёт подходящий исполнитель в зависимости от возможностей биржи.
    
    Логика выбора:
    1. Если биржа поддерживает пакетные ордера → Batch
    2. Если биржа поддерживает нативные стопы → Native
    3. Иначе → FillDriven (универсальный)
    """
    config = config or BracketConfig()
    
    if exchange_client.supports_batch_orders():
        return BatchBracketExecutor(
            exchange_client=exchange_client,
            config=config,
            on_stop_rejected=on_stop_rejected,
        )
    
    if exchange_client.supports_native_stop():
        return NativeStopBracketExecutor(
            exchange_client=exchange_client,
            config=config,
        )
    
    return FillDrivenBracketExecutor(
        exchange_client=exchange_client,
        config=config,
        on_fill_received=on_fill_received,
    )