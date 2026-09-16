"""
Отслеживание активных ордеров.

Отвечает за:
- Отслеживание всех активных ордеров (входы, стопы, тейки)
- Периодическую проверку статуса ордеров
- Обработку исполнения/отмены/истечения
- Уведомление о событиях ордеров через коллбэки
- Синхронизацию состояния с биржей

Используется в связке с:
- ExecutionEngine (исполнение ордеров)
- PositionManager (управление позициями)
- StopsManager (стопы и тейки)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from proscalper.core.types import OrderSide
from proscalper.execution.execution_engine import (
    ExchangeAdapter,
    OrderStatus,
)


class OrderPurpose(Enum):
    """
    Назначение ордера.
    
    Определяет, какую роль играет ордер в стратегии.
    """
    ENTRY = auto()              # Входной ордер
    STOP_LOSS = auto()          # Стоп-лосс
    TAKE_PROFIT = auto()        # Тейк-профит
    TRAILING_STOP = auto()      # Трейлинг-стоп
    PARTIAL_CLOSE = auto()      # Частичное закрытие позиции


class OrderEventType(Enum):
    """
    Тип события ордера.
    
    Используется для уведомления подписчиков.
    """
    FILLED = auto()             # Ордер исполнен
    PARTIALLY_FILLED = auto()   # Ордер частично исполнен
    CANCELLED = auto()          # Ордер отменён
    REJECTED = auto()           # Ордер отклонён
    EXPIRED = auto()            # Ордер истёк
    STATUS_CHANGED = auto()     # Статус изменился


@dataclass
class TrackedOrder:
    """
    Отслеживаемый ордер.
    
    Хранит информацию об ордере и его текущем состоянии.
    """
    # Идентификация
    order_id: str
    symbol: str
    purpose: OrderPurpose
    
    # Параметры ордера
    side: OrderSide
    quantity: float = 0.0
    price: float = 0.0
    
    # Состояние
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    
    # Время
    created_ts_ns: int = 0
    last_check_ts_ns: int = 0
    filled_ts_ns: int = 0
    
    # Связь с позицией
    position_id: Optional[str] = None
    
    # Дополнительные данные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Ордер активен (не исполнен и не отменён)."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )
    
    @property
    def is_filled(self) -> bool:
        """Ордер полностью исполнен."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_cancelled(self) -> bool:
        """Ордер отменён."""
        return self.status == OrderStatus.CANCELLED
    
    @property
    def age_ms(self) -> float:
        """Возраст ордера в миллисекундах."""
        if self.created_ts_ns == 0:
            return 0.0
        return (time.time_ns() - self.created_ts_ns) / 1_000_000


@dataclass
class OrderTrackerConfig:
    """
    Конфигурация трекера ордеров.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    # Интервал опроса
    poll_interval_ms: int = 100         # интервал проверки ордеров
    
    # Ограничения
    max_tracked_orders: int = 100       # максимум отслеживаемых ордеров
    stale_order_timeout_ms: int = 30000 # таймаут для устаревших ордеров
    
    # Поведение
    auto_untrack_filled: bool = True    # автоматически удалять исполненные
    auto_untrack_cancelled: bool = True # автоматически удалять отменённые


@dataclass
class OrderEvent:
    """
    Событие ордера.
    
    Передаётся в коллбэки при изменении состояния ордера.
    """
    event_type: OrderEventType
    order: TrackedOrder
    ts_ns: int = 0
    
    # Для событий исполнения
    filled_quantity: float = 0.0
    average_price: float = 0.0


class OrderTracker:
    """
    Трекер активных ордеров.
    
    Отслеживает все активные ордера и периодически проверяет их статус.
    При изменении состояния уведомляет подписчиков через коллбэки.
    
    Использование:
        tracker = OrderTracker(adapter=binance_adapter)
        
        # Регистрируем обработчик событий
        tracker.on_order_event(my_handler)
        
        # Начинаем отслеживание ордера
        tracker.track_order(
            order_id="12345",
            symbol="BTCUSDT",
            purpose=OrderPurpose.ENTRY,
            side=OrderSide.BUY,
        )
        
        # Запускаем цикл проверки
        await tracker.run()
    """
    
    def __init__(
        self,
        adapter: ExchangeAdapter,
        config: Optional[OrderTrackerConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or OrderTrackerConfig()
        
        # Отслеживаемые ордера
        self._orders: Dict[str, TrackedOrder] = {}
        
        # Подписчики на события
        self._event_handlers: List[Callable[[OrderEvent], None]] = []
        
        # Флаг работы
        self._running = False
    
    def track_order(
        self,
        order_id: str,
        symbol: str,
        purpose: OrderPurpose,
        side: OrderSide,
        quantity: float = 0.0,
        price: float = 0.0,
        position_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackedOrder]:
        """
        Начинает отслеживание ордера.
        
        Вызывается после отправки ордера на биржу.
        
        Возвращает TrackedOrder или None, если лимит превышен.
        """
        # Проверяем лимит
        if len(self._orders) >= self.config.max_tracked_orders:
            return None
        
        now_ns = time.time_ns()
        
        order = TrackedOrder(
            order_id=order_id,
            symbol=symbol,
            purpose=purpose,
            side=side,
            quantity=quantity,
            price=price,
            status=OrderStatus.SUBMITTED,
            created_ts_ns=now_ns,
            last_check_ts_ns=now_ns,
            position_id=position_id,
            metadata=metadata or {},
        )
        
        self._orders[order_id] = order
        
        return order
    
    def untrack_order(self, order_id: str) -> None:
        """Прекращает отслеживание ордера."""
        if order_id in self._orders:
            del self._orders[order_id]
    
    def get_order(self, order_id: str) -> Optional[TrackedOrder]:
        """Возвращает отслеживаемый ордер по ID."""
        return self._orders.get(order_id)
    
    def get_active_orders(self) -> List[TrackedOrder]:
        """Возвращает все активные ордера."""
        return [
            order for order in self._orders.values()
            if order.is_active
        ]
    
    def get_orders_by_symbol(self, symbol: str) -> List[TrackedOrder]:
        """Возвращает все ордера для символа."""
        return [
            order for order in self._orders.values()
            if order.symbol == symbol.upper()
        ]
    
    def get_orders_by_position(self, position_id: str) -> List[TrackedOrder]:
        """Возвращает все ордера для позиции."""
        return [
            order for order in self._orders.values()
            if order.position_id == position_id
        ]
    
    def get_filled_orders(self) -> List[TrackedOrder]:
        """Возвращает все исполненные ордера."""
        return [
            order for order in self._orders.values()
            if order.is_filled
        ]
    
    def get_cancelled_orders(self) -> List[TrackedOrder]:
        """Возвращает все отменённые ордера."""
        return [
            order for order in self._orders.values()
            if order.is_cancelled
        ]
    
    def on_order_event(
        self,
        handler: Callable[[OrderEvent], None],
    ) -> None:
        """
        Регистрирует обработчик событий ордеров.
        
        Обработчик вызывается при любом изменении состояния ордера.
        """
        self._event_handlers.append(handler)
    
    async def run(self) -> None:
        """
        Запускает цикл проверки ордеров.
        
        Работает до вызова stop().
        """
        self._running = True
        
        while self._running:
            try:
                await self.check_all_orders()
            except Exception:
                pass  # Логируем ошибку, но не прерываем цикл
            
            await asyncio.sleep(self.config.poll_interval_ms / 1000)
    
    def stop(self) -> None:
        """Останавливает цикл проверки."""
        self._running = False
    
    async def check_all_orders(self) -> None:
        """
        Проверяет статус всех активных ордеров.
        
        Вызывается периодически из цикла run().
        """
        now_ns = time.time_ns()
        
        for order_id, order in list(self._orders.items()):
            # Проверяем только активные ордера
            if not order.is_active:
                continue
            
            # Проверяем, не устарел ли ордер
            age_ms = (now_ns - order.created_ts_ns) / 1_000_000
            if age_ms > self.config.stale_order_timeout_ms:
                order.status = OrderStatus.EXPIRED
                self._emit_event(OrderEventType.EXPIRED, order)
                continue
            
            # Проверяем статус на бирже
            try:
                new_status = await self.adapter.get_order_status(
                    order.symbol, order.order_id
                )
                
                # Если статус изменился
                if new_status != order.status:
                    old_status = order.status
                    order.status = new_status
                    order.last_check_ts_ns = now_ns
                    
                    # Обновляем информацию об исполнении
                    if new_status in (
                        OrderStatus.FILLED,
                        OrderStatus.PARTIALLY_FILLED,
                    ):
                        fill_info = await self.adapter.get_order_fill_info(
                            order.symbol, order.order_id
                        )
                        order.filled_quantity = fill_info.get("filled_quantity", 0.0)
                        order.average_price = fill_info.get("average_price", 0.0)
                        
                        if new_status == OrderStatus.FILLED:
                            order.filled_ts_ns = now_ns
                    
                    # Уведомляем о событии
                    self._emit_status_change(order, old_status, new_status)
            
            except Exception:
                pass  # Логируем ошибку, но не прерываем проверку
    
    async def check_single_order(self, order_id: str) -> Optional[OrderStatus]:
        """
        Проверяет статус одного ордера.
        
        Возвращает новый статус или None, если ордер не найден.
        """
        order = self._orders.get(order_id)
        if order is None:
            return None
        
        try:
            new_status = await self.adapter.get_order_status(
                order.symbol, order_id
            )
            
            if new_status != order.status:
                old_status = order.status
                order.status = new_status
                order.last_check_ts_ns = time.time_ns()
                
                self._emit_status_change(order, old_status, new_status)
            
            return new_status
        
        except Exception:
            return None
    
    def _emit_event(
        self,
        event_type: OrderEventType,
        order: TrackedOrder,
    ) -> None:
        """Уведомляет подписчиков о событии."""
        event = OrderEvent(
            event_type=event_type,
            order=order,
            ts_ns=time.time_ns(),
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
        )
        
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception:
                pass  # Логируем ошибку, но не прерываем уведомление
    
    def _emit_status_change(
        self,
        order: TrackedOrder,
        old_status: OrderStatus,
        new_status: OrderStatus,
    ) -> None:
        """Уведомляет об изменении статуса."""
        # Определяем тип события
        if new_status == OrderStatus.FILLED:
            event_type = OrderEventType.FILLED
        elif new_status == OrderStatus.PARTIALLY_FILLED:
            event_type = OrderEventType.PARTIALLY_FILLED
        elif new_status == OrderStatus.CANCELLED:
            event_type = OrderEventType.CANCELLED
        elif new_status == OrderStatus.REJECTED:
            event_type = OrderEventType.REJECTED
        elif new_status == OrderStatus.EXPIRED:
            event_type = OrderEventType.EXPIRED
        else:
            event_type = OrderEventType.STATUS_CHANGED
        
        self._emit_event(event_type, order)
        
        # Автоматически удаляем исполненные/отменённые ордера
        if self.config.auto_untrack_filled and new_status == OrderStatus.FILLED:
            self.untrack_order(order.order_id)
        elif self.config.auto_untrack_cancelled and new_status == OrderStatus.CANCELLED:
            self.untrack_order(order.order_id)
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику трекера."""
        active = len(self.get_active_orders())
        filled = len(self.get_filled_orders())
        cancelled = len(self.get_cancelled_orders())
        
        return {
            "total_tracked": len(self._orders),
            "active_orders": active,
            "filled_orders": filled,
            "cancelled_orders": cancelled,
            "event_handlers": len(self._event_handlers),
        }
    
    def reset(self) -> None:
        """Сбрасывает состояние трекера."""
        self._orders.clear()
        self._event_handlers.clear()
        self._running = False