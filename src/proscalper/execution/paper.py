"""
Бумажное исполнение ордеров (Paper Execution).

Симулирует исполнение ордеров на живых данных без отправки
на биржу. Использует:
- Реальный стакан (через FastOrderBook)
- MatchingEngine из бэктеста (проскальзывание, частичные филлы)
- Модель задержек (латентность сети и биржи)
- Учёт комиссий

Отличие от бэктеста:
- Бэктест работает на записанных данных (история)
- Paper работает на живых данных (реальное время)

Используется в связке с:
- app/paper_runner.py (бумажная торговля)
- app/replay_runner.py (прогон на записанных данных)
- backtest/matching.py (движок матчинга)
- backtest/latency.py (модель задержек)
- backtest/costs.py (комиссии)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from proscalper.backtest.costs import CostCalculator, FeeConfig
from proscalper.backtest.latency import (
    LatencyBreakdown,
    LatencyModel,
    ProfileLatency,
    Region,
    ms_to_ns,
)
from proscalper.backtest.matching import (
    FillResult,
    MatchingEngine,
    MatchStatus,
)
from proscalper.core.types import OrderSide, OrderStatus, OrderType
from proscalper.market_data.orderbook_fast import FastOrderBook


# ============================================================
# Типы
# ============================================================

class PaperOrderState(str, Enum):
    """Состояние бумажного ордера."""
    PENDING = "PENDING"           # Ожидает применения (в очереди задержки)
    FILLED = "FILLED"             # Полностью исполнен
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Частично исполнен
    REJECTED = "REJECTED"         # Отклонён
    CANCELLED = "CANCELLED"       # Отменён


@dataclass
class PaperOrder:
    """Бумажный ордер."""
    # Идентификация
    order_id: str
    client_order_id: str
    signal_id: str
    symbol: str
    
    # Параметры
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None       # для LIMIT
    stop_price: Optional[float] = None  # для STOP
    
    # Состояние
    state: PaperOrderState = PaperOrderState.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    
    # Издержки
    fees: float = 0.0
    slippage_ticks: float = 0.0
    slippage_cost: float = 0.0
    
    # Время
    created_ts_ns: int = 0
    filled_ts_ns: int = 0
    
    # Метаданные
    leg_type: str = ""  # "entry", "stop", "close"
    reason: str = ""
    
    @property
    def is_filled(self) -> bool:
        return self.state == PaperOrderState.FILLED
    
    @property
    def notional(self) -> float:
        return self.filled_qty * self.avg_fill_price


@dataclass
class PaperExecutionConfig:
    """Конфигурация бумажного исполнения."""
    # Задержки
    latency_region: Region = Region.TOKYO
    
    # Комиссии (по умолчанию Binance Futures)
    taker_fee_pct: float = 0.04
    maker_fee_pct: float = 0.02
    
    # Матчинг
    depth_for_matching: int = 50
    
    # Поведение
    simulate_latency: bool = True
    simulate_slippage: bool = True


# ============================================================
# Callback'и
# ============================================================

@dataclass
class PaperFillEvent:
    """Событие исполнения бумажного ордера."""
    order: PaperOrder
    fill_result: FillResult
    latency_breakdown: LatencyBreakdown
    ts_ns: int


# ============================================================
# Основной класс
# ============================================================

class PaperExecutor:
    """
    Бумажный исполнитель ордеров.
    
    Симулирует исполнение на живом стакане:
    1. Получает ордер
    2. Применяет модель задержек
    3. Через latency исполняет ордер по текущему стакану
    4. Возвращает результат с учётом проскальзывания и комиссий
    
    Использование:
        executor = PaperExecutor(
            tick_size=0.1,
            config=PaperExecutionConfig(),
        )
        
        # Привязываем стакан
        executor.bind_book(books["BTCUSDT"])
        
        # Отправляем ордер
        order = executor.submit_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=0.001,
            signal_id="sig_123",
            leg_type="entry",
        )
        
        # Ожидаем исполнения (через tick)
        await asyncio.sleep(0.01)
        executor.tick()
        
        if order.is_filled:
            print(f"Исполнено: {order.filled_qty} @ {order.avg_fill_price}")
    """
    
    def __init__(
        self,
        tick_size: float,
        config: Optional[PaperExecutionConfig] = None,
    ) -> None:
        self.config = config or PaperExecutionConfig()
        self.tick_size = tick_size
        
        # Движок матчинга
        self.costs = CostCalculator(
            tick_size=tick_size,
            fee_config=FeeConfig(
                taker_fee_pct=self.config.taker_fee_pct,
                maker_fee_pct=self.config.maker_fee_pct,
            ),
        )
        self.matching = MatchingEngine(
            tick_size=tick_size,
            cost_calculator=self.costs,
        )
        
        # Модель задержек
        self.latency: LatencyModel = ProfileLatency(
            region=self.config.latency_region,
        )
        
        # Стаканы (привязанные)
        self._books: Dict[str, FastOrderBook] = {}
        
        # Активные ордера
        self._orders: Dict[str, PaperOrder] = {}
        
        # Очередь отложенных исполнений (по задержке)
        # (fill_ts_ns, order_id)
        self._pending_fills: List[tuple] = []
        
        # Callback'и
        self._on_fill: Optional[Callable[[PaperFillEvent], None]] = None
        self._on_reject: Optional[Callable[[PaperOrder, str], None]] = None
        
        # Статистика
        self._total_submitted: int = 0
        self._total_filled: int = 0
        self._total_rejected: int = 0
    
    # ============================================
    # Привязка стаканов
    # ============================================
    
    def bind_book(self, book: FastOrderBook) -> None:
        """Привязывает стакан для исполнения."""
        self._books[book.symbol.upper()] = book
    
    def unbind_book(self, symbol: str) -> None:
        """Отвязывает стакан."""
        self._books.pop(symbol.upper(), None)
    
    # ============================================
    # Callback'и
    # ============================================
    
    def on_fill(self, callback: Callable[[PaperFillEvent], None]) -> None:
        """Регистрирует обработчик исполнения."""
        self._on_fill = callback
    
    def on_reject(self, callback: Callable[[PaperOrder, str], None]) -> None:
        """Регистрирует обработчик отклонения."""
        self._on_reject = callback
    
    # ============================================
    # Отправка ордеров
    # ============================================
    
    def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        signal_id: str = "",
        leg_type: str = "entry",
        reason: str = "",
    ) -> PaperOrder:
        """
        Отправляет MARKET-ордер.
        
        Возвращает объект ордера. Исполнение произойдёт
        после применения задержки (в методе tick()).
        """
        return self._submit_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            signal_id=signal_id,
            leg_type=leg_type,
            reason=reason,
        )
    
    def submit_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        signal_id: str = "",
        leg_type: str = "entry",
        reason: str = "",
    ) -> PaperOrder:
        """Отправляет LIMIT-ордер (мейкер)."""
        return self._submit_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            signal_id=signal_id,
            leg_type=leg_type,
            reason=reason,
        )
    
    def submit_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        signal_id: str = "",
        leg_type: str = "stop",
        reason: str = "",
    ) -> PaperOrder:
        """Отправляет STOP_MARKET-ордер."""
        return self._submit_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            signal_id=signal_id,
            leg_type=leg_type,
            reason=reason,
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """Отменяет ордер (если он ещё не исполнен)."""
        order = self._orders.get(order_id)
        if order is None:
            return False
        
        if order.state in (PaperOrderState.FILLED, PaperOrderState.CANCELLED):
            return False
        
        order.state = PaperOrderState.CANCELLED
        return True
    
    # ============================================
    # Обработка времени (вызывается периодически)
    # ============================================
    
    def tick(self, now_ns: Optional[int] = None) -> List[PaperFillEvent]:
        """
        Обрабатывает отложенные исполнения.
        
        Вызывается периодически (например, каждые 10мс).
        Возвращает список событий исполнения за этот тик.
        """
        if now_ns is None:
            now_ns = time.time_ns()
        
        events: List[PaperFillEvent] = []
        
        # Проверяем стопы (триггер по цене)
        self._check_stop_triggers(now_ns)
        
        # Применяем отложенные исполнения
        while self._pending_fills:
            # Сортируем по времени (минимальное время — первый)
            self._pending_fills.sort(key=lambda x: x[0])
            
            fill_ts_ns, order_id = self._pending_fills[0]
            
            if fill_ts_ns > now_ns:
                break  # Время ещё не пришло
            
            self._pending_fills.pop(0)
            
            order = self._orders.get(order_id)
            if order is None:
                continue
            
            if order.state != PaperOrderState.PENDING:
                continue
            
            # Исполняем по текущему стакану
            event = self._execute_order(order, now_ns)
            if event is not None:
                events.append(event)
        
        return events
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    def _submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        signal_id: str = "",
        leg_type: str = "entry",
        reason: str = "",
    ) -> PaperOrder:
        """Создаёт и ставит ордер в очередь."""
        now_ns = time.time_ns()
        
        order = PaperOrder(
            order_id=f"paper_{uuid.uuid4().hex[:12]}",
            client_order_id=f"paper_{uuid.uuid4().hex[:12]}",
            signal_id=signal_id,
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            state=PaperOrderState.PENDING,
            created_ts_ns=now_ns,
            leg_type=leg_type,
            reason=reason,
        )
        
        self._orders[order.order_id] = order
        self._total_submitted += 1
        
        # Для MARKET/LIMIT — сразу ставим в очередь исполнения
        # Для STOP — ждём триггера
        if order_type != OrderType.STOP_MARKET:
            self._schedule_execution(order, now_ns)
        
        return order
    
    def _schedule_execution(self, order: PaperOrder, now_ns: int) -> None:
        """Ставит ордер в очередь исполнения с учётом задержки."""
        if self.config.simulate_latency:
            breakdown = self.latency.sample()
            delay_ns = ms_to_ns(breakdown.submit_to_fill_ms)
        else:
            delay_ns = 0
        
        fill_ts_ns = now_ns + delay_ns
        self._pending_fills.append((fill_ts_ns, order.order_id))
    
    def _execute_order(
        self,
        order: PaperOrder,
        now_ns: int,
    ) -> Optional[PaperFillEvent]:
        """Исполняет ордер по текущему стакану."""
        book = self._books.get(order.symbol)
        if book is None:
            order.state = PaperOrderState.REJECTED
            self._total_rejected += 1
            if self._on_reject is not None:
                self._on_reject(order, "NO_BOOK")
            return None
        
        # Получаем стакан
        depth = self.config.depth_for_matching
        bids = book.top_bids(depth)
        asks = book.top_asks(depth)
        
        is_buy = (order.side == OrderSide.BUY)
        
        # Матчинг в зависимости от типа ордера
        if order.order_type == OrderType.MARKET:
            fill = self.matching.match_market(
                order_qty=order.quantity,
                bids=bids,
                asks=asks,
                is_buy=is_buy,
            )
        elif order.order_type == OrderType.LIMIT:
            fill = self.matching.match_marketable_limit_ioc(
                order_qty=order.quantity,
                limit_price=order.price or 0.0,
                bids=bids,
                asks=asks,
                is_buy=is_buy,
            )
        elif order.order_type == OrderType.STOP_MARKET:
            best_bid = bids[0][0] if bids else 0.0
            best_ask = asks[0][0] if asks else 0.0
            fill = self.matching.match_stop_market(
                order_qty=order.quantity,
                stop_price=order.stop_price or 0.0,
                best_bid=best_bid,
                best_ask=best_ask,
                bids=bids,
                asks=asks,
                is_buy=is_buy,
            )
        else:
            order.state = PaperOrderState.REJECTED
            self._total_rejected += 1
            return None
        
        # Обрабатываем результат
        if fill.status == MatchStatus.REJECTED:
            order.state = PaperOrderState.REJECTED
            self._total_rejected += 1
            if self._on_reject is not None:
                self._on_reject(order, fill.reject_reason or "REJECTED")
            return None
        
        if fill.status == MatchStatus.SKIPPED:
            order.state = PaperOrderState.REJECTED
            self._total_rejected += 1
            if self._on_reject is not None:
                self._on_reject(order, fill.reject_reason or "SKIPPED")
            return None
        
        # Применяем исполнение
        order.filled_qty = fill.filled_qty
        order.avg_fill_price = fill.avg_fill_price
        order.fees = fill.fees
        order.slippage_ticks = fill.slippage_ticks
        order.slippage_cost = fill.slippage_cost
        order.filled_ts_ns = now_ns
        
        if fill.status == MatchStatus.FILLED:
            order.state = PaperOrderState.FILLED
        elif fill.status == MatchStatus.PARTIALLY_FILLED:
            order.state = PaperOrderState.PARTIALLY_FILLED
        
        self._total_filled += 1
        
        # Получаем модель задержки для события
        breakdown = self.latency.sample() if self.config.simulate_latency else LatencyBreakdown(
            signal_processing_ms=0,
            network_roundtrip_ms=0,
            exchange_matching_ms=0,
            ack_ms=0,
        )
        
        event = PaperFillEvent(
            order=order,
            fill_result=fill,
            latency_breakdown=breakdown,
            ts_ns=now_ns,
        )
        
        if self._on_fill is not None:
            self._on_fill(event)
        
        return event
    
    def _check_stop_triggers(self, now_ns: int) -> None:
        """Проверяет триггеры стоп-ордеров по текущей цене."""
        for order in list(self._orders.values()):
            if order.order_type != OrderType.STOP_MARKET:
                continue
            
            if order.state != PaperOrderState.PENDING:
                continue
            
            book = self._books.get(order.symbol)
            if book is None:
                continue
            
            # Получаем текущую цену
            mid_price = book.mid_price
            if mid_price is None:
                continue
            
            # Проверяем триггер
            triggered = False
            if order.side == OrderSide.BUY:
                # Стоп на покупку (закрытие шорта): триггер выше
                if order.stop_price is not None and mid_price >= order.stop_price:
                    triggered = True
            else:
                # Стоп на продажу (закрытие лонга): триггер ниже
                if order.stop_price is not None and mid_price <= order.stop_price:
                    triggered = True
            
            if triggered:
                self._schedule_execution(order, now_ns)
    
    # ============================================
    # Запросы состояния
    # ============================================
    
    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Возвращает ордер по ID."""
        return self._orders.get(order_id)
    
    def get_active_orders(self) -> List[PaperOrder]:
        """Возвращает все активные (неисполненные) ордера."""
        return [
            o for o in self._orders.values()
            if o.state == PaperOrderState.PENDING
        ]
    
    def get_filled_orders(self) -> List[PaperOrder]:
        """Возвращает все исполненные ордера."""
        return [
            o for o in self._orders.values()
            if o.state == PaperOrderState.FILLED
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика исполнения."""
        return {
            "total_submitted": self._total_submitted,
            "total_filled": self._total_filled,
            "total_rejected": self._total_rejected,
            "active_orders": len(self.get_active_orders()),
            "pending_fills": len(self._pending_fills),
        }
    
    def reset(self) -> None:
        """Сбрасывает состояние."""
        self._orders.clear()
        self._pending_fills.clear()
        self._total_submitted = 0
        self._total_filled = 0
        self._total_rejected = 0