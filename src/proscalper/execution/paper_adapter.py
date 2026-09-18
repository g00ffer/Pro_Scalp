"""Adapter exposing the existing paper simulator through the execution port."""
from __future__ import annotations

from typing import Optional

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.executor import ExecutionCallback, ExecutionEvent
from proscalper.execution.order_state import Fill, OrderLifecycle
from proscalper.execution.paper import PaperExecutor, PaperFillEvent, PaperOrder


class PaperOrderExecutor:
    """Translate the existing paper simulator into canonical execution events."""

    def __init__(self, paper: PaperExecutor) -> None:
        self.paper = paper
        self._metadata: dict[str, tuple[str, str, bool]] = {}
        self._callback: Optional[ExecutionCallback] = None
        self.paper.on_fill(self._handle_fill)
        on_reject = getattr(self.paper, "on_reject", None)
        if on_reject is not None:
            on_reject(self._handle_reject)

    def submit_market(
        self,
        *,
        intent_id: str,
        position_id: str,
        symbol: Symbol,
        side: OrderSide,
        quantity: float,
        signal_id: str,
        entry_type: EntryType = EntryType.MARKET,
        stop_price: float | None = None,
        closing: bool = False,
    ) -> str:
        if entry_type != EntryType.MARKET:
            raise ValueError(f"unsupported paper entry type: {entry_type}")
        if quantity <= 0:
            raise ValueError("order quantity must be positive")
        if stop_price is not None:
            raise ValueError("stop_price is not supported by submit_market")
        order = self.paper.submit_market_order(
            symbol=str(symbol),
            side=side,
            quantity=quantity,
            signal_id=signal_id,
            leg_type="close" if closing else "entry",
            reason=f"intent:{intent_id}",
        )
        self._metadata[order.order_id] = (intent_id, position_id, closing)
        return order.order_id

    def submit_stop(
        self,
        *,
        intent_id: str,
        position_id: str,
        symbol: Symbol,
        side: OrderSide,
        quantity: float,
        signal_id: str,
        stop_price: float,
    ) -> str:
        if quantity <= 0:
            raise ValueError("stop quantity must be positive")
        if stop_price <= 0:
            raise ValueError("stop_price must be positive")
        order = self.paper.submit_stop_order(
            symbol=str(symbol),
            side=side,
            quantity=quantity,
            stop_price=stop_price,
            signal_id=signal_id,
            leg_type="stop",
            reason=f"protection:{intent_id}",
        )
        # A protective stop is a closing order, but has its own intent identity.
        self._metadata[order.order_id] = (intent_id, position_id, True)
        return order.order_id

    def cancel(self, order_id: str) -> bool:
        return self.paper.cancel_order(order_id)

    def set_event_callback(self, callback: ExecutionCallback) -> None:
        self._callback = callback

    def _handle_fill(self, event: PaperFillEvent) -> None:
        metadata = self._metadata.get(event.order.order_id)
        if metadata is None or self._callback is None:
            return
        intent_id, position_id, closing = metadata
        result = event.fill_result
        if result.filled_qty <= 0:
            return
        fill = Fill(
            fill_id=f"{event.order.order_id}:{event.ts_ns}:{result.filled_qty}",
            order_id=event.order.order_id,
            price=result.avg_fill_price,
            quantity=result.filled_qty,
            fee=result.fees,
            timestamp_ns=event.ts_ns,
        )
        lifecycle = (
            OrderLifecycle.FILLED
            if result.filled_qty >= result.requested_qty - 1e-12
            else OrderLifecycle.PARTIALLY_FILLED
        )
        self._callback(ExecutionEvent(
            order_id=event.order.order_id,
            intent_id=intent_id,
            position_id=position_id,
            lifecycle=lifecycle,
            fill=fill,
            closing=closing,
        ))

    def _handle_reject(self, order: PaperOrder, reason: str) -> None:
        metadata = self._metadata.get(order.order_id)
        if metadata is None or self._callback is None:
            return
        intent_id, position_id, closing = metadata
        self._callback(ExecutionEvent(
            order_id=order.order_id,
            intent_id=intent_id,
            position_id=position_id,
            lifecycle=OrderLifecycle.REJECTED,
            closing=closing,
            reject_reason=reason,
        ))
