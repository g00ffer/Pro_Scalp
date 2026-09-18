"""Venue-neutral execution adapter contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.order_state import Fill, OrderLifecycle


@dataclass(frozen=True)
class ExecutionEvent:
    """Normalized order lifecycle event emitted by an executor adapter."""

    order_id: str
    intent_id: str
    position_id: str
    lifecycle: OrderLifecycle
    fill: Fill | None = None
    closing: bool = False
    reject_reason: str | None = None


ExecutionCallback = Callable[[ExecutionEvent], None]


class OrderExecutor(Protocol):
    """Minimal interface shared by paper and exchange executors."""

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
        """Submit a market order and return the venue order id."""
        ...

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
        """Submit a protective STOP_MARKET order and return its venue id."""
        ...

    def cancel(self, order_id: str) -> bool:
        """Request cancellation of an active order."""
        ...

    def set_event_callback(self, callback: ExecutionCallback) -> None:
        """Register the normalized lifecycle event sink."""
        ...
