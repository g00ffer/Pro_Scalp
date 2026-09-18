"""Venue-neutral execution adapter contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.order_state import Fill


@dataclass(frozen=True)
class ExecutionEvent:
    """Normalized execution event emitted by an executor adapter."""

    order_id: str
    intent_id: str
    position_id: str
    fill: Fill
    closing: bool = False


ExecutionCallback = Callable[[ExecutionEvent], None]


class OrderExecutor(Protocol):
    """Minimal interface shared by paper and exchange executors.

    The execution engine owns intents and lifecycle; adapters only translate
    intents into venue-specific orders and report normalized fills/rejections.
    """

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
        """Submit an order and return the venue order id."""
        ...

    def cancel(self, order_id: str) -> bool:
        """Request cancellation of an active order."""
        ...

    def set_event_callback(self, callback: ExecutionCallback) -> None:
        """Register the normalized fill/rejection event sink."""
        ...
