"""Venue-neutral order lifecycle state."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrderLifecycle(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Fill:
    """A single execution/fill event."""

    fill_id: str
    order_id: str
    price: float
    quantity: float
    fee: float = 0.0
    timestamp_ns: int = 0


@dataclass
class OrderState:
    """Mutable aggregate for one executable order."""

    order_id: str
    intent_id: str
    lifecycle: OrderLifecycle = OrderLifecycle.NEW
    requested_qty: float = 0.0
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    reject_reason: Optional[str] = None

    def apply_fill(self, fill: Fill) -> None:
        if fill.quantity <= 0:
            raise ValueError("fill quantity must be positive")
        if self.filled_qty + fill.quantity > self.requested_qty + 1e-12:
            raise ValueError("fill quantity exceeds requested quantity")

        previous = self.filled_qty
        self.fills.append(fill)
        self.filled_qty += fill.quantity
        self.avg_fill_price = (
            ((self.avg_fill_price * previous) + (fill.price * fill.quantity))
            / self.filled_qty
        )
        self.lifecycle = (
            OrderLifecycle.FILLED
            if self.filled_qty >= self.requested_qty - 1e-12
            else OrderLifecycle.PARTIALLY_FILLED
        )
