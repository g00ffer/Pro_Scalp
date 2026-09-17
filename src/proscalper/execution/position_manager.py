"""Single owner of local position state.

The manager is deliberately venue-neutral: it consumes fills and exchange
snapshots rather than assuming that order submission means a position exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from proscalper.core.types import OrderSide, PositionSide, Symbol
from proscalper.execution.models import OrderIntent, PositionSnapshot
from proscalper.execution.order_state import Fill


class PositionLifecycle(str, Enum):
    FLAT = "FLAT"
    ENTRY_PENDING = "ENTRY_PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    RECONCILING = "RECONCILING"
    ORPHAN = "ORPHAN"
    ERROR = "ERROR"


@dataclass
class _Position:
    position_id: str
    symbol: Symbol
    side: PositionSide
    lifecycle: PositionLifecycle = PositionLifecycle.ENTRY_PENDING
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


class PositionManager:
    """Owns position state and derives it from execution events."""

    def __init__(self) -> None:
        self._positions: dict[str, _Position] = {}
        self._by_symbol: dict[Symbol, str] = {}

    def create_pending(self, intent: OrderIntent) -> str:
        """Register an entry without pretending it has filled."""
        if intent.quantity <= 0:
            raise ValueError("position quantity must be positive")
        if intent.symbol in self._by_symbol:
            raise ValueError(f"position already exists for {intent.symbol}")

        side = PositionSide.LONG if intent.side == OrderSide.BUY else PositionSide.SHORT
        position = _Position(
            position_id=intent.position_id,
            symbol=intent.symbol,
            side=side,
            stop_price=intent.stop_price,
        )
        self._positions[position.position_id] = position
        self._by_symbol[position.symbol] = position.position_id
        return position.position_id

    def on_fill(self, position_id: str, fill: Fill, *, closing: bool = False) -> None:
        """Apply an execution to a position.

        Entry fills transition PENDING -> PARTIALLY_FILLED/OPEN. Closing fills
        reduce quantity and eventually transition the position to CLOSED.
        """
        position = self._require(position_id)
        if closing:
            if fill.quantity > position.quantity + 1e-12:
                raise ValueError("closing fill exceeds open position quantity")
            position.quantity -= fill.quantity
            if position.quantity <= 1e-12:
                position.quantity = 0.0
                position.lifecycle = PositionLifecycle.CLOSED
                self._by_symbol.pop(position.symbol, None)
            else:
                position.lifecycle = PositionLifecycle.EXIT_PENDING
            return

        previous_qty = position.quantity
        position.quantity += fill.quantity
        position.entry_price = (
            ((position.entry_price * previous_qty) + fill.price * fill.quantity)
            / position.quantity
        )
        position.lifecycle = (
            PositionLifecycle.OPEN
            if position.quantity >= 1e-12
            and position.quantity >= self._requested_quantity(position.position_id) - 1e-12
            else PositionLifecycle.PARTIALLY_FILLED
        )

    def mark_exit_pending(self, position_id: str) -> None:
        position = self._require(position_id)
        if position.quantity <= 0:
            raise ValueError("cannot exit a flat position")
        position.lifecycle = PositionLifecycle.EXIT_PENDING

    def snapshot(self, position_id: str) -> PositionSnapshot:
        position = self._require(position_id)
        return PositionSnapshot(
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            stop_price=position.stop_price,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=position.unrealized_pnl,
        )

    def state(self, position_id: str) -> PositionLifecycle:
        return self._require(position_id).lifecycle

    def _requested_quantity(self, position_id: str) -> float:
        # Until the execution engine owns intent registration, the first fill
        # is sufficient to establish an OPEN position. Partial-fill handling
        # remains explicit through a PARTIALLY_FILLED state when requested size
        # is supplied via set_requested_quantity().
        return getattr(self._require(position_id), "requested_quantity", 0.0) or self._require(position_id).quantity

    def set_requested_quantity(self, position_id: str, quantity: float) -> None:
        if quantity <= 0:
            raise ValueError("requested quantity must be positive")
        position = self._require(position_id)
        position.requested_quantity = quantity
        if position.quantity >= quantity - 1e-12:
            position.lifecycle = PositionLifecycle.OPEN

    def _require(self, position_id: str) -> _Position:
        try:
            return self._positions[position_id]
        except KeyError as exc:
            raise KeyError(f"unknown position_id: {position_id}") from exc
