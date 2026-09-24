"""Single owner of local position state."""
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
    requested_quantity: float
    lifecycle: PositionLifecycle = PositionLifecycle.ENTRY_PENDING
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class VenuePositionSnapshot:
    """Normalized position exposure read from the trading venue."""

    symbol: Symbol
    side: PositionSide
    quantity: float
    entry_price: float


@dataclass(frozen=True)
class ReconciliationResult:
    restored: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    orphaned: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class PositionManager:
    """Owns all local position state, including reconciliation state."""

    EPSILON = 1e-12

    def __init__(self) -> None:
        self._positions: dict[str, _Position] = {}
        self._by_symbol: dict[Symbol, str] = {}

    def create_pending(self, intent: OrderIntent) -> str:
        if intent.quantity <= 0:
            raise ValueError("position quantity must be positive")
        if intent.symbol in self._by_symbol:
            raise ValueError(f"position already exists for {intent.symbol}")
        if intent.position_id in self._positions:
            raise ValueError(f"position already exists: {intent.position_id}")

        side = PositionSide.LONG if intent.side == OrderSide.BUY else PositionSide.SHORT
        position = _Position(
            position_id=intent.position_id,
            symbol=intent.symbol,
            side=side,
            requested_quantity=intent.quantity,
            stop_price=intent.stop_price,
        )
        self._positions[position.position_id] = position
        self._by_symbol[position.symbol] = position.position_id
        return position.position_id

    def on_fill(self, position_id: str, fill: Fill, *, closing: bool = False) -> None:
        position = self._require(position_id)
        if fill.quantity <= 0:
            raise ValueError("fill quantity must be positive")

        if closing:
            if fill.quantity > position.quantity + self.EPSILON:
                raise ValueError("closing fill exceeds open position quantity")
            position.quantity -= fill.quantity
            if position.quantity <= self.EPSILON:
                position.quantity = 0.0
                position.lifecycle = PositionLifecycle.CLOSED
                self._by_symbol.pop(position.symbol, None)
            else:
                position.lifecycle = PositionLifecycle.EXIT_PENDING
            return

        if position.lifecycle == PositionLifecycle.EXIT_PENDING:
            raise ValueError("cannot add entry fill while exit is pending")

        previous_qty = position.quantity
        position.quantity += fill.quantity
        position.entry_price = (
            ((position.entry_price * previous_qty) + fill.price * fill.quantity)
            / position.quantity
        )
        if position.quantity > position.requested_quantity + self.EPSILON:
            raise ValueError("entry fill exceeds requested position quantity")
        position.lifecycle = (
            PositionLifecycle.OPEN
            if position.quantity >= position.requested_quantity - self.EPSILON
            else PositionLifecycle.PARTIALLY_FILLED
        )

    def mark_exit_pending(self, position_id: str) -> None:
        position = self._require(position_id)
        if position.quantity <= 0:
            raise ValueError("cannot exit a flat position")
        position.lifecycle = PositionLifecycle.EXIT_PENDING

    def restore_open(self, position_id: str) -> None:
        position = self._require(position_id)
        if position.quantity <= 0:
            raise ValueError("cannot restore a flat position")
        position.lifecycle = PositionLifecycle.OPEN

    def mark_orphan(self, position_id: str) -> None:
        """Mark a live quantity as unsafe and requiring reconciliation."""
        position = self._require(position_id)
        if position.quantity <= 0:
            raise ValueError("cannot orphan a flat position")
        position.lifecycle = PositionLifecycle.ORPHAN

    def reject_pending(self, position_id: str) -> None:
        position = self._require(position_id)
        if position.quantity > self.EPSILON:
            raise ValueError("cannot reject a position that has already filled")
        position.lifecycle = PositionLifecycle.ERROR
        self._positions.pop(position_id, None)
        self._by_symbol.pop(position.symbol, None)

    def snapshot(self, position_id: str) -> PositionSnapshot:
        return self._snapshot(self._require(position_id))

    def snapshots(self) -> tuple[PositionSnapshot, ...]:
        """Return every locally known position, including pending/orphan states."""
        return tuple(self._snapshot(position) for position in self._positions.values())

    def position_id_for_symbol(self, symbol: Symbol) -> str | None:
        return self._by_symbol.get(symbol)

    def begin_reconciliation(self, position_id: str) -> None:
        position = self._require(position_id)
        position.lifecycle = PositionLifecycle.RECONCILING

    def reconcile_open(
        self,
        *,
        position_id: str,
        symbol: Symbol,
        side: PositionSide,
        quantity: float,
        entry_price: float,
    ) -> None:
        """Replace local exposure with the exchange snapshot after reconciliation."""
        if quantity <= 0:
            raise ValueError("reconciled quantity must be positive")
        if entry_price <= 0:
            raise ValueError("reconciled entry_price must be positive")

        existing_id = self._by_symbol.get(symbol)
        if existing_id is not None and existing_id != position_id:
            raise ValueError(f"another local position already exists for {symbol}")

        position = self._positions.get(position_id)
        if position is None:
            position = _Position(
                position_id=position_id,
                symbol=symbol,
                side=side,
                requested_quantity=quantity,
                quantity=quantity,
                entry_price=entry_price,
                lifecycle=PositionLifecycle.RECONCILING,
            )
            self._positions[position_id] = position
        elif position.symbol != symbol:
            raise ValueError("position symbol mismatch during reconciliation")
        elif position.side != side:
            raise ValueError("position side mismatch during reconciliation")

        position.requested_quantity = quantity
        position.quantity = quantity
        position.entry_price = entry_price
        position.lifecycle = PositionLifecycle.OPEN
        self._by_symbol[symbol] = position_id

    def mark_reconciliation_error(self, position_id: str) -> None:
        position = self._require(position_id)
        position.lifecycle = PositionLifecycle.ERROR

    def reconcile(
        self,
        venue_positions: tuple[VenuePositionSnapshot, ...] | list[VenuePositionSnapshot],
    ) -> ReconciliationResult:
        """Reconcile local exposure against the exchange snapshot.

        Exchange quantity/side/entry price are authoritative at this boundary.
        A local position absent from the venue is marked ORPHAN rather than
        silently deleted, so the discrepancy remains visible.
        """
        venue_by_symbol = {
            item.symbol: item
            for item in venue_positions
            if item.quantity > self.EPSILON
        }
        restored: list[str] = []
        unchanged: list[str] = []
        orphaned: list[str] = []
        errors: list[str] = []

        for local in self.snapshots():
            venue = venue_by_symbol.pop(local.symbol, None)
            if venue is None:
                if local.quantity > self.EPSILON:
                    self.mark_orphan(local.position_id)
                    orphaned.append(local.position_id)
                continue

            same = (
                local.side == venue.side
                and abs(local.quantity - venue.quantity) <= self.EPSILON
                and abs(local.entry_price - venue.entry_price) <= self.EPSILON
            )
            if same:
                unchanged.append(local.position_id)
                continue

            try:
                self.begin_reconciliation(local.position_id)
                self.reconcile_open(
                    position_id=local.position_id,
                    symbol=venue.symbol,
                    side=venue.side,
                    quantity=venue.quantity,
                    entry_price=venue.entry_price,
                )
                restored.append(local.position_id)
            except ValueError:
                self.mark_reconciliation_error(local.position_id)
                errors.append(local.position_id)

        for venue in venue_by_symbol.values():
            position_id = f"reconciled-{venue.symbol}"
            try:
                self.reconcile_open(
                    position_id=position_id,
                    symbol=venue.symbol,
                    side=venue.side,
                    quantity=venue.quantity,
                    entry_price=venue.entry_price,
                )
                restored.append(position_id)
            except ValueError:
                errors.append(position_id)

        return ReconciliationResult(
            restored=tuple(restored),
            unchanged=tuple(unchanged),
            orphaned=tuple(orphaned),
            errors=tuple(errors),
        )

    def assert_safe(self) -> None:
        unsafe = [
            snapshot.position_id
            for snapshot in self.snapshots()
            if self.state(snapshot.position_id)
            in {
                PositionLifecycle.ORPHAN,
                PositionLifecycle.ERROR,
                PositionLifecycle.RECONCILING,
            }
        ]
        if unsafe:
            raise RuntimeError(f"unsafe positions require reconciliation: {', '.join(unsafe)}")

    def state(self, position_id: str) -> PositionLifecycle:
        return self._require(position_id).lifecycle

    def set_requested_quantity(self, position_id: str, quantity: float) -> None:
        if quantity <= 0:
            raise ValueError("requested quantity must be positive")
        position = self._require(position_id)
        position.requested_quantity = quantity
        if position.quantity >= quantity - self.EPSILON:
            position.lifecycle = PositionLifecycle.OPEN

    def _snapshot(self, position: _Position) -> PositionSnapshot:
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

    def _require(self, position_id: str) -> _Position:
        try:
            return self._positions[position_id]
        except KeyError as exc:
            raise KeyError(f"unknown position_id: {position_id}") from exc


class ReconciliationService:
    """Application-facing reconciliation boundary around PositionManager."""

    def __init__(self, positions: PositionManager) -> None:
        self.positions = positions

    def reconcile(
        self,
        venue_positions: tuple[VenuePositionSnapshot, ...] | list[VenuePositionSnapshot],
    ) -> ReconciliationResult:
        return self.positions.reconcile(venue_positions)

    def assert_safe(self) -> None:
        self.positions.assert_safe()
