"""Canonical execution-pipeline domain models.

These models intentionally separate strategy intent from risk-approved sizing.
They contain no exchange-specific request/response details.
"""
from __future__ import annotations

from typing import Any, Optional

import msgspec

from proscalper.core.types import EntryType, OrderSide, PositionSide, Symbol


class Signal(msgspec.Struct, frozen=True):
    """A strategy decision to consider entering a position.

    A signal does not contain exchange quantity or client order identifiers.
    """

    signal_id: str
    symbol: Symbol
    side: OrderSide
    level_id: str
    entry_reference: float
    stop_reference: float
    confidence: float
    created_ts_ns: int
    reason: str = ""
    feature_snapshot: Optional[Any] = None


class RiskDecision(msgspec.Struct, frozen=True):
    """Result of applying account and trade risk policy to a signal."""

    accepted: bool
    signal_id: str
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    risk_amount: float = 0.0
    notional: float = 0.0
    rejection_reason: Optional[str] = None
    risk_snapshot: Optional[Any] = None


class OrderIntent(msgspec.Struct, frozen=True):
    """Venue-neutral executable instruction produced after risk approval."""

    intent_id: str
    signal_id: str
    position_id: str
    symbol: Symbol
    side: OrderSide
    quantity: float
    entry_type: EntryType = EntryType.MARKET
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    reduce_only: bool = False
    client_order_ids: tuple[str, ...] = ()


class PositionSnapshot(msgspec.Struct, frozen=True):
    """Immutable position state exposed to risk, journals and reconciliation."""

    position_id: str
    symbol: Symbol
    side: PositionSide
    quantity: float
    entry_price: float
    stop_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
