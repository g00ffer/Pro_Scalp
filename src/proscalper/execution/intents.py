"""Helpers for turning approved risk decisions into order intents."""
from __future__ import annotations

import uuid

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.models import OrderIntent, RiskDecision, Signal


def build_entry_intent(
    signal: Signal,
    decision: RiskDecision,
    *,
    position_id: str | None = None,
    entry_type: EntryType = EntryType.MARKET,
) -> OrderIntent:
    """Create an entry intent from an accepted risk decision.

    Rejected decisions are never executable and therefore raise immediately.
    """
    if not decision.accepted:
        raise ValueError("cannot create OrderIntent from rejected RiskDecision")
    if decision.signal_id != signal.signal_id:
        raise ValueError("signal_id mismatch between Signal and RiskDecision")
    if decision.quantity <= 0:
        raise ValueError("accepted RiskDecision must have positive quantity")

    return OrderIntent(
        intent_id=uuid.uuid4().hex,
        signal_id=signal.signal_id,
        position_id=position_id or uuid.uuid4().hex,
        symbol=Symbol(signal.symbol),
        side=signal.side,
        quantity=decision.quantity,
        entry_type=entry_type,
        entry_price=decision.entry_price,
        stop_price=decision.stop_price,
        reduce_only=False,
    )
