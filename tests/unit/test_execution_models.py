from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.intents import build_entry_intent
from proscalper.execution.models import RiskDecision, Signal
from proscalper.execution.order_state import Fill, OrderLifecycle, OrderState


def make_signal() -> Signal:
    return Signal(
        signal_id="sig-1",
        symbol=Symbol("BTCUSDT"),
        side=OrderSide.BUY,
        level_id="level-1",
        entry_reference=100.0,
        stop_reference=99.0,
        confidence=0.8,
        created_ts_ns=1,
        reason="breakout",
    )


def test_build_entry_intent_requires_accepted_decision() -> None:
    signal = make_signal()
    decision = RiskDecision(accepted=False, signal_id=signal.signal_id)

    try:
        build_entry_intent(signal, decision)
    except ValueError as exc:
        assert "rejected" in str(exc)
    else:
        raise AssertionError("rejected risk decision must not become an order intent")


def test_build_entry_intent_preserves_risk_approved_quantity() -> None:
    signal = make_signal()
    decision = RiskDecision(
        accepted=True,
        signal_id=signal.signal_id,
        quantity=2.5,
        entry_price=100.2,
        stop_price=99.0,
    )

    intent = build_entry_intent(
        signal,
        decision,
        position_id="pos-1",
        entry_type=EntryType.MARKET,
    )

    assert intent.position_id == "pos-1"
    assert intent.quantity == 2.5
    assert intent.stop_price == 99.0


def test_order_state_aggregates_partial_fills() -> None:
    state = OrderState(order_id="ord-1", intent_id="int-1", requested_qty=3.0)
    state.apply_fill(Fill("fill-1", "ord-1", 100.0, 1.0))
    state.apply_fill(Fill("fill-2", "ord-1", 102.0, 2.0))

    assert state.lifecycle is OrderLifecycle.FILLED
    assert state.filled_qty == 3.0
    assert state.avg_fill_price == (100.0 + 204.0) / 3.0
