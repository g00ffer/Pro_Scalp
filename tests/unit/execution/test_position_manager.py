from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.models import OrderIntent
from proscalper.execution.order_state import Fill
from proscalper.execution.position_manager import PositionLifecycle, PositionManager


def make_intent() -> OrderIntent:
    return OrderIntent(
        intent_id="intent-1",
        signal_id="signal-1",
        position_id="position-1",
        symbol=Symbol("BTCUSDT"),
        side=OrderSide.BUY,
        quantity=2.0,
        entry_type=EntryType.MARKET,
        stop_price=99.0,
    )


def test_submission_does_not_open_position() -> None:
    manager = PositionManager()
    manager.create_pending(make_intent())
    manager.set_requested_quantity("position-1", 2.0)

    assert manager.state("position-1") == PositionLifecycle.ENTRY_PENDING


def test_partial_and_full_entry_fills_drive_lifecycle() -> None:
    manager = PositionManager()
    manager.create_pending(make_intent())
    manager.set_requested_quantity("position-1", 2.0)

    manager.on_fill(
        "position-1",
        Fill(fill_id="fill-1", order_id="order-1", price=100.0, quantity=0.75),
    )
    assert manager.state("position-1") == PositionLifecycle.PARTIALLY_FILLED
    assert manager.snapshot("position-1").quantity == 0.75

    manager.on_fill(
        "position-1",
        Fill(fill_id="fill-2", order_id="order-1", price=102.0, quantity=1.25),
    )
    assert manager.state("position-1") == PositionLifecycle.OPEN
    assert manager.snapshot("position-1").quantity == 2.0
    assert manager.snapshot("position-1").entry_price == 101.25


def test_closing_fill_transitions_to_closed() -> None:
    manager = PositionManager()
    manager.create_pending(make_intent())
    manager.set_requested_quantity("position-1", 2.0)
    manager.on_fill(
        "position-1",
        Fill(fill_id="fill-1", order_id="order-1", price=100.0, quantity=2.0),
    )

    manager.mark_exit_pending("position-1")
    manager.on_fill(
        "position-1",
        Fill(fill_id="fill-2", order_id="order-2", price=101.0, quantity=2.0),
        closing=True,
    )

    assert manager.state("position-1") == PositionLifecycle.CLOSED
    assert manager.snapshot("position-1").quantity == 0.0
