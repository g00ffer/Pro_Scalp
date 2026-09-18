from dataclasses import dataclass, field

import pytest

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.engine import ExecutionEngine
from proscalper.execution.executor import ExecutionEvent
from proscalper.execution.models import OrderIntent
from proscalper.execution.order_state import Fill, OrderLifecycle
from proscalper.execution.position_manager import PositionLifecycle, PositionManager


@dataclass
class FakeExecutor:
    next_order_number: int = 0
    callback: object = None
    submitted: list[dict] = field(default_factory=list)
    cancel_result: bool = True

    def set_event_callback(self, callback):
        self.callback = callback

    def _new_order(self, kind, **kwargs):
        self.submitted.append({"kind": kind, **kwargs})
        self.next_order_number += 1
        return f"order-{self.next_order_number}"

    def submit_market(self, **kwargs):
        return self._new_order("market", **kwargs)

    def submit_stop(self, **kwargs):
        return self._new_order("stop", **kwargs)

    def cancel(self, order_id):
        return self.cancel_result

    def emit(self, event):
        self.callback(event)


def make_intent(position_id="position-1", quantity=2.0, side=OrderSide.BUY):
    return OrderIntent(
        intent_id="intent-1",
        signal_id="signal-1",
        position_id=position_id,
        symbol=Symbol("BTCUSDT"),
        side=side,
        quantity=quantity,
        entry_type=EntryType.MARKET,
        stop_price=99.0,
    )


def test_engine_position_opens_only_after_fill():
    executor = FakeExecutor()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions)

    result = engine.submit(make_intent())
    assert positions.state(result.position_id) == PositionLifecycle.ENTRY_PENDING
    assert engine.order_state(result.order_id).lifecycle == OrderLifecycle.ACKNOWLEDGED

    executor.emit(ExecutionEvent(
        order_id=result.order_id,
        intent_id="intent-1",
        position_id=result.position_id,
        lifecycle=OrderLifecycle.PARTIALLY_FILLED,
        fill=Fill("fill-1", result.order_id, 100.0, 0.5),
    ))
    assert positions.snapshot(result.position_id).quantity == 0.5
    assert engine.order_state(result.order_id).lifecycle == OrderLifecycle.PARTIALLY_FILLED

    executor.emit(ExecutionEvent(
        order_id=result.order_id,
        intent_id="intent-1",
        position_id=result.position_id,
        lifecycle=OrderLifecycle.FILLED,
        fill=Fill("fill-2", result.order_id, 102.0, 1.5),
    ))
    assert positions.state(result.position_id) == PositionLifecycle.OPEN
    assert engine.order_state(result.order_id).avg_fill_price == 101.5


def test_engine_rejected_entry_does_not_leave_position():
    executor = FakeExecutor()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions)
    result = engine.submit(make_intent())

    executor.emit(ExecutionEvent(
        order_id=result.order_id,
        intent_id="intent-1",
        position_id=result.position_id,
        lifecycle=OrderLifecycle.REJECTED,
        reject_reason="NO_BOOK",
    ))

    assert engine.order_state(result.order_id).lifecycle == OrderLifecycle.REJECTED
    with pytest.raises(KeyError):
        positions.snapshot(result.position_id)


def test_engine_close_rejection_restores_open_position():
    executor = FakeExecutor()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions)
    entry = engine.submit(make_intent())
    executor.emit(ExecutionEvent(
        order_id=entry.order_id,
        intent_id="intent-1",
        position_id=entry.position_id,
        lifecycle=OrderLifecycle.FILLED,
        fill=Fill("fill-1", entry.order_id, 100.0, 2.0),
    ))

    close_intent = OrderIntent(
        intent_id="intent-close",
        signal_id="signal-close",
        position_id=entry.position_id,
        symbol=Symbol("BTCUSDT"),
        side=OrderSide.SELL,
        quantity=2.0,
    )
    close = engine.submit(close_intent, closing=True)
    assert positions.state(entry.position_id) == PositionLifecycle.EXIT_PENDING

    executor.emit(ExecutionEvent(
        order_id=close.order_id,
        intent_id="intent-close",
        position_id=entry.position_id,
        lifecycle=OrderLifecycle.REJECTED,
        closing=True,
        reject_reason="EXCHANGE_REJECT",
    ))
    assert positions.state(entry.position_id) == PositionLifecycle.OPEN


def test_engine_rejects_mismatched_execution_identity():
    executor = FakeExecutor()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions)
    result = engine.submit(make_intent())

    with pytest.raises(ValueError, match="position_id mismatch"):
        executor.emit(ExecutionEvent(
            order_id=result.order_id,
            intent_id="intent-1",
            position_id="other-position",
            lifecycle=OrderLifecycle.FILLED,
            fill=Fill("fill-1", result.order_id, 100.0, 2.0),
        ))
