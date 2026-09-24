from __future__ import annotations

import pytest

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.engine import ExecutionEngine
from proscalper.execution.executor import ExecutionEvent
from proscalper.execution.models import OrderIntent
from proscalper.execution.order_state import Fill, OrderLifecycle
from proscalper.execution.position_manager import PositionLifecycle, PositionManager


class FakeExecutor:
    def __init__(self) -> None:
        self.callback = None
        self.next_order_id = 0
        self.fail = False
        self.cancelled: list[str] = []
        self.submitted: list[dict] = []

    def _new_order_id(self) -> str:
        self.next_order_id += 1
        return f"order-{self.next_order_id}"

    def submit_market(self, **kwargs) -> str:
        if self.fail:
            raise RuntimeError("submit failed")
        order_id = self._new_order_id()
        self.submitted.append({"kind": "market", "order_id": order_id, **kwargs})
        return order_id

    def submit_stop(self, **kwargs) -> str:
        order_id = self._new_order_id()
        self.submitted.append({"kind": "stop", "order_id": order_id, **kwargs})
        return order_id

    def cancel(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True

    def set_event_callback(self, callback) -> None:
        self.callback = callback

    def emit_fill(
        self,
        *,
        order_id: str,
        intent_id: str,
        position_id: str,
        price: float,
        quantity: float,
        lifecycle: OrderLifecycle,
        closing: bool = False,
    ) -> None:
        assert self.callback is not None
        self.callback(
            ExecutionEvent(
                order_id=order_id,
                intent_id=intent_id,
                position_id=position_id,
                lifecycle=lifecycle,
                fill=Fill(
                    fill_id=f"fill-{price}-{quantity}",
                    order_id=order_id,
                    price=price,
                    quantity=quantity,
                    timestamp_ns=1,
                ),
                closing=closing,
            )
        )


def make_intent(*, position_id: str = "position-1", quantity: float = 2.0) -> OrderIntent:
    return OrderIntent(
        intent_id="intent-1",
        signal_id="signal-1",
        position_id=position_id,
        symbol=Symbol("BTCUSDT"),
        side=OrderSide.BUY,
        quantity=quantity,
        entry_type=EntryType.MARKET,
        stop_price=99.0,
    )


def test_submit_registers_order_without_opening_position() -> None:
    executor = FakeExecutor()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions)

    result = engine.submit(make_intent())

    assert result.order_id == "order-1"
    assert positions.state("position-1") == PositionLifecycle.ENTRY_PENDING
    assert engine.order_state("order-1").lifecycle == OrderLifecycle.ACKNOWLEDGED
    assert engine.position_snapshot("position-1").quantity == 0.0


def test_partial_and_full_fills_update_order_and_position() -> None:
    executor = FakeExecutor()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions)
    engine.submit(make_intent())

    executor.emit_fill(
        order_id="order-1", intent_id="intent-1", position_id="position-1",
        price=100.0, quantity=1.0, lifecycle=OrderLifecycle.PARTIALLY_FILLED,
    )
    assert engine.order_state("order-1").lifecycle == OrderLifecycle.PARTIALLY_FILLED
    assert positions.state("position-1") == PositionLifecycle.PARTIALLY_FILLED

    executor.emit_fill(
        order_id="order-1", intent_id="intent-1", position_id="position-1",
        price=102.0, quantity=1.0, lifecycle=OrderLifecycle.FILLED,
    )
    state = engine.order_state("order-1")
    assert state.lifecycle == OrderLifecycle.FILLED
    assert state.filled_qty == pytest.approx(2.0)
    assert state.avg_fill_price == pytest.approx(101.0)
    assert positions.state("position-1") == PositionLifecycle.OPEN
    assert engine.position_snapshot("position-1").entry_price == pytest.approx(101.0)
    assert len([x for x in executor.submitted if x["kind"] == "stop"]) == 2


def test_submission_failure_does_not_leave_pending_position() -> None:
    executor = FakeExecutor()
    executor.fail = True
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions)

    with pytest.raises(RuntimeError, match="submit failed"):
        engine.submit(make_intent())

    with pytest.raises(KeyError):
        positions.snapshot("position-1")


def test_cancel_marks_order_pending() -> None:
    executor = FakeExecutor()
    engine = ExecutionEngine(executor, PositionManager())
    engine.submit(make_intent())

    assert engine.cancel("order-1") is True
    assert engine.order_state("order-1").lifecycle == OrderLifecycle.CANCEL_PENDING
    assert executor.cancelled == ["order-1"]
