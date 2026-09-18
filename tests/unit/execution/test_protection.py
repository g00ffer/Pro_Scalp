from dataclasses import dataclass, field

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.engine import ExecutionEngine
from proscalper.execution.executor import ExecutionEvent
from proscalper.execution.models import OrderIntent
from proscalper.execution.order_state import Fill, OrderLifecycle
from proscalper.execution.position_manager import PositionLifecycle, PositionManager
from proscalper.execution.protection import ProtectionLifecycle, ProtectionManager


@dataclass
class FakeExecutor:
    next_id: int = 0
    callback: object = None
    submitted: list[dict] = field(default_factory=list)
    fail_stop: bool = False
    fail_emergency: bool = False

    def set_event_callback(self, callback):
        self.callback = callback

    def _new_order_id(self) -> str:
        self.next_id += 1
        return f"order-{self.next_id}"

    def submit_market(self, **kwargs):
        if kwargs.get("closing") and self.fail_emergency:
            raise RuntimeError("MARKET_REJECT")
        order_id = self._new_order_id()
        self.submitted.append({"kind": "market", "order_id": order_id, **kwargs})
        return order_id

    def submit_stop(self, **kwargs):
        if self.fail_stop:
            raise RuntimeError("STOP_REJECT")
        order_id = self._new_order_id()
        self.submitted.append({"kind": "stop", "order_id": order_id, **kwargs})
        return order_id

    def cancel(self, order_id):
        return True

    def emit(self, event):
        self.callback(event)


def entry_intent():
    return OrderIntent(
        intent_id="entry-intent",
        signal_id="signal-1",
        position_id="position-1",
        symbol=Symbol("BTCUSDT"),
        side=OrderSide.BUY,
        quantity=2.0,
        entry_type=EntryType.MARKET,
        stop_price=99.0,
    )


def test_full_entry_creates_protective_stop_for_filled_quantity():
    executor = FakeExecutor()
    protection = ProtectionManager()
    engine = ExecutionEngine(executor, PositionManager(), protection)
    entry = engine.submit(entry_intent())

    executor.emit(ExecutionEvent(
        order_id=entry.order_id,
        intent_id="entry-intent",
        position_id="position-1",
        lifecycle=OrderLifecycle.FILLED,
        fill=Fill("fill-1", entry.order_id, 100.0, 2.0),
    ))

    stop = next(x for x in executor.submitted if x["kind"] == "stop")
    assert stop["quantity"] == 2.0
    assert stop["side"] == OrderSide.SELL
    protection_id = next(iter(protection._legs))
    assert protection.state(protection_id) == ProtectionLifecycle.ACTIVE


def test_stop_rejection_triggers_emergency_close():
    executor = FakeExecutor(fail_stop=True)
    protection = ProtectionManager()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions, protection)
    entry = engine.submit(entry_intent())

    executor.emit(ExecutionEvent(
        order_id=entry.order_id,
        intent_id="entry-intent",
        position_id="position-1",
        lifecycle=OrderLifecycle.FILLED,
        fill=Fill("fill-1", entry.order_id, 100.0, 2.0),
    ))

    emergency = next(x for x in executor.submitted if x["kind"] == "market" and x["closing"])
    assert emergency["quantity"] == 2.0
    protection_id = next(iter(protection._legs))
    assert protection.state(protection_id) == ProtectionLifecycle.EMERGENCY_EXIT_PENDING
    assert positions.state("position-1") == PositionLifecycle.OPEN


def test_emergency_rejection_marks_position_orphan():
    executor = FakeExecutor(fail_stop=True, fail_emergency=True)
    protection = ProtectionManager()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions, protection)
    entry = engine.submit(entry_intent())

    executor.emit(ExecutionEvent(
        order_id=entry.order_id,
        intent_id="entry-intent",
        position_id="position-1",
        lifecycle=OrderLifecycle.FILLED,
        fill=Fill("fill-1", entry.order_id, 100.0, 2.0),
    ))

    assert positions.state("position-1") == PositionLifecycle.ORPHAN


def test_protective_stop_fill_closes_position():
    executor = FakeExecutor()
    protection = ProtectionManager()
    positions = PositionManager()
    engine = ExecutionEngine(executor, positions, protection)
    entry = engine.submit(entry_intent())
    executor.emit(ExecutionEvent(
        order_id=entry.order_id,
        intent_id="entry-intent",
        position_id="position-1",
        lifecycle=OrderLifecycle.FILLED,
        fill=Fill("fill-1", entry.order_id, 100.0, 2.0),
    ))

    stop = next(x for x in executor.submitted if x["kind"] == "stop")
    protection_id = next(iter(protection._legs))
    executor.emit(ExecutionEvent(
        order_id=stop["order_id"],
        intent_id=f"{protection_id}-intent",
        position_id="position-1",
        lifecycle=OrderLifecycle.FILLED,
        fill=Fill("fill-stop", stop["order_id"], 98.0, 2.0),
        closing=True,
    ))

    assert positions.state("position-1") == PositionLifecycle.CLOSED
    assert protection.state(protection_id) == ProtectionLifecycle.CLOSED
