from dataclasses import dataclass

import pytest

from proscalper.backtest.matching import FillResult, MatchStatus
from proscalper.core.types import EntryType, OrderSide
from proscalper.execution.executor import ExecutionEvent
from proscalper.execution.paper_adapter import PaperOrderExecutor


@dataclass
class FakeOrder:
    order_id: str


class FakePaper:
    def __init__(self) -> None:
        self.callback = None
        self.calls = []

    def on_fill(self, callback):
        self.callback = callback

    def submit_market_order(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOrder("paper-1")

    def cancel_order(self, order_id):
        return order_id == "paper-1"

    def emit_fill(self):
        event = type(
            "Event",
            (),
            {
                "order": FakeOrder("paper-1"),
                "fill_result": FillResult(
                    status=MatchStatus.FILLED,
                    requested_qty=2.0,
                    filled_qty=2.0,
                    avg_fill_price=101.5,
                    fees=0.0812,
                ),
                "ts_ns": 123456,
            },
        )()
        self.callback(event)


def test_adapter_translates_submit_and_normalizes_fill() -> None:
    paper = FakePaper()
    adapter = PaperOrderExecutor(paper)
    events: list[ExecutionEvent] = []
    adapter.set_event_callback(events.append)

    order_id = adapter.submit_market(
        intent_id="intent-1",
        position_id="position-1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=2.0,
        signal_id="signal-1",
    )

    assert order_id == "paper-1"
    assert paper.calls[0]["symbol"] == "BTCUSDT"
    assert paper.calls[0]["leg_type"] == "entry"

    paper.emit_fill()

    assert len(events) == 1
    assert events[0].intent_id == "intent-1"
    assert events[0].position_id == "position-1"
    assert events[0].fill.quantity == 2.0
    assert events[0].fill.price == 101.5
    assert events[0].fill.fee == 0.0812
    assert events[0].closing is False


def test_adapter_marks_close_orders() -> None:
    paper = FakePaper()
    adapter = PaperOrderExecutor(paper)

    adapter.submit_market(
        intent_id="intent-2",
        position_id="position-2",
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        quantity=1.0,
        signal_id="signal-2",
        closing=True,
    )

    assert paper.calls[0]["leg_type"] == "close"


def test_adapter_rejects_non_market_entry_type() -> None:
    paper = FakePaper()
    adapter = PaperOrderExecutor(paper)

    with pytest.raises(ValueError, match="unsupported paper entry type"):
        adapter.submit_market(
            intent_id="intent-3",
            position_id="position-3",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            signal_id="signal-3",
            entry_type=EntryType.LIMIT,
        )
