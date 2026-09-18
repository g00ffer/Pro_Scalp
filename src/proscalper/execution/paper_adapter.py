"""Adapter exposing the existing paper simulator through the execution port."""
from __future__ import annotations

from typing import Optional

from proscalper.core.types import EntryType, OrderSide, Symbol
from proscalper.execution.executor import ExecutionCallback, ExecutionEvent
from proscalper.execution.order_state import Fill
from proscalper.execution.paper import PaperExecutor, PaperFillEvent


class PaperOrderExecutor:
    """Translate canonical execution calls into the existing PaperExecutor.

    The matching/latency/cost model stays in ``execution.paper``. This adapter
    only supplies the venue-neutral port and normalizes fills for the domain
    execution pipeline.
    """

    def __init__(self, paper: PaperExecutor) -> None:
        self.paper = paper
        self._metadata: dict[str, tuple[str, str, bool]] = {}
        self._callback: Optional[ExecutionCallback] = None
        self.paper.on_fill(self._handle_fill)

    def submit_market(
        self,
        *,
        intent_id: str,
        position_id: str,
        symbol: Symbol,
        side: OrderSide,
        quantity: float,
        signal_id: str,
        entry_type: EntryType = EntryType.MARKET,
        stop_price: float | None = None,
        closing: bool = False,
    ) -> str:
        """Submit a canonical market intent to the paper simulator."""
        if entry_type != EntryType.MARKET:
            raise ValueError(f"unsupported paper entry type: {entry_type}")
        if quantity <= 0:
            raise ValueError("order quantity must be positive")
        if stop_price is not None:
            raise ValueError("stop_price is not supported by submit_market")

        order = self.paper.submit_market_order(
            symbol=str(symbol),
            side=side,
            quantity=quantity,
            signal_id=signal_id,
            leg_type="close" if closing else "entry",
            reason=f"intent:{intent_id}",
        )
        self._metadata[order.order_id] = (intent_id, position_id, closing)
        return order.order_id

    def cancel(self, order_id: str) -> bool:
        return self.paper.cancel_order(order_id)

    def set_event_callback(self, callback: ExecutionCallback) -> None:
        self._callback = callback

    def _handle_fill(self, event: PaperFillEvent) -> None:
        metadata = self._metadata.get(event.order.order_id)
        if metadata is None or self._callback is None:
            return

        intent_id, position_id, closing = metadata
        result = event.fill_result
        if result.filled_qty <= 0:
            return

        fill = Fill(
            fill_id=f"{event.order.order_id}:{event.ts_ns}:{result.filled_qty}",
            order_id=event.order.order_id,
            price=result.avg_fill_price,
            quantity=result.filled_qty,
            fee=result.fees,
            timestamp_ns=event.ts_ns,
        )
        self._callback(
            ExecutionEvent(
                order_id=event.order.order_id,
                intent_id=intent_id,
                position_id=position_id,
                fill=fill,
                closing=closing,
            )
        )
