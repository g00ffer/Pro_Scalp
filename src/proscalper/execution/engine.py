"""Central coordinator for order and position execution state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from proscalper.execution.executor import ExecutionEvent, OrderExecutor
from proscalper.execution.models import OrderIntent, PositionSnapshot
from proscalper.execution.order_state import OrderLifecycle, OrderState
from proscalper.execution.position_manager import PositionLifecycle, PositionManager


@dataclass(frozen=True)
class SubmissionResult:
    """Result of accepting an intent for execution."""

    order_id: str
    position_id: str


class ExecutionEngine:
    """Own the canonical order lifecycle and feed fills into positions.

    The engine deliberately does not decide *whether* a trade should exist.
    Strategy/risk layers produce an ``OrderIntent``; the engine coordinates
    submission, lifecycle state, and position state around that intent.
    """

    def __init__(self, executor: OrderExecutor, position_manager: PositionManager) -> None:
        self.executor = executor
        self.positions = position_manager
        self._orders: dict[str, OrderState] = {}
        self._intent_to_order: dict[str, str] = {}
        self._intent_closing: dict[str, bool] = {}
        self.executor.set_event_callback(self._on_execution_event)

    def submit(self, intent: OrderIntent, *, closing: bool = False) -> SubmissionResult:
        """Register an intent, submit it, and return the venue order id.

        Position state is registered before venue submission so an asynchronous
        fill can never create a position that the local manager does not know.
        A submission exception leaves the position in ``ERROR`` rather than
        falsely reporting an open trade.
        """
        self._validate_intent(intent)
        if intent.intent_id in self._intent_to_order:
            raise ValueError(f"intent already submitted: {intent.intent_id}")

        if closing:
            self.positions.mark_exit_pending(intent.position_id)
        else:
            self.positions.create_pending(intent)

        try:
            order_id = self.executor.submit_market(
                intent_id=intent.intent_id,
                position_id=intent.position_id,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                signal_id=intent.signal_id,
                entry_type=intent.entry_type,
                stop_price=intent.stop_price,
                closing=closing,
            )
        except Exception:
            if not closing:
                self.positions.reject_pending(intent.position_id)
            raise

        state = OrderState(
            order_id=order_id,
            intent_id=intent.intent_id,
            lifecycle=OrderLifecycle.ACKNOWLEDGED,
            requested_qty=intent.quantity,
        )
        self._orders[order_id] = state
        self._intent_to_order[intent.intent_id] = order_id
        self._intent_closing[intent.intent_id] = closing
        return SubmissionResult(order_id=order_id, position_id=intent.position_id)

    def cancel(self, order_id: str) -> bool:
        """Request cancellation for an active order."""
        state = self._require_order(order_id)
        if state.lifecycle in {
            OrderLifecycle.FILLED,
            OrderLifecycle.CANCELLED,
            OrderLifecycle.REJECTED,
            OrderLifecycle.EXPIRED,
        }:
            return False
        state.lifecycle = OrderLifecycle.CANCEL_PENDING
        accepted = self.executor.cancel(order_id)
        if not accepted:
            state.lifecycle = OrderLifecycle.ACKNOWLEDGED
        return accepted

    def order_state(self, order_id: str) -> OrderState:
        """Return the mutable canonical state for an order."""
        return self._require_order(order_id)

    def position_snapshot(self, position_id: str) -> PositionSnapshot:
        return self.positions.snapshot(position_id)

    def _on_execution_event(self, event: ExecutionEvent) -> None:
        state = self._orders.get(event.order_id)
        if state is None:
            raise KeyError(f"execution event for unknown order_id: {event.order_id}")
        if state.intent_id != event.intent_id:
            raise ValueError("execution event intent_id mismatch")

        state.apply_fill(event.fill)
        self.positions.on_fill(event.position_id, event.fill, closing=event.closing)

    def _validate_intent(self, intent: OrderIntent) -> None:
        if intent.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if not intent.intent_id:
            raise ValueError("intent_id must not be empty")
        if not intent.position_id:
            raise ValueError("position_id must not be empty")
        if not intent.signal_id:
            raise ValueError("signal_id must not be empty")

    def _require_order(self, order_id: str) -> OrderState:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise KeyError(f"unknown order_id: {order_id}") from exc
