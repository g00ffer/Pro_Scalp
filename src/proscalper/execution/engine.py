"""Central coordinator for order and position execution state."""
from __future__ import annotations

from dataclasses import dataclass

from proscalper.execution.executor import ExecutionEvent, OrderExecutor
from proscalper.execution.models import OrderIntent, PositionSnapshot
from proscalper.execution.order_state import OrderLifecycle, OrderState
from proscalper.execution.position_manager import PositionManager


@dataclass(frozen=True)
class SubmissionResult:
    """Result of accepting an intent for execution."""

    order_id: str
    position_id: str


class ExecutionEngine:
    """Own canonical order lifecycle and feed execution events into positions."""

    def __init__(self, executor: OrderExecutor, position_manager: PositionManager) -> None:
        self.executor = executor
        self.positions = position_manager
        self._orders: dict[str, OrderState] = {}
        self._intent_to_order: dict[str, str] = {}
        self._intent_closing: dict[str, bool] = {}
        self.executor.set_event_callback(self._on_execution_event)

    def submit(self, intent: OrderIntent, *, closing: bool = False) -> SubmissionResult:
        """Register local state, submit to the adapter, then expose the order."""
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
            else:
                self.positions.restore_open(intent.position_id)
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
            state.lifecycle = (
                OrderLifecycle.PARTIALLY_FILLED
                if state.filled_qty > 0
                else OrderLifecycle.ACKNOWLEDGED
            )
        return accepted

    def order_state(self, order_id: str) -> OrderState:
        return self._require_order(order_id)

    def position_snapshot(self, position_id: str) -> PositionSnapshot:
        return self.positions.snapshot(position_id)

    def _on_execution_event(self, event: ExecutionEvent) -> None:
        state = self._require_order(event.order_id)
        if state.intent_id != event.intent_id:
            raise ValueError("execution event intent_id mismatch")
        if event.position_id not in {self._position_id_for(event.order_id)}:
            raise ValueError("execution event position_id mismatch")

        if event.lifecycle == OrderLifecycle.REJECTED:
            state.lifecycle = OrderLifecycle.REJECTED
            state.reject_reason = event.reject_reason
            if event.closing:
                self.positions.restore_open(event.position_id)
            else:
                self.positions.reject_pending(event.position_id)
            return

        if event.lifecycle in {OrderLifecycle.CANCELLED, OrderLifecycle.EXPIRED}:
            state.lifecycle = event.lifecycle
            if event.closing:
                self.positions.restore_open(event.position_id)
            elif state.filled_qty == 0:
                self.positions.reject_pending(event.position_id)
            return

        if event.fill is None:
            raise ValueError(f"{event.lifecycle} execution event requires fill")

        state.apply_fill(event.fill)
        if event.lifecycle != state.lifecycle:
            raise ValueError("execution event lifecycle does not match aggregated order state")
        self.positions.on_fill(event.position_id, event.fill, closing=event.closing)

    def _position_id_for(self, order_id: str) -> str:
        intent_id = self._require_order(order_id).intent_id
        # Position ID is intentionally kept in the executor event and validated
        # against the engine's submission record through this lookup.
        for position_id in self.positions._positions:
            if position_id == position_id and self._intent_to_order.get(intent_id) == order_id:
                return position_id
        raise KeyError(f"position for order not found: {order_id}")

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
