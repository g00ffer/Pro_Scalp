"""Central coordinator for order, position and protection state."""
from __future__ import annotations

from dataclasses import dataclass
import uuid

from proscalper.core.types import EntryType, OrderSide
from proscalper.execution.executor import ExecutionEvent, OrderExecutor
from proscalper.execution.models import OrderIntent, PositionSnapshot
from proscalper.execution.order_state import OrderLifecycle, OrderState
from proscalper.execution.position_manager import PositionManager
from proscalper.execution.protection import ProtectionLifecycle, ProtectionManager


@dataclass(frozen=True)
class SubmissionResult:
    """Result of accepting an intent for execution."""

    order_id: str
    position_id: str


class ExecutionEngine:
    """Own canonical order lifecycle and enforce position protection."""

    def __init__(
        self,
        executor: OrderExecutor,
        position_manager: PositionManager,
        protection_manager: ProtectionManager | None = None,
    ) -> None:
        self.executor = executor
        self.positions = position_manager
        self.protection = protection_manager or ProtectionManager()
        self._orders: dict[str, OrderState] = {}
        self._intent_to_order: dict[str, str] = {}
        self._order_to_position: dict[str, str] = {}
        self._intent_closing: dict[str, bool] = {}
        self._order_kind: dict[str, str] = {}
        self._order_to_protection: dict[str, str] = {}
        self._submitting = True
        self._buffered_events: list[ExecutionEvent] = []
        self.executor.set_event_callback(self._on_execution_event)
        self._submitting = False

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
            if closing:
                self.positions.restore_open(intent.position_id)
            else:
                self.positions.reject_pending(intent.position_id)
            raise

        self._register_order(order_id, intent.intent_id, intent.position_id, intent.quantity, closing, "emergency" if closing and intent.reduce_only else "normal")
        self._flush_buffered_events()
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

    def protection_state(self, protection_id: str) -> ProtectionLifecycle:
        return self.protection.state(protection_id)

    def _on_execution_event(self, event: ExecutionEvent) -> None:
        if event.order_id not in self._orders:
            self._buffered_events.append(event)
            return

        state = self._require_order(event.order_id)
        if state.intent_id != event.intent_id:
            raise ValueError("execution event intent_id mismatch")
        if self._order_to_position.get(event.order_id) != event.position_id:
            raise ValueError("execution event position_id mismatch")
        if self._intent_closing.get(event.intent_id) != event.closing:
            raise ValueError("execution event closing flag mismatch")

        kind = self._order_kind.get(event.order_id, "normal")
        protection_id = self._order_to_protection.get(event.order_id)

        if event.lifecycle == OrderLifecycle.REJECTED:
            state.lifecycle = OrderLifecycle.REJECTED
            state.reject_reason = event.reject_reason
            if kind == "protection":
                assert protection_id is not None
                self.protection.mark_rejected(protection_id)
                self._emergency_close(event.position_id, protection_id)
            elif event.closing:
                self.positions.restore_open(event.position_id)
            else:
                self.positions.reject_pending(event.position_id)
            return

        if event.lifecycle in {OrderLifecycle.CANCELLED, OrderLifecycle.EXPIRED}:
            state.lifecycle = event.lifecycle
            if kind == "protection":
                assert protection_id is not None
                self.protection.mark_rejected(protection_id)
                self._emergency_close(event.position_id, protection_id)
            elif event.closing:
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

        if kind == "normal" and not event.closing:
            self._protect_fill(event, event.fill.quantity)
        elif kind == "protection":
            assert protection_id is not None
            if event.lifecycle == OrderLifecycle.FILLED:
                self.protection.mark_closed(protection_id)
            elif event.lifecycle == OrderLifecycle.PARTIALLY_FILLED:
                self.protection.mark_emergency_pending(protection_id, "")
                self._emergency_close(event.position_id, protection_id, already_closed=event.fill.quantity)
        elif kind == "emergency":
            if event.lifecycle == OrderLifecycle.FILLED:
                self.protection.mark_closed(protection_id) if protection_id else None
            elif event.lifecycle == OrderLifecycle.PARTIALLY_FILLED:
                self.positions.mark_orphan(event.position_id)

    def _protect_fill(self, event: ExecutionEvent, quantity: float) -> None:
        snapshot = self.positions.snapshot(event.position_id)
        if snapshot.stop_price is None or quantity <= 0:
            return
        protection_id = f"protect-{uuid.uuid4().hex}"
        leg = self.protection.create(
            event.position_id,
            quantity,
            snapshot.stop_price,
            protection_id,
        )
        leg.lifecycle = ProtectionLifecycle.SUBMITTING
        stop_side = OrderSide.SELL if snapshot.side.value == "LONG" else OrderSide.BUY
        intent_id = f"{protection_id}-intent"
        try:
            order_id = self.executor.submit_stop(
                intent_id=intent_id,
                position_id=event.position_id,
                symbol=snapshot.symbol,
                side=stop_side,
                quantity=quantity,
                signal_id=event.intent_id,
                stop_price=snapshot.stop_price,
            )
        except Exception as exc:
            self.protection.mark_rejected(protection_id)
            self._emergency_close(event.position_id, protection_id)
            return
        self._register_order(order_id, intent_id, event.position_id, quantity, True, "protection", protection_id)
        self.protection.bind_order(protection_id, order_id)
        self._flush_buffered_events()

    def _emergency_close(self, position_id: str, protection_id: str, *, already_closed: float = 0.0) -> None:
        snapshot = self.positions.snapshot(position_id)
        remaining = snapshot.quantity - already_closed
        if remaining <= 1e-12:
            self.protection.mark_closed(protection_id)
            return
        self.protection._require(protection_id).lifecycle = ProtectionLifecycle.EMERGENCY_EXIT_PENDING
        intent_id = f"emergency-{uuid.uuid4().hex}"
        side = OrderSide.SELL if snapshot.side.value == "LONG" else OrderSide.BUY
        intent = OrderIntent(
            intent_id=intent_id,
            signal_id=f"protection-failure:{protection_id}",
            position_id=position_id,
            symbol=snapshot.symbol,
            side=side,
            quantity=remaining,
            entry_type=EntryType.MARKET,
            reduce_only=True,
        )
        try:
            order_id = self.executor.submit_market(
                intent_id=intent.intent_id,
                position_id=intent.position_id,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                signal_id=intent.signal_id,
                entry_type=intent.entry_type,
                closing=True,
            )
        except Exception:
            self.positions.mark_orphan(position_id)
            return
        self._register_order(order_id, intent.intent_id, position_id, remaining, True, "emergency", protection_id)
        self.protection.mark_emergency_pending(protection_id, order_id)
        self._flush_buffered_events()

    def _register_order(
        self,
        order_id: str,
        intent_id: str,
        position_id: str,
        quantity: float,
        closing: bool,
        kind: str,
        protection_id: str | None = None,
    ) -> None:
        if order_id in self._orders:
            raise ValueError(f"executor returned duplicate order_id: {order_id}")
        self._orders[order_id] = OrderState(
            order_id=order_id,
            intent_id=intent_id,
            lifecycle=OrderLifecycle.ACKNOWLEDGED,
            requested_qty=quantity,
        )
        self._intent_to_order[intent_id] = order_id
        self._order_to_position[order_id] = position_id
        self._intent_closing[intent_id] = closing
        self._order_kind[order_id] = kind
        if protection_id is not None:
            self._order_to_protection[order_id] = protection_id

    def _flush_buffered_events(self) -> None:
        if not self._buffered_events:
            return
        pending = self._buffered_events
        self._buffered_events = []
        for event in pending:
            self._on_execution_event(event)

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
