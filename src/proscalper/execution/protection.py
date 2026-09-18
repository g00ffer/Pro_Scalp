"""Protective-stop lifecycle for filled position quantities."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProtectionLifecycle(str, Enum):
    REQUIRED = "REQUIRED"
    SUBMITTING = "SUBMITTING"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    EMERGENCY_EXIT_PENDING = "EMERGENCY_EXIT_PENDING"
    CLOSED = "CLOSED"


@dataclass
class ProtectionLeg:
    protection_id: str
    position_id: str
    quantity: float
    stop_price: float
    lifecycle: ProtectionLifecycle = ProtectionLifecycle.REQUIRED
    order_id: str | None = None
    emergency_order_id: str | None = None


class ProtectionManager:
    """Track protective-stop legs independently from ordinary exit orders."""

    def __init__(self) -> None:
        self._legs: dict[str, ProtectionLeg] = {}
        self._order_to_leg: dict[str, str] = {}

    def create(self, position_id: str, quantity: float, stop_price: float, protection_id: str) -> ProtectionLeg:
        if quantity <= 0:
            raise ValueError("protection quantity must be positive")
        if stop_price <= 0:
            raise ValueError("stop_price must be positive")
        if protection_id in self._legs:
            raise ValueError(f"protection already exists: {protection_id}")
        leg = ProtectionLeg(protection_id, position_id, quantity, stop_price)
        self._legs[protection_id] = leg
        return leg

    def mark_submitting(self, protection_id: str) -> None:
        self._require(protection_id).lifecycle = ProtectionLifecycle.SUBMITTING

    def bind_order(self, protection_id: str, order_id: str) -> None:
        leg = self._require(protection_id)
        leg.order_id = order_id
        leg.lifecycle = ProtectionLifecycle.ACTIVE
        self._order_to_leg[order_id] = protection_id

    def mark_rejected(self, protection_id: str) -> None:
        self._require(protection_id).lifecycle = ProtectionLifecycle.REJECTED

    def mark_emergency_submitting(self, protection_id: str) -> None:
        self._require(protection_id).lifecycle = ProtectionLifecycle.EMERGENCY_EXIT_PENDING

    def bind_emergency_order(self, protection_id: str, order_id: str) -> None:
        leg = self._require(protection_id)
        leg.emergency_order_id = order_id
        leg.lifecycle = ProtectionLifecycle.EMERGENCY_EXIT_PENDING
        self._order_to_leg[order_id] = protection_id

    def mark_closed(self, protection_id: str) -> None:
        self._require(protection_id).lifecycle = ProtectionLifecycle.CLOSED

    def leg_for_order(self, order_id: str) -> ProtectionLeg | None:
        protection_id = self._order_to_leg.get(order_id)
        return self._legs.get(protection_id) if protection_id else None

    def state(self, protection_id: str) -> ProtectionLifecycle:
        return self._require(protection_id).lifecycle

    def get(self, protection_id: str) -> ProtectionLeg:
        return self._require(protection_id)

    def active_for_position(self, position_id: str) -> tuple[ProtectionLeg, ...]:
        return tuple(
            leg for leg in self._legs.values()
            if leg.position_id == position_id and leg.lifecycle == ProtectionLifecycle.ACTIVE
        )

    def _require(self, protection_id: str) -> ProtectionLeg:
        try:
            return self._legs[protection_id]
        except KeyError as exc:
            raise KeyError(f"unknown protection_id: {protection_id}") from exc
