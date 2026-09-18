"""Execution domain and pipeline primitives."""

from proscalper.execution.engine import ExecutionEngine, SubmissionResult
from proscalper.execution.executor import ExecutionEvent, OrderExecutor
from proscalper.execution.intents import build_entry_intent
from proscalper.execution.models import OrderIntent, PositionSnapshot, RiskDecision, Signal
from proscalper.execution.order_state import Fill, OrderLifecycle, OrderState
from proscalper.execution.paper_adapter import PaperOrderExecutor
from proscalper.execution.position_manager import PositionLifecycle, PositionManager
from proscalper.execution.protection import ProtectionLeg, ProtectionLifecycle, ProtectionManager

__all__ = [
    "ExecutionEngine",
    "ExecutionEvent",
    "Fill",
    "OrderExecutor",
    "OrderIntent",
    "OrderLifecycle",
    "OrderState",
    "PaperOrderExecutor",
    "PositionLifecycle",
    "PositionManager",
    "PositionSnapshot",
    "ProtectionLeg",
    "ProtectionLifecycle",
    "ProtectionManager",
    "RiskDecision",
    "Signal",
    "SubmissionResult",
    "build_entry_intent",
]
