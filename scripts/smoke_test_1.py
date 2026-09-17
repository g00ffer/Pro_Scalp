"""Smoke-тест всех модулей проекта."""
from __future__ import annotations
import sys
from typing import List, Tuple

CHECKS: List[Tuple[str, str]] = [
    ("core.types", "from proscalper.core.types import OrderSide, BookSide"),
    ("core.events", "from proscalper.core.events import TradeEvent, BookTickerEvent"),
    ("core.config", "from proscalper.core.config import load_app_config"),
    ("core.logging", "from proscalper.core.logging import get_logger"),
    ("core.clock", "from proscalper.core.clock import SystemClock, VirtualClock"),
    ("exchange.rest", "from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient"),
    ("exchange.auth", "from proscalper.exchange.binance_futures.auth import BinanceAuth"),
    ("exchange.order", "from proscalper.exchange.binance_futures.order import BinanceOrderClient"),
    ("exchange.ws_market", "from proscalper.exchange.binance_futures.ws_market import BinanceFuturesMarketDataWS"),
    ("exchange.ws_user", "from proscalper.exchange.binance_futures.ws_user import BinanceFuturesUserStreamWS"),
    ("market_data.orderbook_fast", "from proscalper.market_data.orderbook_fast import FastOrderBook"),
    ("market_data.tape", "from proscalper.market_data.tape import TapeManager"),
    ("market_data.bar_aggregator", "from proscalper.market_data.bar_aggregator import MultiTimeframeAggregator"),
    ("features.levels", "from proscalper.features.levels import LevelManager"),
    ("features.book_analyzer", "from proscalper.features.book_analyzer import BookAnalyzer"),
    ("features.spoof_detector", "from proscalper.features.spoof_detector import SpoofDetector"),
    ("features.iceberg", "from proscalper.features.iceberg import IcebergDetector"),
    ("features.symbol_metrics", "from proscalper.features.symbol_metrics import SymbolMetrics"),
    ("signals.breakout", "from proscalper.signals.breakout import BreakoutDetector"),
    ("signals.signal_generator", "from proscalper.signals.signal_generator import SignalGenerator"),
    ("risk.limits", "from proscalper.risk.limits import RiskLimitsManager"),
    ("risk.sizing", "from proscalper.risk.sizing import PositionSizer"),
    ("risk.stops", "from proscalper.risk.stops import StopsManager"),
    ("risk.position_manager", "from proscalper.risk.position_manager import PositionManager"),
    ("execution.order_builder", "from proscalper.execution.order_builder import OrderBuilder"),
    ("execution.bracket", "from proscalper.execution.bracket import BaseBracketExecutor"),
    ("execution.protection_watchdog", "from proscalper.execution.protection_watchdog import ProtectionWatchdog"),
    ("execution.order_manager", "from proscalper.execution.order_manager import OrderManager"),
    ("execution.paper", "from proscalper.execution.paper import PaperExecutor"),
    ("storage.delta_writer", "from proscalper.storage.delta_writer import MarketDataRecorder"),
    ("storage.keyframe_writer", "from proscalper.storage.keyframe_writer import KeyframeWriter"),
    ("storage.compactor", "from proscalper.storage.compactor import Compactor"),
    ("storage.replay_builder", "from proscalper.storage.replay_builder import ReplayBuilder"),
    ("journal.market_state_snapshot", "from proscalper.journal.market_state_snapshot import MarketStateSnapshot"),
    ("journal.decision_journal", "from proscalper.journal.decision_journal import DecisionJournal"),
    ("journal.incident_journal", "from proscalper.journal.incident_journal import IncidentJournal"),
    ("backtest.costs", "from proscalper.backtest.costs import CostCalculator"),
    ("backtest.latency", "from proscalper.backtest.latency import ProfileLatency"),
    ("backtest.matching", "from proscalper.backtest.matching import MatchingEngine"),
    ("backtest.engine", "from proscalper.backtest.engine import BacktestEngine"),
    ("backtest.report", "from proscalper.backtest.report import BacktestReport"),
]

def main() -> int:
    ok = fail = 0
    for label, stmt in CHECKS:
        try:
            exec(stmt, {})
            print(f"  ✅ {label}")
            ok += 1
        except Exception as exc:
            print(f"  ❌ {label}: {exc}")
            fail += 1
    print(f"\nИтого: {ok} OK, {fail} FAIL из {len(CHECKS)}")
    if fail == 0:
        print("🎉 Все модули импортируются корректно")
    return 1 if fail > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
