"""
Smoke-тест: проверяет, что все модули проекта импортируются
и что ключевые объекты создаются без ошибок.

Запуск:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import traceback
from typing import List, Tuple

# Пары (название, import-строка)
CHECKS: List[Tuple[str, str]] = [
    # --- Core ---
    ("core.types", "from proscalper.core.types import OrderSide, BookSide"),
    ("core.events", "from proscalper.core.events import TradeEvent, BookTickerEvent"),
    ("core.price_math", "from proscalper.core.price_math import PriceConverter, QuantityConverter, SlippageCalculator, safe_ratio, clamp"),
    ("core.instrument_registry", "from proscalper.core.instrument_registry import InstrumentRegistry"),
    ("core.secrets", "from proscalper.core.secrets import get_secret"),
    ("core.config", "from proscalper.core.config import load_app_config, AppConfig, CollectorSection, StrategySection, RiskSection, ExecutionSection, JournalSection, BacktestSection"),
    ("core.logging", "from proscalper.core.logging import get_logger, setup_logging, LogContext, StructuredLogger"),
    ("core.clock", "from proscalper.core.clock import SystemClock, VirtualClock, ExchangeTimeSync, get_clock, set_clock, ms_to_ns, ns_to_ms, format_ts_ns, format_duration_ns"),
    ("core.telemetry", "from proscalper.core.telemetry import Telemetry, telemetry, Counter, Gauge, Histogram"),

    # --- Exchange / Public ---
    ("exchange.binance_futures.rest", "from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient"),
    ("exchange.binance_futures.ws_market", "from proscalper.exchange.binance_futures.ws_market import BinanceFuturesMarketDataWS, BinanceFuturesWSConfig"),
    ("exchange.binance_futures.bookticker_guard", "from proscalper.market_data.bookticker_guard import BookTickerGuard"),

    # --- Exchange / Private ---
    ("exchange.binance_futures.auth", "from proscalper.exchange.binance_futures.auth import BinanceAuth, BinanceCredentials, BinanceAuthConfig, BinanceEndpoint, create_auth, create_auth_from_env"),
    ("exchange.binance_futures.order", "from proscalper.exchange.binance_futures.order import BinanceOrderClient, OrderParams, OrderResponse, OrderType, OrderSide as OrderSide2, TimeInForce, WorkingType, OrderStatus"),
    ("exchange.binance_futures.ws_user", "from proscalper.exchange.binance_futures.ws_user import BinanceFuturesUserStreamWS, UserStreamWSConfig, UserStreamHandler, OrderTradeUpdate, AccountUpdate, AccountConfigUpdate, UserEventType"),
    ("exchange.binance_futures.sync", "from proscalper.exchange.binance_futures.sync import BinanceSyncer, SyncResult, BalanceSnapshot, PositionSnapshot, OpenOrderSnapshot"),
    ("exchange.base", "from proscalper.exchange.base import ExchangeBase, ExchangeName, ExchangeOrderType, ExchangeOrderSide, ExchangeOrderStatus, ExchangeOrderRequest, ExchangeOrderResponse, ExchangePosition, ExchangeBalance"),

    # --- Market Data ---
    ("market_data.orderbook_fast", "from proscalper.market_data.orderbook_fast import FastOrderBook"),
    ("market_data.tape", "from proscalper.market_data.tape import TapeManager, TapeMetrics"),
    ("market_data.bar_aggregator", "from proscalper.market_data.bar_aggregator import MultiTimeframeAggregator, Bar, BarAggregator"),
    ("market_data.bookticker_guard", "from proscalper.market_data.bookticker_guard import BookTickerGuard, FastGuardConfig"),

    # --- Features ---
    ("features.levels", "from proscalper.features.levels import LevelManager, LevelDetector, LevelDetectorConfig, Level, LevelSide, LevelState"),
    ("features.book_analyzer", "from proscalper.features.book_analyzer import BookAnalyzer, BookAnalyzerConfig, BookAnalyzerManager"),
    ("features.wall_registry", "from proscalper.features.wall_registry import WallRegistry, WallRegistryConfig, WallRegistryManager"),
    ("features.density", "from proscalper.features.density import DensityDetector, DensityConfig"),
    ("features.imbalance", "from proscalper.features.imbalance import ImbalanceCalculator"),
    ("features.compression", "from proscalper.features.compression import CompressionDetector, CompressionConfig, CompressionManager"),
    ("features.impulse_score", "from proscalper.features.impulse_score import ImpulseScoreCalculator, ImpulseScoreConfig, ImpulseScoreManager"),
    ("features.level_tracker", "from proscalper.features.level_tracker import LevelTracker, LevelTrackerConfig"),
    ("features.tape_analyzer", "from proscalper.features.tape_analyzer import TapeAnalyzer"),
    ("features.spoof_detector", "from proscalper.features.spoof_detector import SpoofDetector, SpoofConfig, SpoofDetectorManager"),
    ("features.iceberg", "from proscalper.features.iceberg import IcebergDetector, IcebergConfig, IcebergDetectorManager"),
    ("features.cross_market", "from proscalper.features.cross_market import CrossMarketAnalyzer, CrossMarketConfig"),
    ("features.symbol_metrics", "from proscalper.features.symbol_metrics import SymbolMetrics, SymbolMetricsConfig, SymbolMetricsManager"),

    # --- Signals ---
    ("signals.base", "from proscalper.signals.base import BaseSignalEngine, BaseEngineConfig, SetupType, EngineState, MarketContext, EngineResult"),
    ("signals.breakout", "from proscalper.signals.breakout import BreakoutDetector, BreakoutManager, BreakoutConfig, BreakoutAttempt, BreakoutSignal, BreakoutState"),
    ("signals.retest", "from proscalper.signals.retest import RetestDetector, RetestConfig"),
    ("signals.false_breakout", "from proscalper.signals.false_breakout import FalseBreakoutEngine, FalseBreakoutConfig"),
    ("signals.filters", "from proscalper.signals.filters import SignalFilterManager, FilterConfig"),
    ("signals.signal_generator", "from proscalper.signals.signal_generator import SignalGenerator, SignalConfig, SignalGeneratorManager"),
    ("signals.approach", "from proscalper.signals.approach import ApproachDetector, ApproachConfig"),

    # --- Risk ---
    ("risk.limits", "from proscalper.risk.limits import RiskLimitsManager, LimitConfig"),
    ("risk.sizing", "from proscalper.risk.sizing import PositionSizer, SizingConfig"),
    ("risk.stops", "from proscalper.risk.stops import StopsManager, StopsConfig"),
    ("risk.position_manager", "from proscalper.risk.position_manager import PositionManager"),
    ("risk.risk_manager", "from proscalper.risk.risk_manager import RiskManager"),

    # --- Execution ---
    ("execution.order_builder", "from proscalper.execution.order_builder import OrderBuilder"),
    ("execution.bracket", "from proscalper.execution.bracket import BaseBracketExecutor, BracketConfig"),
    ("execution.protection_watchdog", "from proscalper.execution.protection_watchdog import ProtectionWatchdog"),
    ("execution.order_manager", "from proscalper.execution.order_manager import OrderManager"),
    ("execution.order_tracker", "from proscalper.execution.order_tracker import OrderTracker"),
    ("execution.execution_engine", "from proscalper.execution.execution_engine import ExecutionEngine"),
    ("execution.paper", "from proscalper.execution.paper import PaperExecutor, PaperExecutionConfig, PaperOrder, PaperOrderState, PaperFillEvent"),
    ("execution.reconciliation", "from proscalper.execution.reconciliation import Reconciliator, ReconciliationResult, Discrepancy, DiscrepancyType, DiscrepancySeverity, LocalPosition"),

    # --- Storage ---
    ("storage.delta_writer", "from proscalper.storage.delta_writer import MarketDataRecorder"),
    ("storage.keyframe_writer", "from proscalper.storage.keyframe_writer import KeyframeWriter"),
    ("storage.compactor", "from proscalper.storage.compactor import Compactor, CompactorConfig"),
    ("storage.replay_builder", "from proscalper.storage.replay_builder import ReplayBuilder"),

    # --- Journal ---
    ("journal.market_state_snapshot", "from proscalper.journal.market_state_snapshot import MarketStateSnapshot, MarketStateSnapshotBuilder"),
    ("journal.decision_journal", "from proscalper.journal.decision_journal import DecisionJournal, DecisionJournalConfig"),
    ("journal.incident_journal", "from proscalper.journal.incident_journal import IncidentJournal, IncidentJournalConfig"),
    ("journal.reasons", "from proscalper.journal.reasons import SignalReason, RejectReason, RiskReason, ExitReason, OrderRejectReason, ProtectionReason, is_valid_reason, get_all_codes"),
    ("journal.trade_journal", "from proscalper.journal.trade_journal import TradeJournal, TradeJournalConfig, TradeJournalEntry, TradeEventType, TradeDirection"),

    # --- Backtest ---
    ("backtest.costs", "from proscalper.backtest.costs import CostCalculator, FeeConfig"),
    ("backtest.latency", "from proscalper.backtest.latency import LatencyModel, ProfileLatency, Region, LatencyBreakdown"),
    ("backtest.matching", "from proscalper.backtest.matching import MatchingEngine, FillResult, MatchStatus"),
    ("backtest.engine", "from proscalper.backtest.engine import BacktestEngine, BacktestEngineConfig, BacktestContext"),
    ("backtest.report", "from proscalper.backtest.report import BacktestReport, ReportConfig"),

    # --- Apps ---
    ("app.collector", "from proscalper.app.collector import CollectorConfig, CollectorHandler, run_collector"),
    ("app.paper_runner", "from proscalper.app.paper_runner import PaperRunnerHandler, PaperPosition, run_paper_runner"),
    ("app.replay_runner", "from proscalper.app.replay_runner import run_replay, SimpleBreakoutStrategy, find_available_dates"),
    ("app.live_runner", "from proscalper.app.live_runner import LiveRunnerHandler, LivePosition, LiveUserStreamHandler, run_live_runner"),
]


def main() -> int:
    ok_count = 0
    fail_count = 0

    for label, stmt in CHECKS:
        try:
            exec(stmt, {})
            print(f"  ✅ {label}")
            ok_count += 1
        except Exception as exc:
            print(f"  ❌ {label}: {type(exc).__name__}: {exc}")
            fail_count += 1

    print()
    print(f"Итого: {ok_count} OK, {fail_count} FAIL из {len(CHECKS)}")

    if fail_count > 0:
        return 1

    print("🎉 Все модули импортируются корректно")
    return 0


if __name__ == "__main__":
    sys.exit(main())