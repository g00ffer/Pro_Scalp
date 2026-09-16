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


# (label, импорт, краткая проверка создания)
CHECKS: List[Tuple[str, str]] = [
    # --- Core ---
    ("core.types", "from proscalper.core.types import OrderSide, BookSide, OrderType, TimeInForce, OrderStatus, InstrumentInfo, LevelZone"),
    ("core.events", "from proscalper.core.events import TradeEvent, BookTickerEvent, BookLevel, BookSnapshotEvent, BookDeltaBatchEvent, Signal, FillEvent, OrderEvent"),
    ("core.price_math", "from proscalper.core.price_math import PriceConverter, QuantityConverter, SlippageCalculator, safe_ratio, clamp"),
    ("core.instrument_registry", "from proscalper.core.instrument_registry import InstrumentRegistry"),
    ("core.secrets", "from proscalper.core.secrets import get_secret"),

    # --- Exchange ---
    ("exchange.binance_futures.rest", "from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient"),
    ("exchange.binance_futures.ws_market", "from proscalper.exchange.binance_futures.ws_market import BinanceFuturesMarketDataWS, BinanceFuturesWSConfig"),

    # --- Market data ---
    ("market_data.orderbook_fast", "from proscalper.market_data.orderbook_fast import FastOrderBook"),
    ("market_data.bookticker_guard", "from proscalper.market_data.bookticker_guard import BookTickerGuard, FastGuardConfig"),
    ("market_data.tape", "from proscalper.market_data.tape import TapeManager"),
    ("market_data.bar_aggregator", "from proscalper.market_data.bar_aggregator import MultiTimeframeAggregator, Bar"),

    # --- Features ---
    ("features.levels", "from proscalper.features.levels import LevelManager, LevelDetectorConfig, Level, LevelSide, LevelState"),
    ("features.level_tracker", "from proscalper.features.level_tracker import LevelTracker, LevelTrackerManager"),
    ("features.book_analyzer", "from proscalper.features.book_analyzer import BookAnalyzer, BookAnalyzerConfig"),
    ("features.wall_registry", "from proscalper.features.wall_registry import WallRegistry, WallRegistryConfig"),
    ("features.density", "from proscalper.features.density import DensityDetector, DensityConfig"),
    ("features.imbalance", "from proscalper.features.imbalance import ImbalanceCalculator"),
    ("features.compression", "from proscalper.features.compression import CompressionDetector, CompressionConfig"),
    ("features.impulse_score", "from proscalper.features.impulse_score import ImpulseScoreCalculator, ImpulseScoreConfig"),
    ("features.spoof_detector", "from proscalper.features.spoof_detector import SpoofDetector, SpoofConfig"),
    ("features.iceberg", "from proscalper.features.iceberg import IcebergDetector, IcebergConfig"),
    ("features.cross_market", "from proscalper.features.cross_market import CrossMarketAnalyzer, CrossMarketConfig"),
    ("features.symbol_metrics", "from proscalper.features.symbol_metrics import SymbolMetrics, SymbolMetricsConfig"),

    # --- Signals ---
    ("signals.base", "from proscalper.signals.base import BaseSignalEngine, MarketContext, EngineResult"),
    ("signals.breakout", "from proscalper.signals.breakout import BreakoutDetector, BreakoutConfig"),
    ("signals.retest", "from proscalper.signals.retest import RetestDetector, RetestConfig"),
    ("signals.false_breakout", "from proscalper.signals.false_breakout import FalseBreakoutEngine"),
    ("signals.filters", "from proscalper.signals.filters import SignalFilterManager, FilterConfig"),
    ("signals.signal_generator", "from proscalper.signals.signal_generator import SignalGenerator, SignalConfig"),

    # --- Risk ---
    ("risk.limits", "from proscalper.risk.limits import RiskLimitsManager, LimitConfig"),
    ("risk.sizing", "from proscalper.risk.sizing import PositionSizer, SizingConfig"),
    ("risk.stops", "from proscalper.risk.stops import StopsManager, StopsConfig"),
    ("risk.position_manager", "from proscalper.risk.position_manager import PositionManager"),
    ("risk.risk_manager", "from proscalper.risk.risk_manager import RiskManager"),

    # --- Execution ---
    ("execution.order_builder", "from proscalper.execution.order_builder import OrderBuilder, OrderBuilderConfig"),
    ("execution.bracket", "from proscalper.execution.bracket import BaseBracketExecutor, BracketConfig"),
    ("execution.protection_watchdog", "from proscalper.execution.protection_watchdog import ProtectionWatchdog"),
    ("execution.order_manager", "from proscalper.execution.order_manager import OrderManager"),

    # --- Storage ---
    ("storage.delta_writer", "from proscalper.storage.delta_writer import MarketDataRecorder, AsyncEventWriter"),
    ("storage.keyframe_writer", "from proscalper.storage.keyframe_writer import KeyframeWriter, KeyframeConfig"),
    ("storage.compactor", "from proscalper.storage.compactor import Compactor, CompactorConfig"),
    ("storage.replay_builder", "from proscalper.storage.replay_builder import ReplayBuilder, ReplayConfig"),

    # --- Journal ---
    ("journal.market_state_snapshot", "from proscalper.journal.market_state_snapshot import MarketStateSnapshot, MarketStateSnapshotBuilder"),
    ("journal.decision_journal", "from proscalper.journal.decision_journal import DecisionJournal, DecisionJournalConfig"),
    ("journal.incident_journal", "from proscalper.journal.incident_journal import IncidentJournal, IncidentJournalConfig"),

    # --- Backtest ---
    ("backtest.costs", "from proscalper.backtest.costs import FeeModel, SlippageModel, CostCalculator"),
    ("backtest.latency", "from proscalper.backtest.latency import ProfileLatency, Region, StochasticLatency"),
    ("backtest.matching", "from proscalper.backtest.matching import MatchingEngine, MatchStatus, FillResult"),
    ("backtest.engine", "from proscalper.backtest.engine import BacktestEngine, BacktestEngineConfig"),
    ("backtest.report", "from proscalper.backtest.report import BacktestReport, ReportConfig"),
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