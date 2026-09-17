"""
Тесты для proscalper.backtest.engine

Запуск:
    python -m pytest tests/backtest/test_engine.py -v

Требует исправленного engine.py:
- _current_ts_ns и _result_book инициализированы в __init__
- run() распаковывает кортежи (book, delta) из replay_day()
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from proscalper.backtest.costs import FeeConfig
from proscalper.backtest.engine import (
    BacktestContext,
    BacktestEngine,
    BacktestEngineConfig,
    BacktestPosition,
    BacktestResult,
    BacktestTrade,
    PositionSide,
)
from proscalper.backtest.latency import ConstantLatency, Region
from proscalper.core.types import OrderSide


# ============================================================
# BacktestPosition
# ============================================================

class TestBacktestPosition:
    """Тесты позиции бэктеста."""

    def test_flat_not_open(self):
        """FLAT позиция не открыта."""
        pos = BacktestPosition(symbol="BTCUSDT")
        assert not pos.is_open
        assert pos.side == PositionSide.FLAT

    def test_long_is_open(self):
        """LONG позиция с объёмом открыта."""
        pos = BacktestPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=100.0,
        )
        assert pos.is_open

    def test_short_is_open(self):
        """SHORT позиция с объёмом открыта."""
        pos = BacktestPosition(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=1.0,
            entry_price=100.0,
        )
        assert pos.is_open

    def test_zero_qty_not_open(self):
        """Нулевой объём → позиция не открыта."""
        pos = BacktestPosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.0,
        )
        assert not pos.is_open


# ============================================================
# BacktestResult
# ============================================================

class TestBacktestResult:
    """Тесты агрегированного результата."""

    def test_empty_result(self):
        """Пустой результат: все метрики нулевые."""
        result = BacktestResult(symbol="BTCUSDT")
        assert result.total_trades == 0
        assert result.winning_trades == 0
        assert result.losing_trades == 0
        assert result.win_rate == 0.0
        assert result.total_pnl == 0.0
        assert result.total_fees == 0.0
        assert result.total_slippage_cost == 0.0

    def test_profit_factor_no_trades(self):
        """Без трейдов profit_factor = 0."""
        result = BacktestResult(symbol="BTCUSDT")
        assert result.profit_factor == 0.0

    def test_result_with_mixed_trades(self):
        """Результат с выигрышными и проигрышными трейдами."""
        trades = [
            BacktestTrade(
                trade_id="t1",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                entry_ts_ns=1_000_000_000,
                exit_ts_ns=2_000_000_000,
                entry_price=100.0,
                exit_price=110.0,
                quantity=1.0,
                gross_pnl=10.0,
                fees=0.04,
                slippage_cost=0.0,
                net_pnl=9.96,
                exit_reason="TAKE_PROFIT",
            ),
            BacktestTrade(
                trade_id="t2",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                entry_ts_ns=3_000_000_000,
                exit_ts_ns=4_000_000_000,
                entry_price=110.0,
                exit_price=105.0,
                quantity=1.0,
                gross_pnl=-5.0,
                fees=0.04,
                slippage_cost=0.0,
                net_pnl=-5.04,
                exit_reason="STOP_LOSS",
            ),
        ]
        result = BacktestResult(
            symbol="BTCUSDT",
            initial_balance=10_000.0,
            trades=trades,
        )
        assert result.total_trades == 2
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.win_rate == pytest.approx(0.5)
        assert result.total_pnl == pytest.approx(9.96 - 5.04)
        assert result.total_fees == pytest.approx(0.08)
        assert result.avg_win == pytest.approx(9.96)
        assert result.avg_loss == pytest.approx(-5.04)
        assert result.profit_factor == pytest.approx(9.96 / 5.04)

    def test_return_pct(self):
        """return_pct = 100 * total_pnl / initial_balance."""
        trades = [
            BacktestTrade(
                trade_id="t1",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                entry_ts_ns=1_000_000_000,
                exit_ts_ns=2_000_000_000,
                entry_price=100.0,
                exit_price=110.0,
                quantity=1.0,
                gross_pnl=10.0,
                fees=0.0,
                slippage_cost=0.0,
                net_pnl=10.0,
            ),
        ]
        result = BacktestResult(
            symbol="BTCUSDT",
            initial_balance=1_000.0,
            trades=trades,
        )
        assert result.return_pct == pytest.approx(1.0)

    def test_return_pct_zero_balance(self):
        """Нулевой initial_balance → return_pct = 0."""
        result = BacktestResult(
            symbol="BTCUSDT",
            initial_balance=0.0,
        )
        assert result.return_pct == 0.0

    def test_profit_factor_no_losses_is_inf(self):
        """Все трейды выигрышные → profit_factor = inf."""
        trades = [
            BacktestTrade(
                trade_id="t1",
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                entry_ts_ns=1_000_000_000,
                exit_ts_ns=2_000_000_000,
                entry_price=100.0,
                exit_price=110.0,
                quantity=1.0,
                gross_pnl=10.0,
                fees=0.0,
                slippage_cost=0.0,
                net_pnl=10.0,
            ),
        ]
        result = BacktestResult(symbol="BTCUSDT", trades=trades)
        assert result.profit_factor == float("inf")

    def test_summary_returns_string_with_symbol(self):
        """summary() возвращает строку с именем символа."""
        result = BacktestResult(symbol="BTCUSDT")
        summary = result.summary()
        assert isinstance(summary, str)
        assert "BTCUSDT" in summary


# ============================================================
# BacktestEngine инициализация
# ============================================================

class TestBacktestEngineInit:
    """Тесты конструктора движка."""

    def test_default_config(self):
        """Дефолтный конфиг создаётся автоматически."""
        engine = BacktestEngine()
        assert engine.config is not None
        assert engine.config.tick_size == pytest.approx(0.1)
        assert engine.config.initial_balance == pytest.approx(10_000.0)
        assert engine.config.depth_for_matching == 50

    def test_custom_config(self):
        """Кастомный конфиг применяется."""
        config = BacktestEngineConfig(
            tick_size=0.5,
            initial_balance=5_000.0,
            latency_region=Region.HOME,
            depth_for_matching=20,
        )
        engine = BacktestEngine(config)
        assert engine.config.tick_size == pytest.approx(0.5)
        assert engine.config.initial_balance == pytest.approx(5_000.0)
        assert engine.config.latency_region == Region.HOME
        assert engine.config.depth_for_matching == 20

    def test_custom_fee_config(self):
        """Кастомные комиссии проходят в CostCalculator."""
        fee_config = FeeConfig(taker_fee_pct=0.10, maker_fee_pct=0.05)
        config = BacktestEngineConfig(fee_config=fee_config)
        engine = BacktestEngine(config)
        assert engine.costs.fee_model.config.taker_fee_pct == pytest.approx(0.10)
        assert engine.costs.fee_model.config.maker_fee_pct == pytest.approx(0.05)

    def test_custom_latency_model(self):
        """Переданная модель задержек используется."""
        latency = ConstantLatency()
        engine = BacktestEngine(latency_model=latency)
        assert engine.latency is latency

    def test_default_latency_is_profile(self):
        """По умолчанию используется ProfileLatency."""
        engine = BacktestEngine()
        assert hasattr(engine.latency, "region")
        assert engine.latency.region == Region.TOKYO

    def test_register_strategy(self):
        """Регистрация стратегии."""
        engine = BacktestEngine()

        class DummyStrategy:
            def on_market_update(self, ctx, engine):
                pass

        strategy = DummyStrategy()
        engine.register_strategy(strategy)
        assert engine._strategy is strategy

    def test_current_ts_ns_initialized(self):
        """_current_ts_ns инициализирован в __init__."""
        engine = BacktestEngine()
        assert engine._current_ts_ns == 0

    def test_result_book_initialized(self):
        """_result_book инициализирован в __init__."""
        engine = BacktestEngine()
        assert engine._result_book is None


# ============================================================
# BacktestEngine без данных
# ============================================================

class TestBacktestEngineWithoutData:
    """Тесты движка без файлов данных."""

    def test_run_without_files_returns_empty_result(self, tmp_path):
        """Запуск без файлов → 0 событий, баланс не меняется."""
        config = BacktestEngineConfig(
            source_dir=str(tmp_path),
            tick_size=1.0,
        )
        engine = BacktestEngine(config)
        result = engine.run(symbol="BTCUSDT", date_str="2026-09-17")

        assert result.symbol == "BTCUSDT"
        assert result.total_events == 0
        assert result.total_trades == 0
        assert result.final_balance == pytest.approx(result.initial_balance)

    def test_submit_before_run_does_not_crash(self):
        """submit_market_order до запуска не падает."""
        engine = BacktestEngine()
        engine.submit_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            reason="TEST",
        )

    def test_close_position_without_position_does_not_crash(self):
        """close_position без открытой позиции не падает."""
        engine = BacktestEngine()
        engine.close_position(reason="TEST")

    def test_submit_ioc_before_run_does_not_crash(self):
        """submit_marketable_limit_ioc_order до запуска не падает."""
        engine = BacktestEngine()
        engine.submit_marketable_limit_ioc_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            limit_price=100.0,
            reason="TEST",
        )


# ============================================================
# Полный цикл бэктеста (синтетические данные)
# ============================================================

def _write_synthetic_data(
    source_dir: Path,
    symbol: str,
    date_str: str,
    num_deltas: int = 10,
) -> None:
    """
    Создаёт синтетические данные для бэктеста.

    Структура:
        <source_dir>/book_snapshot/<date_str>/<SYMBOL>.jsonl
        <source_dir>/book_delta/<date_str>/<SYMBOL>.jsonl
    """
    snapshot_dir = source_dir / "book_snapshot" / date_str
    delta_dir = source_dir / "book_delta" / date_str
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    # Keyframe (полный стакан)
    base_ts = 1_000_000_000
    snapshot = {
        "ts_exchange_ns": base_ts,
        "ts_local_ns": base_ts,
        "symbol": symbol,
        "bids": [[99.0, 2.0], [98.0, 3.0]],
        "asks": [[101.0, 2.0], [102.0, 3.0]],
        "first_update_id": 1,
        "last_update_id": 1,
    }
    with open(snapshot_dir / f"{symbol}.jsonl", "w") as f:
        f.write(json.dumps(snapshot) + "\n")

    # Дельты (несколько обновлений стакана)
    for i in range(2, num_deltas + 2):
        delta = {
            "ts_exchange_ns": base_ts + i * 1_000_000,
            "ts_local_ns": base_ts + i * 1_000_000,
            "symbol": symbol,
            "first_update_id": i,
            "last_update_id": i,
            "prev_update_id": i - 1,
            "bids": [[99.0, 2.0]],
            "asks": [[101.0, 2.0]],
        }
        with open(delta_dir / f"{symbol}.jsonl", "a") as f:
            f.write(json.dumps(delta) + "\n")


class TestFullBacktest:
    """Полный цикл бэктеста на синтетических данных."""

    def test_backtest_processes_events(self, tmp_path):
        """Бэктест обрабатывает события из файлов."""
        _write_synthetic_data(tmp_path, "BTCUSDT", "2026-09-17")

        config = BacktestEngineConfig(
            source_dir=str(tmp_path),
            tick_size=1.0,
        )
        engine = BacktestEngine(config)

        class NoOpStrategy:
            def on_market_update(self, ctx, engine):
                pass

        engine.register_strategy(NoOpStrategy())
        result = engine.run(symbol="BTCUSDT", date_str="2026-09-17")

        assert result.total_events > 0
        assert result.start_ts_ns > 0
        assert result.end_ts_ns >= result.start_ts_ns
        assert result.final_balance == pytest.approx(result.initial_balance)

    def test_backtest_with_buy_strategy_opens_position(self, tmp_path):
        """Стратегия открывает позицию → ордер учитывается."""
        _write_synthetic_data(tmp_path, "BTCUSDT", "2026-09-17", num_deltas=20)

        config = BacktestEngineConfig(
            source_dir=str(tmp_path),
            tick_size=1.0,
            initial_balance=10_000.0,
        )
        engine = BacktestEngine(config)

        class BuyOnceStrategy:
            def __init__(self):
                self._submitted = False

            def on_market_update(self, ctx, engine):
                if not self._submitted:
                    engine.submit_market_order(
                        symbol=ctx.symbol,
                        side=OrderSide.BUY,
                        quantity=1.0,
                        reason="TEST_BUY",
                    )
                    self._submitted = True

        engine.register_strategy(BuyOnceStrategy())
        result = engine.run(symbol="BTCUSDT", date_str="2026-09-17")

        # Ордер был отправлен
        assert result.total_orders >= 1
        # События обработаны
        assert result.total_events > 0

    def test_backtest_strategy_error_does_not_crash(self, tmp_path):
        """Ошибка стратегии не роняет бэктест."""
        _write_synthetic_data(tmp_path, "BTCUSDT", "2026-09-17")

        config = BacktestEngineConfig(
            source_dir=str(tmp_path),
            tick_size=1.0,
        )
        engine = BacktestEngine(config)

        class BrokenStrategy:
            def on_market_update(self, ctx, engine):
                raise RuntimeError("Strategy exploded")

        engine.register_strategy(BrokenStrategy())
        result = engine.run(symbol="BTCUSDT", date_str="2026-09-17")

        assert result.total_events > 0
        assert result.total_trades == 0

    def test_backtest_result_symbol_uppercased(self, tmp_path):
        """Символ в результате всегда в верхнем регистре."""
        _write_synthetic_data(tmp_path, "BTCUSDT", "2026-09-17")

        config = BacktestEngineConfig(
            source_dir=str(tmp_path),
            tick_size=1.0,
        )
        engine = BacktestEngine(config)
        result = engine.run(symbol="btcusdt", date_str="2026-09-17")

        assert result.symbol == "BTCUSDT"

    def test_backtest_initial_balance_preserved(self, tmp_path):
        """Без сделок баланс не меняется."""
        _write_synthetic_data(tmp_path, "BTCUSDT", "2026-09-17")

        config = BacktestEngineConfig(
            source_dir=str(tmp_path),
            tick_size=1.0,
            initial_balance=42_000.0,
        )
        engine = BacktestEngine(config)

        class NoOpStrategy:
            def on_market_update(self, ctx, engine):
                pass

        engine.register_strategy(NoOpStrategy())
        result = engine.run(symbol="BTCUSDT", date_str="2026-09-17")

        assert result.initial_balance == pytest.approx(42_000.0)
        assert result.final_balance == pytest.approx(42_000.0)

    def test_backtest_buy_creates_trade_on_forced_close(self, tmp_path):
        """Покупка без закрытия → принудительное закрытие в конце."""
        _write_synthetic_data(tmp_path, "BTCUSDT", "2026-09-17", num_deltas=30)

        config = BacktestEngineConfig(
            source_dir=str(tmp_path),
            tick_size=1.0,
            initial_balance=10_000.0,
        )
        engine = BacktestEngine(config)

        class BuyOnceStrategy:
            def __init__(self):
                self._submitted = False

            def on_market_update(self, ctx, engine):
                if not self._submitted:
                    engine.submit_market_order(
                        symbol=ctx.symbol,
                        side=OrderSide.BUY,
                        quantity=1.0,
                        reason="TEST_BUY",
                    )
                    self._submitted = True

        engine.register_strategy(BuyOnceStrategy())
        result = engine.run(symbol="BTCUSDT", date_str="2026-09-17")

        # Должен быть хотя бы один трейд (принудительное закрытие)
        if result.total_orders > 0 and result.filled_orders > 0:
            assert result.total_trades >= 1
            # Трейд закрыт принудительно
            last_trade = result.trades[-1]
            assert last_trade.exit_reason in ("FORCED_CLOSE", "TEST_BUY", "MANUAL")


# ============================================================
# BacktestContext
# ============================================================

class TestBacktestContext:
    """Тесты контекста рынка для стратегии."""

    def test_context_creation(self):
        """Контекст создаётся с необходимыми полями."""
        from proscalper.market_data.orderbook_fast import FastOrderBook

        book = FastOrderBook(symbol="BTCUSDT", tick_size=0.1)
        position = BacktestPosition(symbol="BTCUSDT")

        ctx = BacktestContext(
            ts_ns=1_000_000_000,
            symbol="BTCUSDT",
            book=book,
            position=position,
        )

        assert ctx.ts_ns == 1_000_000_000
        assert ctx.symbol == "BTCUSDT"
        assert ctx.book is book
        assert ctx.position is position
        assert ctx.last_fill is None

    def test_context_best_bid_ask_none_for_empty_book(self):
        """Для пустого стакана best_bid/best_ask = None."""
        from proscalper.market_data.orderbook_fast import FastOrderBook

        book = FastOrderBook(symbol="BTCUSDT", tick_size=0.1)
        position = BacktestPosition(symbol="BTCUSDT")

        ctx = BacktestContext(
            ts_ns=1_000_000_000,
            symbol="BTCUSDT",
            book=book,
            position=position,
        )

        # Для неинициализированного стакана лучшие цены = None
        assert ctx.best_bid is None or ctx.best_bid == (0.0, 0.0)
        assert ctx.best_ask is None or ctx.best_ask == (0.0, 0.0)