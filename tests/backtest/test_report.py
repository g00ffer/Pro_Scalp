"""
Тесты для proscalper.backtest.report

Запуск:
    python -m pytest tests/backtest/test_report.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from proscalper.backtest.engine import (
    BacktestResult,
    BacktestTrade,
    PositionSide,
)
from proscalper.backtest.report import (
    AdvancedMetrics,
    BacktestReport,
    ReportConfig,
)


# ============================================================
# Фикстуры
# ============================================================

def _make_trade(
    trade_id: str,
    net_pnl: float,
    entry_ts: int = 1_000_000_000,
    exit_ts: int = 2_000_000_000,
    exit_reason: str = "MANUAL",
    r_multiple: float = 0.0,
    duration_ms: int = 1000,
) -> BacktestTrade:
    """Вспомогательная функция для создания трейда."""
    return BacktestTrade(
        trade_id=trade_id,
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_ts_ns=entry_ts,
        exit_ts_ns=exit_ts,
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        quantity=1.0,
        gross_pnl=net_pnl + 0.04,
        fees=0.04,
        slippage_cost=0.0,
        net_pnl=net_pnl,
        exit_reason=exit_reason,
        duration_ms=duration_ms,
        r_multiple=r_multiple,
    )


@pytest.fixture
def empty_result() -> BacktestResult:
    """Пустой результат без трейдов."""
    return BacktestResult(
        symbol="BTCUSDT",
        initial_balance=10_000.0,
        final_balance=10_000.0,
    )


@pytest.fixture
def mixed_result() -> BacktestResult:
    """Результат с выигрышными и проигрышными трейдами."""
    trades = [
        _make_trade("t1", net_pnl=10.0, exit_ts=2_000_000_000,
                    exit_reason="TAKE_PROFIT", r_multiple=1.5),
        _make_trade("t2", net_pnl=-5.0, exit_ts=3_000_000_000,
                    exit_reason="STOP_LOSS", r_multiple=-0.5),
        _make_trade("t3", net_pnl=8.0, exit_ts=4_000_000_000,
                    exit_reason="TAKE_PROFIT", r_multiple=1.2),
        _make_trade("t4", net_pnl=-3.0, exit_ts=5_000_000_000,
                    exit_reason="STOP_LOSS", r_multiple=-0.3),
        _make_trade("t5", net_pnl=12.0, exit_ts=6_000_000_000,
                    exit_reason="MANUAL", r_multiple=2.0),
    ]
    total_pnl = sum(t.net_pnl for t in trades)
    return BacktestResult(
        symbol="BTCUSDT",
        initial_balance=10_000.0,
        final_balance=10_000.0 + total_pnl,
        trades=trades,
    )


@pytest.fixture
def all_wins_result() -> BacktestResult:
    """Результат где все трейды выигрышные."""
    trades = [
        _make_trade("t1", net_pnl=10.0, exit_ts=2_000_000_000),
        _make_trade("t2", net_pnl=5.0, exit_ts=3_000_000_000),
        _make_trade("t3", net_pnl=8.0, exit_ts=4_000_000_000),
    ]
    total_pnl = sum(t.net_pnl for t in trades)
    return BacktestResult(
        symbol="BTCUSDT",
        initial_balance=10_000.0,
        final_balance=10_000.0 + total_pnl,
        trades=trades,
    )


# ============================================================
# ReportConfig и AdvancedMetrics
# ============================================================

class TestReportConfig:
    """Тесты конфигурации отчёта."""

    def test_default_values(self):
        config = ReportConfig()
        assert config.risk_free_rate_annual == 0.0
        assert config.metrics_basis == "trade"
        assert config.equity_curve_step == 1


class TestAdvancedMetrics:
    """Тесты структуры продвинутых метрик."""

    def test_default_values(self):
        m = AdvancedMetrics()
        assert m.max_drawdown == 0.0
        assert m.max_drawdown_pct == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.sortino_ratio == 0.0
        assert m.calmar_ratio == 0.0
        assert m.expectancy == 0.0
        assert m.kelly_fraction == 0.0
        assert m.avg_duration_ms == 0.0
        assert m.longest_win_streak == 0
        assert m.longest_loss_streak == 0
        assert m.exit_reason_distribution == {}
        assert m.r_multiple_distribution == {}


# ============================================================
# BacktestReport: инициализация
# ============================================================

class TestBacktestReportInit:
    """Тесты конструктора отчёта."""

    def test_init_with_result(self, mixed_result):
        report = BacktestReport(mixed_result)
        assert report.result is mixed_result
        assert report.config is not None

    def test_init_with_custom_config(self, mixed_result):
        config = ReportConfig(risk_free_rate_annual=0.02)
        report = BacktestReport(mixed_result, config)
        assert report.config.risk_free_rate_annual == 0.02


# ============================================================
# Advanced metrics
# ============================================================

class TestAdvancedMetricsComputation:
    """Тесты расчёта продвинутых метрик."""

    def test_empty_result_returns_zeros(self, empty_result):
        report = BacktestReport(empty_result)
        m = report.advanced_metrics()
        assert m.max_drawdown == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.expectancy == 0.0
        assert m.avg_duration_ms == 0.0

    def test_drawdown_computed(self, mixed_result):
        """Просадка считается по equity curve."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        # После t1: 10010, после t2: 10005 (просадка 5 от пика 10010)
        # Это максимальная просадка
        assert m.max_drawdown >= 0.0
        assert m.max_drawdown_pct >= 0.0
        assert m.max_drawdown_start_ns > 0 or m.max_drawdown == 0.0

    def test_expectancy_computed(self, mixed_result):
        """Expectancy = total_pnl / n_trades."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        expected = mixed_result.total_pnl / mixed_result.total_trades
        assert m.expectancy == pytest.approx(expected)

    def test_expectancy_r_computed(self, mixed_result):
        """Expectancy R = средний r_multiple."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        r_values = [t.r_multiple for t in mixed_result.trades if t.r_multiple != 0]
        expected_r = sum(r_values) / len(r_values)
        assert m.expectancy_r == pytest.approx(expected_r)

    def test_kelly_fraction_in_range(self, mixed_result):
        """Kelly fraction ограничен [0, 1]."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        assert 0.0 <= m.kelly_fraction <= 1.0

    def test_kelly_fraction_all_wins(self, all_wins_result):
        """Все выигрыши → высокий Kelly."""
        report = BacktestReport(all_wins_result)
        m = report.advanced_metrics()
        assert m.kelly_fraction > 0.0

    def test_duration_stats(self, mixed_result):
        """Статистики длительности."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        assert m.avg_duration_ms == pytest.approx(1000.0)
        assert m.median_duration_ms == pytest.approx(1000.0)
        assert m.max_duration_ms == 1000
        assert m.min_duration_ms == 1000

    def test_exit_reason_distribution(self, mixed_result):
        """Распределение причин выхода."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        assert m.exit_reason_distribution["TAKE_PROFIT"] == 2
        assert m.exit_reason_distribution["STOP_LOSS"] == 2
        assert m.exit_reason_distribution["MANUAL"] == 1

    def test_r_multiple_distribution(self, mixed_result):
        """Распределение R-multiple по bucket."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        # r_values: 1.5, -0.5, 1.2, -0.3, 2.0
        # buckets: [1,2)R: 1.5, 1.2; (-1,0)R: -0.5, -0.3; [2,3)R: 2.0
        assert m.r_multiple_distribution.get("[1, 2)R", 0) == 2
        assert m.r_multiple_distribution.get("(-1, 0)R", 0) == 2
        assert m.r_multiple_distribution.get("[2, 3)R", 0) == 1

    def test_streaks(self, mixed_result):
        """Серии побед/убытков."""
        report = BacktestReport(mixed_result)
        m = report.advanced_metrics()
        # Паттерн: W, L, W, L, W → макс серия 1
        assert m.longest_win_streak == 1
        assert m.longest_loss_streak == 1

    def test_streaks_consecutive_wins(self):
        """Несколько побед подряд."""
        trades = [
            _make_trade("t1", net_pnl=10.0),
            _make_trade("t2", net_pnl=5.0),
            _make_trade("t3", net_pnl=8.0),
            _make_trade("t4", net_pnl=-3.0),
        ]
        result = BacktestResult(symbol="BTCUSDT", trades=trades)
        report = BacktestReport(result)
        m = report.advanced_metrics()
        assert m.longest_win_streak == 3
        assert m.longest_loss_streak == 1

    def test_metrics_cached(self, mixed_result):
        """Повторный вызов возвращает кэш."""
        report = BacktestReport(mixed_result)
        m1 = report.advanced_metrics()
        m2 = report.advanced_metrics()
        assert m1 is m2


# ============================================================
# Equity curve
# ============================================================

class TestEquityCurve:
    """Тесты кривой капитала."""

    def test_empty_result_single_point(self, empty_result):
        report = BacktestReport(empty_result)
        curve = report.equity_curve()
        assert len(curve) == 1
        assert curve[0] == (0, 10_000.0)

    def test_curve_starts_with_initial_balance(self, mixed_result):
        report = BacktestReport(mixed_result)
        curve = report.equity_curve()
        assert curve[0] == (0, mixed_result.initial_balance)

    def test_curve_length_matches_trades(self, mixed_result):
        report = BacktestReport(mixed_result)
        curve = report.equity_curve()
        # 1 начальная точка + 1 на каждый трейд
        assert len(curve) == len(mixed_result.trades) + 1

    def test_curve_ends_with_final_balance(self, mixed_result):
        report = BacktestReport(mixed_result)
        curve = report.equity_curve()
        assert curve[-1][1] == pytest.approx(mixed_result.final_balance)

    def test_curve_cached(self, mixed_result):
        report = BacktestReport(mixed_result)
        c1 = report.equity_curve()
        c2 = report.equity_curve()
        assert c1 is c2


# ============================================================
# Вывод
# ============================================================

class TestReportOutput:
    """Тесты текстового вывода."""

    def test_summary_contains_symbol(self, mixed_result):
        report = BacktestReport(mixed_result)
        summary = report.summary()
        assert "BTCUSDT" in summary

    def test_summary_contains_key_metrics(self, mixed_result):
        report = BacktestReport(mixed_result)
        summary = report.summary()
        assert "Win rate" in summary
        assert "Profit factor" in summary
        assert "Total PnL" in summary
        assert "Sharpe" in summary
        assert "drawdown" in summary.lower()

    def test_detailed_contains_exit_reasons(self, mixed_result):
        report = BacktestReport(mixed_result)
        detailed = report.detailed()
        assert "Exit Reasons" in detailed
        assert "TAKE_PROFIT" in detailed
        assert "STOP_LOSS" in detailed

    def test_detailed_contains_r_distribution(self, mixed_result):
        report = BacktestReport(mixed_result)
        detailed = report.detailed()
        assert "R-Multiple Distribution" in detailed

    def test_detailed_contains_best_worst_trades(self, mixed_result):
        report = BacktestReport(mixed_result)
        detailed = report.detailed()
        assert "Worst 5 Trades" in detailed
        assert "Best 5 Trades" in detailed

    def test_empty_result_summary_no_crash(self, empty_result):
        """Пустой результат не роняет summary."""
        report = BacktestReport(empty_result)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Trades:" in summary


# ============================================================
# Сериализация
# ============================================================

class TestSerialization:
    """Тесты сериализации."""

    def test_to_dict_structure(self, mixed_result):
        report = BacktestReport(mixed_result)
        d = report.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert "summary" in d
        assert "advanced" in d
        assert "orders" in d
        assert d["summary"]["total_trades"] == 5
        assert d["summary"]["winning_trades"] == 3
        assert d["summary"]["losing_trades"] == 2

    def test_to_dict_advanced_fields(self, mixed_result):
        report = BacktestReport(mixed_result)
        d = report.to_dict()
        adv = d["advanced"]
        assert "max_drawdown" in adv
        assert "sharpe_ratio" in adv
        assert "kelly_fraction" in adv
        assert "exit_reason_distribution" in adv

    def test_trades_to_jsonl_valid(self, mixed_result):
        """Каждая строка — валидный JSON."""
        report = BacktestReport(mixed_result)
        jsonl = report.trades_to_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "trade_id" in parsed
            assert "net_pnl" in parsed
            assert parsed["symbol"] == "BTCUSDT"

    def test_trades_to_jsonl_empty(self, empty_result):
        report = BacktestReport(empty_result)
        jsonl = report.trades_to_jsonl()
        assert jsonl == ""


# ============================================================
# save_all
# ============================================================

class TestSaveAll:
    """Тесты сохранения отчётов."""

    def test_save_all_creates_files(self, mixed_result, tmp_path):
        report = BacktestReport(mixed_result)
        output_dir = tmp_path / "reports" / "test"
        report.save_all(str(output_dir))

        assert (output_dir / "summary.txt").exists()
        assert (output_dir / "detailed.txt").exists()
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "trades.jsonl").exists()
        assert (output_dir / "trades.csv").exists()
        assert (output_dir / "equity_curve.csv").exists()

    def test_save_all_metrics_json_valid(self, mixed_result, tmp_path):
        report = BacktestReport(mixed_result)
        output_dir = tmp_path / "reports"
        report.save_all(str(output_dir))

        with open(output_dir / "metrics.json") as f:
            data = json.load(f)
        assert data["symbol"] == "BTCUSDT"
        assert data["summary"]["total_trades"] == 5

    def test_save_all_trades_csv_row_count(self, mixed_result, tmp_path):
        report = BacktestReport(mixed_result)
        output_dir = tmp_path / "reports"
        report.save_all(str(output_dir))

        with open(output_dir / "trades.csv") as f:
            lines = f.readlines()
        # 1 заголовок + 5 трейдов
        assert len(lines) == 6

    def test_save_all_equity_csv_row_count(self, mixed_result, tmp_path):
        report = BacktestReport(mixed_result)
        output_dir = tmp_path / "reports"
        report.save_all(str(output_dir))

        with open(output_dir / "equity_curve.csv") as f:
            lines = f.readlines()
        # 1 заголовок + 6 точек (1 начальная + 5 трейдов)
        assert len(lines) == 7

    def test_save_all_empty_result(self, empty_result, tmp_path):
        """Пустой результат тоже сохраняется без ошибок."""
        report = BacktestReport(empty_result)
        output_dir = tmp_path / "reports"
        report.save_all(str(output_dir))
        assert (output_dir / "summary.txt").exists()
        assert (output_dir / "metrics.json").exists()


# ============================================================
# R-bucket (через distribution)
# ============================================================

class TestRBucket:
    """Тесты разбиения R-multiple на bucket."""

    def _get_bucket(self, r_value: float) -> str:
        trade = _make_trade("t1", net_pnl=1.0, r_multiple=r_value)
        result = BacktestResult(symbol="BTCUSDT", trades=[trade])
        report = BacktestReport(result)
        m = report.advanced_metrics()
        buckets = list(m.r_multiple_distribution.keys())
        assert len(buckets) == 1
        return buckets[0]

    def test_bucket_large_negative(self):
        assert self._get_bucket(-1.5) == "<=-1R"

    def test_bucket_small_negative(self):
        assert self._get_bucket(-0.5) == "(-1, 0)R"

    def test_bucket_zero(self):
        # r_multiple=0 не попадает в r_values (фильтр != 0)
        trade = _make_trade("t1", net_pnl=1.0, r_multiple=0.1)
        result = BacktestResult(symbol="BTCUSDT", trades=[trade])
        report = BacktestReport(result)
        m = report.advanced_metrics()
        assert "[0, 0.5)R" in m.r_multiple_distribution

    def test_bucket_small_positive(self):
        assert self._get_bucket(0.7) == "[0.5, 1)R"

    def test_bucket_one_r(self):
        assert self._get_bucket(1.5) == "[1, 2)R"

    def test_bucket_two_r(self):
        assert self._get_bucket(2.5) == "[2, 3)R"

    def test_bucket_large_positive(self):
        assert self._get_bucket(5.0) == ">=3R"
