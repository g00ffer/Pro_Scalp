"""
Отчёт по результатам бэктеста (Report Generator).

Принимает BacktestResult, считает продвинутые метрики, строит
текстовый и табличный вывод, сохраняет в JSON/CSV.

Что считает:
- базовые метрики (PnL, win rate, profit factor) — уже в BacktestResult
- продвинутые:
  - max drawdown
  - Sharpe ratio (по трейдам или по equity curve)
  - Sortino ratio
  - Calmar ratio
  - expectancy per trade
  - Kelly fraction
  - средняя длительность в позиции
  - распределение по exit_reason
  - распределение по r_multiple
- equity curve
- консистентность: как результат распределён по времени

Что выводит:
- summary() — короткая текстовая сводка
- detailed() — развёрнутый отчёт с таблицами
- to_dict() — сериализация метрик для JSON
- to_jsonl() — трейды в JSONL для дальнейшего анализа
- to_csv() — трейды в CSV

Использование:
    from proscalper.backtest.report import BacktestReport

    report = BacktestReport(result)
    print(report.summary())
    report.save_all("./reports/BTCUSDT_2026-09-16")
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from proscalper.backtest.engine import BacktestResult, BacktestTrade, PositionSide


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class ReportConfig:
    """Конфигурация генератора отчёта."""
    # "risk_free_rate" для Sharpe/Sortino. 0 = не учитываем
    risk_free_rate_annual: float = 0.0
    # Считать Sharpe/Sortino по трейдам (trade-based) или по equity curve
    metrics_basis: str = "trade"  # "trade" | "equity"
    # Окно для расчёта equity curve (в трейдах)
    equity_curve_step: int = 1


# ============================================================
# Продвинутые метрики
# ============================================================

@dataclass
class AdvancedMetrics:
    """Расширенные метрики производительности."""
    # Drawdown
    max_drawdown: float = 0.0          # в абсолютных единицах (USDT)
    max_drawdown_pct: float = 0.0      # в % от initial_balance
    max_drawdown_start_ns: int = 0
    max_drawdown_end_ns: int = 0

    # Sharpe / Sortino
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Expectancy
    expectancy: float = 0.0            # средний PnL на сделку
    expectancy_r: float = 0.0          # средний R-multiple
    kelly_fraction: float = 0.0        # оптимальная доля капитала

    # Длительность
    avg_duration_ms: float = 0.0
    median_duration_ms: float = 0.0
    max_duration_ms: int = 0
    min_duration_ms: int = 0

    # Distributions
    exit_reason_distribution: Dict[str, int] = field(default_factory=dict)
    r_multiple_distribution: Dict[str, int] = field(default_factory=dict)

    # Устойчивость
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    max_consecutive_losses_pct: float = 0.0


# ============================================================
# Report Generator
# ============================================================

class BacktestReport:
    """
    Генератор отчёта по BacktestResult.
    """

    def __init__(
        self,
        result: BacktestResult,
        config: Optional[ReportConfig] = None,
    ) -> None:
        self.result = result
        self.config = config or ReportConfig()

        # Лениво вычисляем метрики
        self._advanced: Optional[AdvancedMetrics] = None
        self._equity_curve: Optional[List[Tuple[int, float]]] = None

    # ============================================
    # Продвинутые метрики
    # ============================================

    def advanced_metrics(self) -> AdvancedMetrics:
        if self._advanced is not None:
            return self._advanced

        m = AdvancedMetrics()
        trades = self.result.trades

        if not trades:
            self._advanced = m
            return m

        # --- Drawdown ---
        self._compute_drawdown(m)

        # --- Sharpe / Sortino ---
        self._compute_sharpe_sortino(m)

        # --- Expectancy / Kelly ---
        self._compute_expectancy_kelly(m)

        # --- Длительность ---
        durations = [t.duration_ms for t in trades]
        m.avg_duration_ms = sum(durations) / len(durations)
        sorted_d = sorted(durations)
        m.median_duration_ms = float(sorted_d[len(sorted_d) // 2])
        m.max_duration_ms = max(durations)
        m.min_duration_ms = min(durations)

        # --- Распределение exit_reason ---
        reason_counter: Counter = Counter()
        for t in trades:
            reason_counter[t.exit_reason or "UNKNOWN"] += 1
        m.exit_reason_distribution = dict(reason_counter)

        # --- Распределение R-multiple ---
        r_buckets: Dict[str, int] = defaultdict(int)
        for t in trades:
            r_buckets[self._r_bucket(t.r_multiple)] += 1
        m.r_multiple_distribution = dict(r_buckets)

        # --- Streaks ---
        win_streak = loss_streak = 0
        max_win_streak = max_loss_streak = 0
        for t in trades:
            if t.net_pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif t.net_pnl < 0:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        m.longest_win_streak = max_win_streak
        m.longest_loss_streak = max_loss_streak

        # Доля максимальной серии убытков
        if self.result.total_trades > 0:
            m.max_consecutive_losses_pct = (
                100.0 * max_loss_streak / self.result.total_trades
            )

        self._advanced = m
        return m

    def equity_curve(self) -> List[Tuple[int, float]]:
        """Кривая капитала: [(ts_ns, equity), ...]."""
        if self._equity_curve is not None:
            return self._equity_curve

        trades = sorted(self.result.trades, key=lambda t: t.exit_ts_ns)
        balance = self.result.initial_balance
        curve: List[Tuple[int, float]] = [(0, balance)]

        for t in trades:
            balance += t.net_pnl
            curve.append((t.exit_ts_ns, balance))

        self._equity_curve = curve
        return curve

    # ============================================
    # Вывод
    # ============================================

    def summary(self) -> str:
        """Короткая текстовая сводка."""
        m = self.advanced_metrics()
        r = self.result

        lines = [
            f"=== BACKTEST REPORT: {r.symbol} ===",
            f"Trades:          {r.total_trades} "
            f"(win {r.winning_trades} / loss {r.losing_trades})",
            f"Win rate:        {r.win_rate * 100:.1f}%",
            f"Profit factor:   {r.profit_factor:.2f}",
            f"Total PnL:       {r.total_pnl:+.2f} ({r.return_pct:+.2f}%)",
            f"Avg win:         {r.avg_win:+.2f}",
            f"Avg loss:        {r.avg_loss:+.2f}",
            f"Expectancy:      {m.expectancy:+.2f} USDT "
            f"({m.expectancy_r:+.2f} R)",
            f"Fees:            {r.total_fees:.2f}",
            f"Slippage cost:   {r.total_slippage_cost:.2f}",
            f"---",
            f"Max drawdown:    {m.max_drawdown:.2f} "
            f"({m.max_drawdown_pct:.2f}%)",
            f"Sharpe:          {m.sharpe_ratio:.2f}",
            f"Sortino:         {m.sortino_ratio:.2f}",
            f"Calmar:          {m.calmar_ratio:.2f}",
            f"Kelly:           {m.kelly_fraction * 100:.1f}%",
            f"---",
            f"Avg duration:    {m.avg_duration_ms:.0f} ms",
            f"Median duration: {m.median_duration_ms:.0f} ms",
            f"Longest win str: {m.longest_win_streak}",
            f"Longest loss str:{m.longest_loss_streak}",
            f"---",
            f"Orders:          {r.total_orders} "
            f"(filled {r.filled_orders}, "
            f"rejected {r.rejected_orders}, "
            f"skipped {r.skipped_orders})",
            f"Events processed:{r.total_events}",
        ]
        return "\n".join(lines)

    def detailed(self) -> str:
        """Развёрнутый отчёт с таблицами."""
        m = self.advanced_metrics()
        parts = [self.summary(), ""]

        # Exit reasons
        if m.exit_reason_distribution:
            parts.append("=== Exit Reasons ===")
            total = sum(m.exit_reason_distribution.values())
            for reason, count in sorted(
                m.exit_reason_distribution.items(),
                key=lambda x: -x[1],
            ):
                pct = 100.0 * count / total if total else 0.0
                parts.append(f"  {reason:<25} {count:>5} ({pct:>5.1f}%)")
            parts.append("")

        # R-multiple distribution
        if m.r_multiple_distribution:
            parts.append("=== R-Multiple Distribution ===")
            for bucket in sorted(m.r_multiple_distribution.keys()):
                count = m.r_multiple_distribution[bucket]
                parts.append(f"  {bucket:<15} {count:>5}")
            parts.append("")

        # Trade list (top 10 best / worst)
        trades_sorted = sorted(self.result.trades, key=lambda t: t.net_pnl)
        if trades_sorted:
            parts.append("=== Worst 5 Trades ===")
            for t in trades_sorted[:5]:
                parts.append(
                    f"  {t.side.value:<6} entry={t.entry_price:.2f} "
                    f"exit={t.exit_price:.2f} pnl={t.net_pnl:+.2f} "
                    f"({t.exit_reason})"
                )
            parts.append("")

            parts.append("=== Best 5 Trades ===")
            for t in trades_sorted[-5:]:
                parts.append(
                    f"  {t.side.value:<6} entry={t.entry_price:.2f} "
                    f"exit={t.exit_price:.2f} pnl={t.net_pnl:+.2f} "
                    f"({t.exit_reason})"
                )

        return "\n".join(parts)

    # ============================================
    # Сериализация
    # ============================================

    def to_dict(self) -> Dict[str, Any]:
        """Метрики в dict (для JSON)."""
        m = self.advanced_metrics()
        r = self.result

        return {
            "symbol": r.symbol,
            "start_ts_ns": r.start_ts_ns,
            "end_ts_ns": r.end_ts_ns,
            "initial_balance": r.initial_balance,
            "final_balance": r.final_balance,
            "summary": {
                "total_trades": r.total_trades,
                "winning_trades": r.winning_trades,
                "losing_trades": r.losing_trades,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "total_pnl": r.total_pnl,
                "return_pct": r.return_pct,
                "total_fees": r.total_fees,
                "total_slippage_cost": r.total_slippage_cost,
                "avg_win": r.avg_win,
                "avg_loss": r.avg_loss,
            },
            "advanced": {
                "max_drawdown": m.max_drawdown,
                "max_drawdown_pct": m.max_drawdown_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "calmar_ratio": m.calmar_ratio,
                "expectancy": m.expectancy,
                "expectancy_r": m.expectancy_r,
                "kelly_fraction": m.kelly_fraction,
                "avg_duration_ms": m.avg_duration_ms,
                "median_duration_ms": m.median_duration_ms,
                "longest_win_streak": m.longest_win_streak,
                "longest_loss_streak": m.longest_loss_streak,
                "exit_reason_distribution": m.exit_reason_distribution,
                "r_multiple_distribution": m.r_multiple_distribution,
            },
            "orders": {
                "total": r.total_orders,
                "filled": r.filled_orders,
                "rejected": r.rejected_orders,
                "skipped": r.skipped_orders,
            },
            "events_processed": r.total_events,
        }

    def trades_to_jsonl(self) -> str:
        """Трейды в формате JSONL (одна строка — один trade)."""
        lines = []
        for t in self.result.trades:
            lines.append(json.dumps({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side.value,
                "entry_ts_ns": t.entry_ts_ns,
                "exit_ts_ns": t.exit_ts_ns,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "gross_pnl": t.gross_pnl,
                "fees": t.fees,
                "slippage_cost": t.slippage_cost,
                "net_pnl": t.net_pnl,
                "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
                "duration_ms": t.duration_ms,
                "r_multiple": t.r_multiple,
            }, ensure_ascii=False))
        return "\n".join(lines)

    def save_all(self, output_dir: str) -> None:
        """
        Сохраняет все отчёты в директорию:
        - summary.txt
        - detailed.txt
        - metrics.json
        - trades.jsonl
        - trades.csv
        - equity_curve.csv
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "summary.txt").write_text(self.summary(), encoding="utf-8")
        (out / "detailed.txt").write_text(self.detailed(), encoding="utf-8")

        (out / "metrics.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        (out / "trades.jsonl").write_text(
            self.trades_to_jsonl(),
            encoding="utf-8",
        )

        self._save_trades_csv(out / "trades.csv")
        self._save_equity_csv(out / "equity_curve.csv")

    # ============================================
    # Внутренние методы
    # ============================================

    def _save_trades_csv(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id", "symbol", "side",
                "entry_ts_ns", "exit_ts_ns",
                "entry_price", "exit_price", "quantity",
                "gross_pnl", "fees", "slippage_cost", "net_pnl",
                "entry_reason", "exit_reason",
                "duration_ms", "r_multiple",
            ])
            for t in self.result.trades:
                writer.writerow([
                    t.trade_id, t.symbol, t.side.value,
                    t.entry_ts_ns, t.exit_ts_ns,
                    t.entry_price, t.exit_price, t.quantity,
                    t.gross_pnl, t.fees, t.slippage_cost, t.net_pnl,
                    t.entry_reason, t.exit_reason,
                    t.duration_ms, t.r_multiple,
                ])

    def _save_equity_csv(self, path: Path) -> None:
        curve = self.equity_curve()
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_ns", "equity"])
            for ts, eq in curve:
                writer.writerow([ts, eq])

    # -------- Расчёты --------

    def _compute_drawdown(self, m: AdvancedMetrics) -> None:
        curve = self.equity_curve()
        if len(curve) < 2:
            return

        peak = curve[0][1]
        peak_ts = curve[0][0]
        max_dd = 0.0
        max_dd_start = 0
        max_dd_end = 0

        for ts, eq in curve[1:]:
            if eq > peak:
                peak = eq
                peak_ts = ts
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
                max_dd_start = peak_ts
                max_dd_end = ts

        m.max_drawdown = max_dd
        if self.result.initial_balance > 0:
            m.max_drawdown_pct = (
                100.0 * max_dd / self.result.initial_balance
            )
        m.max_drawdown_start_ns = max_dd_start
        m.max_drawdown_end_ns = max_dd_end

    def _compute_sharpe_sortino(self, m: AdvancedMetrics) -> None:
        trades = self.result.trades
        if len(trades) < 2:
            return

        returns = [t.net_pnl for t in trades]
        n = len(returns)
        mean = sum(returns) / n

        # Std
        variance = sum((x - mean) ** 2 for x in returns) / (n - 1)
        std = math.sqrt(variance)

        # Downside deviation (только отрицательные)
        negatives = [x for x in returns if x < 0]
        if negatives:
            downside_var = sum(x ** 2 for x in negatives) / len(negatives)
            downside_std = math.sqrt(downside_var)
        else:
            downside_std = 0.0

        # Risk-free rate per trade (упрощённо: доля от среднего риска)
        # Здесь rf_annual конвертируем в per-trade через n (не идеально,
        # но консервативно)
        rf_per_trade = (
            self.result.initial_balance
            * self.config.risk_free_rate_annual
            / max(n, 1)
        )

        excess_mean = mean - rf_per_trade

        if std > 0:
            m.sharpe_ratio = excess_mean / std * math.sqrt(n)

        if downside_std > 0:
            m.sortino_ratio = excess_mean / downside_std * math.sqrt(n)

        # Calmar = return / max_dd
        if m.max_drawdown > 0:
            m.calmar_ratio = self.result.total_pnl / m.max_drawdown

    def _compute_expectancy_kelly(self, m: AdvancedMetrics) -> None:
        trades = self.result.trades
        if not trades:
            return

        # Expectancy = средний net_pnl на сделку
        m.expectancy = self.result.total_pnl / len(trades)

        # Expectancy в R
        r_values = [t.r_multiple for t in trades if t.r_multiple != 0]
        if r_values:
            m.expectancy_r = sum(r_values) / len(r_values)

        # Kelly fraction
        # f* = (p * avg_win - q * avg_loss) / (avg_win * avg_loss)
        # при этом avg_win/avg_loss — в абсолютных значениях
        p = self.result.win_rate
        q = 1.0 - p
        avg_win = self.result.avg_win
        avg_loss = abs(self.result.avg_loss)

        if avg_win > 0 and avg_loss > 0:
            m.kelly_fraction = (p * avg_win - q * avg_loss) / (avg_win * avg_loss)
            # Ограничиваем [0, 1] — Kelly может быть отрицательным
            m.kelly_fraction = max(0.0, min(1.0, m.kelly_fraction))

    def _r_bucket(self, r: float) -> str:
        """Разбиение R-multiple на bucket."""
        if r <= -1.0:
            return "<=-1R"
        if r < 0:
            return "(-1, 0)R"
        if r < 0.5:
            return "[0, 0.5)R"
        if r < 1.0:
            return "[0.5, 1)R"
        if r < 2.0:
            return "[1, 2)R"
        if r < 3.0:
            return "[2, 3)R"
        return ">=3R"