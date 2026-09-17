"""
Event-driven бэктестер.

Ядро Фазы 6. Прогоняет стратегию по записанным рыночным событиям,
симулирует исполнение ордеров через MatchingEngine, применяет
latency model, ведёт PnL и trade journal.

Принципы:
- НЕТ lookahead: события обрабатываются строго по ts
- детерминированность: одна и та же история → один и тот же результат
- стратегия — внешний объект через StrategyProtocol
- latency применяется через отложенную очередь событий
- все издержки (fees + slippage) учитываются через CostCalculator

Архитектура:
1. Загружаем события дня через ReplayBuilder
2. Прогоняем в хронологическом порядке
3. Для каждого события:
   - обновляем стакан
   - применяем pending fill-события (по latency)
   - вызываем strategy.on_event()
   - если стратегия запросила ордер → matching engine симулирует fill
   - fill применяется через latency (сдвиг ts)
4. В конце — собираем отчёт

Стратегия взаимодействует с движком через интерфейс StrategyProtocol.

Использование:
    class MyStrategy:
        def on_market_update(self, ctx, engine):
            if should_buy:
                engine.submit_market_order(
                    symbol="BTCUSDT", side=OrderSide.BUY,
                    quantity=0.001, reason="BREAKOUT",
                )

    engine = BacktestEngine(config)
    engine.register_strategy(MyStrategy())
    result = engine.run(symbol="BTCUSDT", date_str="2026-09-16")
    print(result.summary())
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Tuple

from proscalper.backtest.costs import CostCalculator, FeeConfig, SlippageConfig
from proscalper.backtest.latency import (
    LatencyModel,
    ProfileLatency,
    Region,
    ms_to_ns,
)
from proscalper.backtest.matching import (
    FillResult,
    MatchStatus,
    MatchingEngine,
)
from proscalper.core.types import OrderSide
from proscalper.market_data.orderbook_fast import FastOrderBook
from proscalper.storage.replay_builder import ReplayBuilder, ReplayConfig


# ============================================================
# Сторона позиции
# ============================================================

class PositionSide(str, Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


# ============================================================
# Позиция
# ============================================================

@dataclass
class BacktestPosition:
    """Текущая позиция в бэктесте."""
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_ts_ns: int = 0
    stop_price: float = 0.0
    target_price: float = 0.0
    signal_id: str = ""

    # Учёт издержек
    entry_fees: float = 0.0
    entry_slippage_cost: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != PositionSide.FLAT and self.quantity > 0


# ============================================================
# Trade (закрытая сделка)
# ============================================================

@dataclass
class BacktestTrade:
    """Записанная закрытая сделка."""
    trade_id: str
    symbol: str
    side: PositionSide
    entry_ts_ns: int
    exit_ts_ns: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    entry_reason: str = ""
    exit_reason: str = ""
    duration_ms: int = 0
    r_multiple: float = 0.0  # PnL / initial_risk


# ============================================================
# Результат бэктеста
# ============================================================

@dataclass
class BacktestResult:
    """Агрегированный результат прогона."""
    symbol: str
    start_ts_ns: int = 0
    end_ts_ns: int = 0
    initial_balance: float = 0.0
    final_balance: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)

    # Метрики
    total_events: int = 0
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    skipped_orders: int = 0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl < 0)

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.fees for t in self.trades)

    @property
    def total_slippage_cost(self) -> float:
        return sum(t.slippage_cost for t in self.trades)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        if gross_loss <= 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def return_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return 100.0 * self.total_pnl / self.initial_balance

    def summary(self) -> str:
        return (
            f"=== Backtest: {self.symbol} ===\n"
            f"  Trades: {self.total_trades} "
            f"(win {self.winning_trades} / loss {self.losing_trades})\n"
            f"  Win rate: {self.win_rate * 100:.1f}%\n"
            f"  Avg win: {self.avg_win:+.2f} | Avg loss: {self.avg_loss:+.2f}\n"
            f"  Profit factor: {self.profit_factor:.2f}\n"
            f"  Total PnL: {self.total_pnl:+.2f} ({self.return_pct:+.2f}%)\n"
            f"  Fees: {self.total_fees:.2f} | Slippage cost: {self.total_slippage_cost:.2f}\n"
            f"  Orders: {self.total_orders} "
            f"(filled {self.filled_orders}, "
            f"rejected {self.rejected_orders}, "
            f"skipped {self.skipped_orders})\n"
            f"  Events: {self.total_events}"
        )


# ============================================================
# Контекст рынка для стратегии
# ============================================================

@dataclass
class BacktestContext:
    """
    Снимок рынка в момент обработки события.
    Стратегия читает book/ts/last_fill_result и вызывает
    методы engine для отправки ордеров.
    """
    ts_ns: int
    symbol: str
    book: FastOrderBook
    position: BacktestPosition
    last_fill: Optional[FillResult] = None

    @property
    def best_bid(self) -> Optional[Tuple[float, float]]:
        return self.book.best_bid

    @property
    def best_ask(self) -> Optional[Tuple[float, float]]:
        return self.book.best_ask

    @property
    def mid_price(self) -> Optional[float]:
        return self.book.mid_price


# ============================================================
# Интерфейс стратегии
# ============================================================

class StrategyProtocol(Protocol):
    """
    Интерфейс пользовательской стратегии.

    Стратегия вызывается на каждом событии. Она может:
    - читать контекст
    - вызывать engine.submit_* для отправки ордеров
    """

    def on_market_update(
        self,
        ctx: BacktestContext,
        engine: "BacktestEngine",
    ) -> None: ...


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class BacktestEngineConfig:
    """Конфигурация бэктестера."""
    # Данные
    source_dir: str = "./data/binance_futures"
    use_parquet: bool = False
    tick_size: float = 0.1
    depth_for_matching: int = 50

    # Издержки
    fee_config: Optional[FeeConfig] = None
    slippage_config: Optional[SlippageConfig] = None

    # Latency
    latency_region: Region = Region.TOKYO

    # Капитал
    initial_balance: float = 10_000.0


# ============================================================
# Движок
# ============================================================

class BacktestEngine:
    """
    Event-driven бэктестер.

    Отвечает за:
    - replay событий
    - применение latency к ордерам
    - симуляцию fills через MatchingEngine
    - учёт позиции, PnL, trade journal
    - вызовы strategy.on_market_update()

    Не отвечает за:
    - торговую логику (это стратегия)
    - риск-менеджмент (в первой версии — стратегия сама)
    """

    def __init__(
        self,
        config: Optional[BacktestEngineConfig] = None,
        latency_model: Optional[LatencyModel] = None,
    ) -> None:
        self.config = config or BacktestEngineConfig()

        self.costs = CostCalculator(
            tick_size=self.config.tick_size,
            fee_config=self.config.fee_config,
            slippage_config=self.config.slippage_config,
        )

        self.matching = MatchingEngine(
            tick_size=self.config.tick_size,
            cost_calculator=self.costs,
        )

        self.latency: LatencyModel = (
            latency_model
            if latency_model is not None
            else ProfileLatency(region=self.config.latency_region)
        )

        self._strategy: Optional[StrategyProtocol] = None
        self._position = BacktestPosition(symbol="")

        # Очередь отложенных fill-событий (по latency)
        # (fill_ts_ns, seq, FillResult, reason, side)
        self._pending_fills: List[Tuple[int, int, FillResult, str, OrderSide]] = []
        self._fill_seq_counter = 0
        self._result: Optional[BacktestResult] = None

        # FIX #1: инициализация обязательна — без этого при пустых данных
        # падает AttributeError в _apply_due_fills / submit_market_order
        self._result_book: Optional[FastOrderBook] = None
        self._current_ts_ns: int = 0

    # ============================================
    # Регистрация
    # ============================================

    def register_strategy(self, strategy: StrategyProtocol) -> None:
        self._strategy = strategy

    # ============================================
    # Публичный API для стратегии
    # ============================================

    def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        reason: str = "",
    ) -> None:
        """
        Отправка MARKET-ордера.

        Немедленно симулируется fill по текущему стакану с применением
        latency. Реальный fill применяется в момент submit_ts + latency.
        """
        if self._result is None:
            return

        self._result.total_orders += 1

        book = self._result_book
        if book is None:
            return

        bids = book.top_bids(self.config.depth_for_matching)
        asks = book.top_asks(self.config.depth_for_matching)

        # В первой версии — если открыта позиция, не даём открывать вторую
        if self._position.is_open:
            return

        fill = self.matching.match_market(
            order_qty=quantity,
            bids=bids,
            asks=asks,
            is_buy=(side == OrderSide.BUY),
        )

        if fill.status in (MatchStatus.REJECTED, MatchStatus.SKIPPED):
            if fill.status == MatchStatus.REJECTED:
                self._result.rejected_orders += 1
            else:
                self._result.skipped_orders += 1
            return

        self._result.filled_orders += 1

        # Latency: сдвигаем момент применения
        breakdown = self.latency.sample()
        fill_ts_ns = self._current_ts_ns + ms_to_ns(breakdown.submit_to_fill_ms)
        self._schedule_fill(fill_ts_ns, fill, reason=reason, side=side)

    def submit_marketable_limit_ioc_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        reason: str = "",
    ) -> None:
        """IOC-ордер: ограничение цены."""
        if self._result is None:
            return

        self._result.total_orders += 1

        book = self._result_book
        if book is None:
            return

        if self._position.is_open:
            return

        bids = book.top_bids(self.config.depth_for_matching)
        asks = book.top_asks(self.config.depth_for_matching)

        fill = self.matching.match_marketable_limit_ioc(
            order_qty=quantity,
            limit_price=limit_price,
            bids=bids,
            asks=asks,
            is_buy=(side == OrderSide.BUY),
        )

        if fill.status in (MatchStatus.REJECTED, MatchStatus.SKIPPED):
            if fill.status == MatchStatus.REJECTED:
                self._result.rejected_orders += 1
            else:
                self._result.skipped_orders += 1
            return

        self._result.filled_orders += 1

        breakdown = self.latency.sample()
        fill_ts_ns = self._current_ts_ns + ms_to_ns(breakdown.submit_to_fill_ms)
        self._schedule_fill(fill_ts_ns, fill, reason=reason, side=side)

    def close_position(
        self,
        reason: str = "MANUAL",
    ) -> None:
        """Закрытие текущей позиции через MARKET."""
        if not self._position.is_open or self._result is None:
            return

        book = self._result_book
        if book is None:
            return

        # Закрывающая сторона — противоположная позиции
        if self._position.side == PositionSide.LONG:
            side = OrderSide.SELL
        else:
            side = OrderSide.BUY

        self._result.total_orders += 1

        bids = book.top_bids(self.config.depth_for_matching)
        asks = book.top_asks(self.config.depth_for_matching)

        fill = self.matching.match_market(
            order_qty=self._position.quantity,
            bids=bids,
            asks=asks,
            is_buy=(side == OrderSide.BUY),
        )

        if fill.status in (MatchStatus.REJECTED, MatchStatus.SKIPPED):
            if fill.status == MatchStatus.REJECTED:
                self._result.rejected_orders += 1
            else:
                self._result.skipped_orders += 1
            return

        self._result.filled_orders += 1

        breakdown = self.latency.sample()
        fill_ts_ns = self._current_ts_ns + ms_to_ns(breakdown.submit_to_fill_ms)
        self._schedule_fill(fill_ts_ns, fill, reason=reason, side=side)

    # ============================================
    # Прогон
    # ============================================

    def run(
        self,
        symbol: str,
        date_str: str,
    ) -> BacktestResult:
        """
        Прогоняет бэктест за день.

        Args:
            symbol: торговый инструмент
            date_str: строка даты в формате YYYY-MM-DD
        """
        symbol = symbol.upper()

        result = BacktestResult(
            symbol=symbol,
            initial_balance=self.config.initial_balance,
            final_balance=self.config.initial_balance,
        )
        self._result = result
        self._position = BacktestPosition(symbol=symbol)
        self._pending_fills.clear()
        self._fill_seq_counter = 0
        self._result_book = None
        self._current_ts_ns = 0

        # Готовим replay
        replay = ReplayBuilder(ReplayConfig(
            source_dir=self.config.source_dir,
            use_parquet=self.config.use_parquet,
        ))

        # FIX #2: replay_day() возвращает кортежи (book, delta).
        # book уже содержит применённую дельту — используем его напрямую,
        # вместо создания отдельного FastOrderBook здесь.
        deltas = replay.replay_day(
            symbol=symbol,
            date_str=date_str,
            tick_size=self.config.tick_size,
        )

        for book, delta in deltas:
            self._result_book = book
            self._current_ts_ns = delta.ts_exchange_ns

            # 1. Применяем отложенные fills, чьё время пришло
            self._apply_due_fills(self._current_ts_ns)

            # 2. Вызываем стратегию
            if self._strategy is not None:
                ctx = BacktestContext(
                    ts_ns=self._current_ts_ns,
                    symbol=symbol,
                    book=book,
                    position=self._position,
                )
                try:
                    self._strategy.on_market_update(ctx, self)
                except Exception:
                    # Ошибки стратегии не роняют бэктест
                    pass

            result.total_events += 1
            if result.start_ts_ns == 0:
                result.start_ts_ns = self._current_ts_ns
            result.end_ts_ns = self._current_ts_ns

        # Финальный flush — применяем все pending fills
        self._apply_due_fills(self._current_ts_ns or 0, force=True)

        # FIX #3: принудительное закрытие позиции по последней цене.
        # Используем self._result_book с проверкой None
        # (при пустых данных _result_book остаётся None).
        final_book = self._result_book
        if (
            self._position.is_open
            and final_book is not None
            and final_book.mid_price
        ):
            mid = final_book.mid_price
            exit_price = mid
            gross_pnl = self._compute_gross_pnl(
                side=self._position.side,
                entry=self._position.entry_price,
                exit_=exit_price,
                qty=self._position.quantity,
            )
            fees = self.costs.fee_model.taker_fee(
                exit_price * self._position.quantity
            )
            self._record_trade(
                exit_price=exit_price,
                exit_ts_ns=self._current_ts_ns or self._position.entry_ts_ns,
                exit_reason="FORCED_CLOSE",
                gross_pnl=gross_pnl,
                fees=fees,
            )

        result.final_balance = result.initial_balance + result.total_pnl
        return result

    # ============================================
    # Внутренние методы
    # ============================================

    def _schedule_fill(
        self,
        fill_ts_ns: int,
        fill: FillResult,
        reason: str,
        side: OrderSide,
    ) -> None:
        """Кладёт fill в очередь отложенных."""
        self._fill_seq_counter += 1
        heapq.heappush(
            self._pending_fills,
            (fill_ts_ns, self._fill_seq_counter, fill, reason, side),
        )

    def _apply_due_fills(self, now_ns: int, force: bool = False) -> None:
        """Применяет все fills, чьё время пришло."""
        while self._pending_fills:
            fill_ts_ns, _, fill, reason, side = self._pending_fills[0]
            if not force and fill_ts_ns > now_ns:
                break
            heapq.heappop(self._pending_fills)
            self._apply_fill(fill, reason, side, fill_ts_ns)

    def _apply_fill(
        self,
        fill: FillResult,
        reason: str,
        side: OrderSide,
        fill_ts_ns: int,
    ) -> None:
        """Применяет fill: открытие или закрытие позиции."""
        if not self._position.is_open:
            # Открытие новой позиции
            self._position = BacktestPosition(
                symbol=self._position.symbol,
                side=PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT,
                quantity=fill.filled_qty,
                entry_price=fill.avg_fill_price,
                entry_ts_ns=fill_ts_ns,
                entry_fees=fill.fees,
                entry_slippage_cost=fill.slippage_cost,
            )
            return

        # Иначе — закрытие
        gross_pnl = self._compute_gross_pnl(
            side=self._position.side,
            entry=self._position.entry_price,
            exit_=fill.avg_fill_price,
            qty=fill.filled_qty,
        )
        total_fees = fill.fees + self._position.entry_fees
        total_slippage = fill.slippage_cost + self._position.entry_slippage_cost

        self._record_trade(
            exit_price=fill.avg_fill_price,
            exit_ts_ns=fill_ts_ns,
            exit_reason=reason,
            gross_pnl=gross_pnl,
            fees=total_fees,
            slippage_cost=total_slippage,
        )

    def _compute_gross_pnl(
        self,
        side: PositionSide,
        entry: float,
        exit_: float,
        qty: float,
    ) -> float:
        if side == PositionSide.LONG:
            return (exit_ - entry) * qty
        elif side == PositionSide.SHORT:
            return (entry - exit_) * qty
        return 0.0

    def _record_trade(
        self,
        exit_price: float,
        exit_ts_ns: int,
        exit_reason: str,
        gross_pnl: float,
        fees: float,
        slippage_cost: Optional[float] = None,
    ) -> None:
        if self._result is None:
            return

        pos = self._position

        if slippage_cost is None:
            slippage_cost = pos.entry_slippage_cost

        net_pnl = gross_pnl - fees

        # R-multiple
        initial_risk_per_unit = (
            abs(pos.entry_price - pos.stop_price) if pos.stop_price else 0.0
        )
        r_multiple = 0.0
        if initial_risk_per_unit > 0:
            r_multiple = net_pnl / (initial_risk_per_unit * pos.quantity)

        trade = BacktestTrade(
            trade_id=f"{pos.symbol}-{exit_ts_ns}",
            symbol=pos.symbol,
            side=pos.side,
            entry_ts_ns=pos.entry_ts_ns,
            exit_ts_ns=exit_ts_ns,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            gross_pnl=gross_pnl,
            fees=fees,
            slippage_cost=slippage_cost,
            net_pnl=net_pnl,
            entry_reason=pos.signal_id or "ENTRY",
            exit_reason=exit_reason,
            duration_ms=int((exit_ts_ns - pos.entry_ts_ns) / 1_000_000),
            r_multiple=r_multiple,
        )
        self._result.trades.append(trade)

        # Сбрасываем позицию
        self._position = BacktestPosition(symbol=pos.symbol)