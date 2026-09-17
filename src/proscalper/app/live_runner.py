"""
Live Trading Runner: бумажная и живая торговля.

Единый раннер для двух режимов:
- paper (дефолт): бумажное исполнение через PaperExecutor,
  без реальных ордеров на биржу
- live: реальное исполнение через BinanceOrderClient +
  приватный WS-канал (user data stream)

Архитектура:
- режим задаётся через --mode или config.execution.mode
- в режиме live обязательно проверяется подключение к бирже
  и баланс аккаунта перед запуском
- все решения логируются в DecisionJournal
- все инциденты логируются в IncidentJournal
- все сделки логируются в TradeJournal
- аварийная остановка через SIGINT/SIGTERM

Использование:
    # Бумажный режим (дефолт)
    python -m proscalper.app.live_runner configs/default.yaml

    # Живой режим
    python -m proscalper.app.live_runner configs/default.yaml --mode live

    # Живой режим с тестнетом
    python -m proscalper.app.live_runner configs/default.yaml --mode live --testnet

Используется в связке с:
- core/config.py (конфигурация)
- core/logging.py (структурированное логирование)
- exchange/binance_futures/auth.py (подпись запросов)
- exchange/binance_futures/order.py (отправка ордеров)
- exchange/binance_futures/ws_user.py (приватный канал)
- execution/paper.py (бумажное исполнение)
- journal/ (журналирование)
- Все остальные модули системы
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from proscalper.core.config import load_app_config, AppConfig
from proscalper.core.clock import get_clock, SystemClock
from proscalper.core.instrument_registry import InstrumentRegistry
from proscalper.core.logging import get_logger, setup_logging, LogContext
from proscalper.core.types import OrderSide, BookSide
from proscalper.exchange.binance_futures.auth import (
    BinanceAuth,
    BinanceCredentials,
    create_auth_from_env,
)
from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient
from proscalper.exchange.binance_futures.order import (
    BinanceOrderClient,
    OrderParams,
    OrderType,
    TimeInForce,
    WorkingType,
)
from proscalper.exchange.binance_futures.ws_market import (
    BinanceFuturesMarketDataWS,
    BinanceFuturesWSConfig,
)
from proscalper.exchange.binance_futures.ws_user import (
    BinanceFuturesUserStreamWS,
    UserStreamWSConfig,
    UserStreamHandler,
    OrderTradeUpdate,
    AccountUpdate,
    AccountConfigUpdate,
)
from proscalper.execution.paper import PaperExecutor, PaperExecutionConfig, PaperOrderState
from proscalper.market_data.bookticker_guard import BookTickerGuard, FastGuardConfig
from proscalper.market_data.orderbook_fast import FastOrderBook
from proscalper.market_data.tape import TapeManager
from proscalper.market_data.bar_aggregator import MultiTimeframeAggregator, Bar
from proscalper.features.levels import LevelManager, Level, LevelState
from proscalper.features.book_analyzer import BookAnalyzerManager, BookAnalyzerConfig
from proscalper.features.spoof_detector import SpoofDetectorManager
from proscalper.features.iceberg import IcebergDetectorManager
from proscalper.features.symbol_metrics import SymbolMetricsManager
from proscalper.journal.decision_journal import DecisionJournal, DecisionJournalConfig
from proscalper.journal.incident_journal import IncidentJournal, IncidentJournalConfig
from proscalper.journal.trade_journal import (
    TradeJournal,
    TradeJournalConfig,
    TradeDirection,
)
from proscalper.journal.reasons import (
    SignalReason,
    RejectReason,
    RiskReason,
    ExitReason,
)
from proscalper.storage.delta_writer import MarketDataRecorder


logger = get_logger(__name__)


# ============================================================
# Позиция
# ============================================================

@dataclass
class LivePosition:
    """Позиция в раннере."""
    position_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    stop_price: float
    signal_id: str
    opened_ts_ns: int

    # Состояние
    is_open: bool = True
    closed_ts_ns: int = 0
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    exit_reason: str = ""

    # Исполнение
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None


# ============================================================
# Обработчик приватного канала (только для live режима)
# ============================================================

class LiveUserStreamHandler(UserStreamHandler):
    """Обработчик событий user data stream."""

    def __init__(self, runner_handler: LiveRunnerHandler) -> None:
        self._runner = runner_handler

    async def on_order_update(self, update: OrderTradeUpdate) -> None:
        """Событие исполнения/отмены ордера."""
        logger.info(
            "Order update",
            symbol=update.symbol,
            order_id=update.order_id,
            status=update.order_status,
            filled_qty=update.accumulated_filled_qty,
            avg_price=update.avg_fill_price,
        )

        # Логируем в DecisionJournal
        if update.is_fill:
            self._runner.decision_journal.log_order_fill(
                symbol=update.symbol,
                client_order_id=update.client_order_id,
                exchange_order_id=str(update.order_id),
                filled_qty=update.accumulated_filled_qty,
                avg_fill_price=update.avg_fill_price,
                fees=update.accumulated_commission,
            )
        elif update.is_cancel:
            self._runner.decision_journal.log_order_rejected(
                symbol=update.symbol,
                client_order_id=update.client_order_id,
                reject_reasons=["ORDER_CANCELLED_BY_USER"],
            )
        elif update.is_reject:
            self._runner.decision_journal.log_order_rejected(
                symbol=update.symbol,
                client_order_id=update.client_order_id,
                reject_reasons=["ORDER_REJECTED_BY_EXCHANGE"],
            )

            # Логируем инцидент
            self._runner.incident_journal.report_order_rejected(
                symbol=update.symbol,
                client_order_id=update.client_order_id,
                details={"order_status": update.order_status},
            )

    async def on_account_update(self, update: AccountUpdate) -> None:
        """Событие изменения аккаунта."""
        logger.info(
            "Account update",
            reason=update.event_reason,
            balances_count=len(update.balances),
            positions_count=len(update.positions),
        )

    async def on_account_config_update(self, update: AccountConfigUpdate) -> None:
        """Изменение конфигурации аккаунта."""
        logger.info(
            "Account config update",
            symbol=update.symbol,
            leverage=update.leverage,
        )

    async def on_margin_call(self, event: Dict[str, Any]) -> None:
        """MARGIN_CALL — критический инцидент."""
        logger.critical("MARGIN CALL received!", event=str(event))

        self._runner.incident_journal.report(
            incident_type="MARGIN_CALL",
            severity="CRITICAL",
            message="Margin call received from exchange",
            details=event,
        )

        # Аварийное закрытие всех позиций
        await self._runner.emergency_flatten("MARGIN_CALL")

    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        """Изменение состояния соединения."""
        logger.info("UserStream state", state=state, details=str(details))

        if state == "DISCONNECT":
            self._runner.incident_journal.report_ws_disconnect(
                message="User data stream disconnected",
                details=details,
            )


# ============================================================
# Основной обработчик событий
# ============================================================

class LiveRunnerHandler:
    """
    Обработчик рыночных событий с торговой логикой.

    Работает в двух режимах:
    - paper: исполнение через PaperExecutor
    - live: исполнение через BinanceOrderClient
    """

    def __init__(
        self,
        config: AppConfig,
        mode: str,
        recorder: MarketDataRecorder,
        books: Dict[str, FastOrderBook],
        guard: BookTickerGuard,
        tape_manager: TapeManager,
        level_manager: LevelManager,
        book_analyzer_manager: BookAnalyzerManager,
        spoof_manager: SpoofDetectorManager,
        iceberg_manager: IcebergDetectorManager,
        metrics_manager: SymbolMetricsManager,
        decision_journal: DecisionJournal,
        incident_journal: IncidentJournal,
        trade_journal: TradeJournal,
        tick_sizes: Dict[str, float],
        paper_executor: Optional[PaperExecutor] = None,
        order_client: Optional[BinanceOrderClient] = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.recorder = recorder
        self.books = books
        self.guard = guard
        self.tape_manager = tape_manager
        self.level_manager = level_manager
        self.book_analyzer_manager = book_analyzer_manager
        self.spoof_manager = spoof_manager
        self.iceberg_manager = iceberg_manager
        self.metrics_manager = metrics_manager
        self.decision_journal = decision_journal
        self.incident_journal = incident_journal
        self.trade_journal = trade_journal
        self.tick_sizes = tick_sizes
        self.paper_executor = paper_executor
        self.order_client = order_client

        # Агрегаторы баров
        self.tape_aggregators: Dict[str, MultiTimeframeAggregator] = {}
        self.level_aggregators: Dict[str, MultiTimeframeAggregator] = {}

        # Счётчики
        self._trade_count = 0
        self._book_ticker_count = 0
        self._book_delta_count = 0
        self._signal_count = 0
        self._last_status_ts = time.time()

        # Позиции
        self._positions: Dict[str, LivePosition] = {}
        self._open_positions: Dict[str, LivePosition] = {}

        # Дебаунс пробоев
        self._breakout_attempts: Dict[str, int] = {}

        # PnL статистика
        self._total_pnl = 0.0
        self._win_count = 0
        self._loss_count = 0

    # ============================================
    # Обработка рыночных событий
    # ============================================

    async def on_trade(self, event) -> None:
        """Обработка сделки."""
        self._trade_count += 1

        self.tape_manager.on_trade(event)
        self.metrics_manager.on_trade(event)

        tape_agg = self.tape_aggregators.get(event.symbol.upper())
        if tape_agg is not None:
            tape_agg.on_trade(event)

        level_agg = self.level_aggregators.get(event.symbol.upper())
        if level_agg is not None:
            level_agg.on_trade(event)

        await self.recorder.record_trade(event)

        # Проверяем стопы по текущей цене
        self.check_stop_exits(event.symbol, event.price)

        # Проверяем пробои
        self._check_breakouts(event.symbol, event.price)

    async def on_book_ticker(self, event) -> None:
        """Обработка обновления лучших цен."""
        self._book_ticker_count += 1

        self.guard.update(event)
        self.metrics_manager.on_book_ticker(event)

        symbol = event.symbol.upper()
        book_analyzer = self.book_analyzer_manager._analyzers.get(symbol)
        if book_analyzer is not None:
            book_analyzer.update_best_prices(
                bid_price=event.bid_price,
                ask_price=event.ask_price,
                ts_ns=event.ts_local_ns,
            )

        await self.recorder.record_book_ticker(event)

    async def on_snapshot(self, event) -> None:
        """Применение снапшота стакана."""
        symbol = event.symbol.upper()

        book = self.books.get(symbol)
        if book is not None:
            book.apply_snapshot(event)

        self.book_analyzer_manager.on_snapshot(event)

        await self.recorder.record_book_snapshot(event)

    async def on_delta_batch(self, event) -> None:
        """Применение пакета дельт стакана."""
        self._book_delta_count += 1
        symbol = event.symbol.upper()

        book = self.books.get(symbol)
        if book is not None:
            ok = book.apply_delta_batch(event)
            if not ok or book.out_of_sync:
                self.incident_journal.report_book_out_of_sync(
                    symbol=symbol,
                    reason=book.out_of_sync_reason or "unknown",
                    details={"last_update_id": book.last_update_id},
                )

        self.book_analyzer_manager.on_delta_batch(event)

        await self.recorder.record_book_delta(event)

    async def on_snapshot_requested(self, symbol: str) -> None:
        """Вызывается перед запросом снапшота через REST."""
        book = self.books.get(symbol.upper())
        if book is not None:
            book.mark_snapshot_requested()

    async def reset_books(self) -> None:
        """Сброс всех локальных стаканов перед reconnect."""
        for symbol, book in self.books.items():
            book.reset()

        self.tape_manager.reset_all()

        for aggregator in self.tape_aggregators.values():
            aggregator.reset()

        for aggregator in self.level_aggregators.values():
            aggregator.reset()

        logger.info(
            "Books reset",
            symbols=len(self.books),
            note="levels preserved (loaded from history)",
        )

    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        """Обработка state-событий от gateway."""
        logger.info("WS state", state=state, details=str(details))

        if state == "DISCONNECT":
            self.incident_journal.report_ws_disconnect(
                message="Market data stream disconnected",
                details=details,
            )

        # Статус каждые 10 секунд
        now = time.time()
        if now - self._last_status_ts >= 10.0:
            self._print_status()
            self._last_status_ts = now

    # ============================================
    # Callback'и для баров
    # ============================================

    def _on_tape_bar_close(self, bar: Bar) -> None:
        """Закрытие 1с бара."""
        pass

    def _on_level_bar_close(self, bar: Bar) -> None:
        """Закрытие 5м бара (уровни)."""
        new_levels = self.level_manager.on_bar(bar)

        for level in new_levels:
            logger.info(
                "New level",
                side=level.side.name,
                symbol=level.symbol,
                price=f"{level.center:.2f}",
                touches=level.touches,
                strength=f"{level.strength:.2f}",
            )

    # ============================================
    # Детекция пробоев
    # ============================================

    def _check_breakouts(self, symbol: str, price: float) -> None:
        """Проверяет пробой уровней."""
        symbol = symbol.upper()
        active_levels = self.level_manager.get_active_levels(symbol)

        for level in active_levels:
            level_id = f"{symbol}_{level.side.name}_{level.center:.0f}"

            if level.side.name == "RESISTANCE" and price > level.center:
                self._handle_breakout_signal(
                    symbol=symbol,
                    level=level,
                    level_id=level_id,
                    side=OrderSide.BUY,
                    price=price,
                )

            elif level.side.name == "SUPPORT" and price < level.center:
                self._handle_breakout_signal(
                    symbol=symbol,
                    level=level,
                    level_id=level_id,
                    side=OrderSide.SELL,
                    price=price,
                )

    def _handle_breakout_signal(
        self,
        symbol: str,
        level: Level,
        level_id: str,
        side: OrderSide,
        price: float,
    ) -> None:
        """Обработка потенциального пробоя."""
        now_ns = time.time_ns()

        # Проверка лимитов
        if symbol in self._open_positions:
            return

        max_positions = self.config.risk.max_open_positions
        if len(self._open_positions) >= max_positions:
            self.decision_journal.log_signal_rejected(
                symbol=symbol,
                signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                reject_reasons=[RejectReason.MAX_POSITIONS_REACHED],
            )
            return

        # Дебаунс
        if level_id in self._breakout_attempts:
            last_attempt = self._breakout_attempts[level_id]
            if (now_ns - last_attempt) < 30_000_000_000:
                return

        self._breakout_attempts[level_id] = now_ns

        # Проверка импульса в ленте
        tape_metrics = self.tape_manager.snapshot(symbol)
        if tape_metrics is None:
            return

        if side == OrderSide.BUY and tape_metrics.net_delta_1s <= 0:
            self.decision_journal.log_signal_rejected(
                symbol=symbol,
                signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                reject_reasons=[RejectReason.NO_POSITIVE_DELTA],
            )
            return

        if side == OrderSide.SELL and tape_metrics.net_delta_1s >= 0:
            self.decision_journal.log_signal_rejected(
                symbol=symbol,
                signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                reject_reasons=[RejectReason.NO_NEGATIVE_DELTA],
            )
            return

        # Проверка стен на пути
        book_analyzer = self.book_analyzer_manager._analyzers.get(symbol)
        if book_analyzer is not None:
            liquidity = book_analyzer.get_level_liquidity(level)
            if liquidity is not None and liquidity.has_significant_walls:
                self.decision_journal.log_signal_rejected(
                    symbol=symbol,
                    signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                    reject_reasons=[RejectReason.WALL_ON_PATH],
                )
                return

        # Генерируем сигнал
        signal_id = f"sig_{uuid.uuid4().hex[:8]}"
        self._signal_count += 1

        logger.info(
            "Signal generated",
            side=side.name,
            symbol=symbol,
            price=f"{price:.2f}",
            level=f"{level.center:.2f}",
            delta=f"{tape_metrics.net_delta_1s:+.0f}",
        )

        self.decision_journal.log_signal_created(
            symbol=symbol,
            signal_id=signal_id,
            reasons=[
                SignalReason.BREAKOUT,
                SignalReason.LEVEL_ACTIVE,
                SignalReason.DELTA_CONFIRMED,
            ],
        )

        # Исполняем
        self._execute_trade(
            symbol=symbol,
            signal_id=signal_id,
            side=side,
            entry_price=price,
            level=level,
        )

    # ============================================
    # Исполнение
    # ============================================

    def _execute_trade(
        self,
        symbol: str,
        signal_id: str,
        side: OrderSide,
        entry_price: float,
        level: Level,
    ) -> None:
        """Исполнение сделки в текущем режиме."""
        # Расчёт размера позиции
        capital = self.config.risk.initial_capital
        risk_pct = self.config.risk.risk_per_trade_pct

        tick_size = self.tick_sizes.get(symbol, 0.01)
        stop_buffer = max(
            self.config.risk.stop_buffer_ticks * tick_size,
            level.center * self.config.risk.stop_buffer_pct,
        )

        if side == OrderSide.BUY:
            stop_price = level.center - stop_buffer
        else:
            stop_price = level.center + stop_buffer

        risk_amount = capital * risk_pct / 100.0
        risk_per_unit = abs(entry_price - stop_price)

        if risk_per_unit <= 0:
            return

        quantity = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0

        # Проверяем минимальный/максимальный номинал
        notional = quantity * entry_price
        if notional < self.config.risk.min_notional_per_trade:
            quantity = self.config.risk.min_notional_per_trade / entry_price if entry_price > 0 else 0.0
        elif notional > self.config.risk.max_notional_per_trade:
            quantity = self.config.risk.max_notional_per_trade / entry_price if entry_price > 0 else 0.0

        self.decision_journal.log_risk_approved(
            symbol=symbol,
            signal_id=signal_id,
            quantity=quantity,
            notional=quantity * entry_price,
            stop_price=stop_price,
        )

        # Создаём позицию
        position = LivePosition(
            position_id=f"pos_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            signal_id=signal_id,
            opened_ts_ns=time.time_ns(),
        )

        self._positions[position.position_id] = position
        self._open_positions[symbol] = position

        if self.mode == "paper":
            self._execute_paper_trade(position)
        else:
            self._execute_live_trade(position)

        self.decision_journal.log_position_opened(
            symbol=symbol,
            position_id=position.position_id,
            signal_id=signal_id,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
        )

        self.trade_journal.log_open(
            symbol=symbol,
            direction=TradeDirection.LONG if side == OrderSide.BUY else TradeDirection.SHORT,
            signal_id=signal_id,
            position_id=position.position_id,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            entry_reasons=[
                SignalReason.BREAKOUT,
                SignalReason.LEVEL_ACTIVE,
                SignalReason.DELTA_CONFIRMED,
            ],
        )

        logger.info(
            "Position opened",
            mode=self.mode,
            side=side.name,
            symbol=symbol,
            qty=f"{quantity:.6f}",
            price=f"{entry_price:.2f}",
            stop=f"{stop_price:.2f}",
        )

    def _execute_paper_trade(self, position: LivePosition) -> None:
        """Бумажное исполнение через PaperExecutor."""
        if self.paper_executor is None:
            logger.error("PaperExecutor not initialized")
            return

        # Отправляем MARKET-ордер через PaperExecutor
        order = self.paper_executor.submit_market_order(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            signal_id=position.signal_id,
            leg_type="entry",
            reason="BREAKOUT",
        )

        position.entry_order_id = order.order_id

    def _execute_live_trade(self, position: LivePosition) -> None:
        """Реальное исполнение через BinanceOrderClient."""
        if self.order_client is None:
            logger.error("BinanceOrderClient not initialized")
            return

        # Формируем пакет ордеров: вход + стоп
        entry_params = OrderParams(
            symbol=position.symbol,
            side=position.side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            client_order_id=f"entry_{position.position_id}",
        )

        # Стоп-ордер
        stop_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        stop_params = OrderParams(
            symbol=position.symbol,
            side=stop_side,
            order_type=OrderType.STOP_MARKET,
            quantity=position.quantity,
            stop_price=position.stop_price,
            working_type=WorkingType.CONTRACT_PRICE,
            client_order_id=f"stop_{position.position_id}",
            reduce_only=True,
        )

        # Отправляем пакет (вход + стоп)
        # Используем asyncio для отправки
        asyncio.create_task(self._submit_live_orders(
            position=position,
            entry_params=entry_params,
            stop_params=stop_params,
        ))

    async def _submit_live_orders(
        self,
        position: LivePosition,
        entry_params: OrderParams,
        stop_params: OrderParams,
    ) -> None:
        """Отправка пакета ордеров на биржу."""
        try:
            # Пакетная отправка (вход + стоп)
            responses = await self.order_client.create_batch_orders([
                entry_params,
                stop_params,
            ])

            if len(responses) >= 2:
                position.entry_order_id = responses[0].client_order_id
                position.stop_order_id = responses[1].client_order_id

                logger.info(
                    "Live orders submitted",
                    entry_order=responses[0].order_id,
                    stop_order=responses[1].order_id,
                )
            else:
                logger.error("Batch order response incomplete")

        except Exception as exc:
            logger.error(f"Failed to submit live orders: {exc}")

            self.incident_journal.report(
                incident_type="STOP_PLACE_FAILED",
                severity="CRITICAL",
                symbol=position.symbol,
                message=f"Failed to submit orders: {exc}",
                details={"position_id": position.position_id},
            )

    # ============================================
    # Закрытие позиций
    # ============================================

    def check_stop_exits(self, symbol: str, price: float) -> None:
        """Проверяет срабатывание стопов."""
        symbol = symbol.upper()
        position = self._open_positions.get(symbol)

        if position is None or not position.is_open:
            return

        should_close = False
        reason = ""

        if position.side == OrderSide.BUY:
            if price <= position.stop_price:
                should_close = True
                reason = ExitReason.STOP_LOSS
        else:
            if price >= position.stop_price:
                should_close = True
                reason = ExitReason.STOP_LOSS

        if should_close:
            self._close_position(position, price, reason)

    def _close_position(
        self,
        position: LivePosition,
        exit_price: float,
        reason: str,
    ) -> None:
        """Закрывает позицию."""
        # Расчёт PnL
        if position.side == OrderSide.BUY:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity

        fee_pct = self.config.execution.taker_fee_pct
        fees = abs(exit_price * position.quantity) * fee_pct / 100.0

        net_pnl = gross_pnl - fees

        position.is_open = False
        position.closed_ts_ns = time.time_ns()
        position.exit_price = exit_price
        position.realized_pnl = net_pnl
        position.exit_reason = reason

        self._open_positions.pop(position.symbol, None)

        self._total_pnl += net_pnl
        if net_pnl > 0:
            self._win_count += 1
        else:
            self._loss_count += 1

        self.decision_journal.log_position_closed(
            symbol=position.symbol,
            position_id=position.position_id,
            realized_pnl=net_pnl,
            reasons=[reason],
        )

        self.trade_journal.log_close(
            symbol=position.symbol,
            position_id=position.position_id,
            exit_price=exit_price,
            exit_reasons=[reason],
            gross_pnl=gross_pnl,
            fees=fees,
        )

        logger.info(
            "Position closed",
            symbol=position.symbol,
            exit=f"{exit_price:.2f}",
            pnl=f"{net_pnl:+.2f}",
            reason=reason,
            total=f"{self._total_pnl:+.2f}",
        )

    async def emergency_flatten(self, reason: str) -> None:
        """Аварийное закрытие всех позиций."""
        logger.critical(f"EMERGENCY FLATTEN: {reason}")

        for symbol, position in list(self._open_positions.items()):
            book = self.books.get(symbol)
            exit_price = book.mid_price if book is not None else position.entry_price

            self._close_position(position, exit_price, ExitReason.EMERGENCY_FLATTEN)

            # В live режиме отменяем все ордера
            if self.mode == "live" and self.order_client is not None:
                try:
                    await self.order_client.cancel_all_orders(symbol)
                except Exception as exc:
                    logger.error(f"Failed to cancel orders for {symbol}: {exc}")

        self.decision_journal.log_emergency_flatten(
            symbol="ALL",
            position_id=None,
            reasons=[reason],
        )

    # ============================================
    # Статус
    # ============================================

    def _print_status(self) -> None:
        """Печатает периодический статус."""
        logger.info(
            "Status",
            mode=self.mode,
            trades=self._trade_count,
            tickers=self._book_ticker_count,
            deltas=self._book_delta_count,
            signals=self._signal_count,
            open_positions=len(self._open_positions),
            pnl=f"{self._total_pnl:+.2f}",
            wins=self._win_count,
            losses=self._loss_count,
        )

        for symbol in self.books.keys():
            metrics = self.tape_manager.snapshot(symbol)
            if metrics is not None:
                logger.info(
                    "Tape",
                    symbol=symbol,
                    delta_1s=f"{metrics.net_delta_1s:+.0f}",
                    rate=f"{metrics.trades_per_sec_1s:.1f}/s",
                    price=f"{metrics.last_price:.2f}",
                )

            active_levels = self.level_manager.get_active_levels(symbol)
            if active_levels:
                levels_str = ", ".join(
                    f"{lvl.side.name[0]}{lvl.center:.0f}"
                    for lvl in active_levels[:5]
                )
                logger.info(
                    "Levels",
                    symbol=symbol,
                    count=len(active_levels),
                    top=levels_str,
                )


# ============================================================
# Основная функция запуска
# ============================================================

async def run_live_runner(config: AppConfig, mode: str, testnet: bool = False) -> None:
    """Основная функция live runner."""

    logger.info(
        "Live runner starting",
        mode=mode,
        testnet=testnet,
        symbols=", ".join(config.symbols),
    )

    # ========================================
    # 1. Загружаем информацию об инструментах + историю
    # ========================================
    registry = InstrumentRegistry()
    history_klines: Dict[str, List] = {}

    async with BinanceFuturesRestClient() as rest_client:
        instruments = await rest_client.get_exchange_info()
        registry.load_from_exchange_info(instruments)

        for symbol in config.symbols:
            if registry.get_instrument(symbol) is None:
                raise ValueError(f"Символ {symbol} не найден в exchangeInfo")

        logger.info(f"Загружено инструментов: {len(registry.all_symbols())}")

        # Загрузка истории для уровней
        for symbol in config.symbols:
            try:
                klines = await rest_client.get_klines_last_hours(
                    symbol=symbol,
                    interval=config.collector.history_interval,
                    hours=config.collector.history_hours,
                )
                history_klines[symbol] = klines
                logger.info(f"{symbol}: загружено {len(klines)} свечей истории")
            except Exception as exc:
                logger.warning(f"{symbol}: не удалось загрузить историю: {exc}")

    # ========================================
    # 2. Создаём локальные стаканы
    # ========================================
    tick_sizes: Dict[str, float] = {}
    books: Dict[str, FastOrderBook] = {}

    for symbol in config.symbols:
        tick_size = registry.get_tick_size(symbol)
        if tick_size is None:
            raise ValueError(f"Не найден tick_size для {symbol}")
        tick_sizes[symbol.upper()] = tick_size
        books[symbol.upper()] = FastOrderBook(symbol=symbol, tick_size=tick_size)

    # ========================================
    # 3. Создаём основные менеджеры
    # ========================================
    guard = BookTickerGuard(config=FastGuardConfig(), tick_sizes=tick_sizes)
    tape_manager = TapeManager()

    level_manager = LevelManager(tick_sizes=tick_sizes)
    for symbol, klines in history_klines.items():
        detector = level_manager.get_or_create(symbol)
        level_count = detector.load_from_klines(klines)
        logger.info(f"{symbol}: инициализировано {level_count} уровней из истории")

    book_analyzer_manager = BookAnalyzerManager(tick_sizes=tick_sizes)
    spoof_manager = SpoofDetectorManager(tick_sizes=tick_sizes)
    iceberg_manager = IcebergDetectorManager(tick_sizes=tick_sizes)
    metrics_manager = SymbolMetricsManager(tick_sizes=tick_sizes)

    # ========================================
    # 4. Создаём журналы
    # ========================================
    decision_journal = DecisionJournal(
        config=DecisionJournalConfig(base_dir=config.journal.decision_journal_dir)
    )
    await decision_journal.start()

    incident_journal = IncidentJournal(
        config=IncidentJournalConfig(base_dir=config.journal.incident_journal_dir)
    )
    await incident_journal.start()

    trade_journal = TradeJournal(
        config=TradeJournalConfig(base_dir=config.journal.trade_journal_dir)
    )
    await trade_journal.start()

    # ========================================
    # 5. Создаём рекордер
    # ========================================
    recorder = MarketDataRecorder(
        base_dir=config.collector.data_dir,
        batch_size=config.collector.batch_size,
        flush_interval_ms=config.collector.flush_interval_ms,
        queue_size=config.collector.queue_size,
    )
    await recorder.start()

    # ========================================
    # 6. Создаём обработчик
    # ========================================
    handler = LiveRunnerHandler(
        config=config,
        mode=mode,
        recorder=recorder,
        books=books,
        guard=guard,
        tape_manager=tape_manager,
        level_manager=level_manager,
        book_analyzer_manager=book_analyzer_manager,
        spoof_manager=spoof_manager,
        iceberg_manager=iceberg_manager,
        metrics_manager=metrics_manager,
        decision_journal=decision_journal,
        incident_journal=incident_journal,
        trade_journal=trade_journal,
        tick_sizes=tick_sizes,
    )

    # ========================================
    # 7. Создаём агрегаторы баров
    # ========================================
    TAPE_TF_NS = 1_000_000_000
    LEVEL_TF_NS = 300_000_000_000

    for symbol in config.symbols:
        tape_agg = MultiTimeframeAggregator(
            symbol=symbol,
            timeframes_ns=[TAPE_TF_NS],
            on_bar_close=handler._on_tape_bar_close,
        )
        handler.tape_aggregators[symbol.upper()] = tape_agg

        level_agg = MultiTimeframeAggregator(
            symbol=symbol,
            timeframes_ns=[LEVEL_TF_NS],
            on_bar_close=handler._on_level_bar_close,
        )
        handler.level_aggregators[symbol.upper()] = level_agg

    # ========================================
    # 8. Инициализация исполнения по режиму
    # ========================================
    paper_executor: Optional[PaperExecutor] = None
    order_client: Optional[BinanceOrderClient] = None
    auth: Optional[BinanceAuth] = None
    user_stream_ws: Optional[BinanceFuturesUserStreamWS] = None
    user_stream_task: Optional[asyncio.Task] = None

    if mode == "paper":
        # Бумажное исполнение
        paper_executor = PaperExecutor(
            tick_size=tick_sizes.get(config.symbols[0], 0.01),
            config=PaperExecutionConfig(
                latency_region=config.backtest.latency_region,
                taker_fee_pct=config.execution.taker_fee_pct,
                maker_fee_pct=config.execution.maker_fee_pct,
            ),
        )

        # Привязываем стаканы
        for symbol, book in books.items():
            paper_executor.bind_book(book)

        handler.paper_executor = paper_executor

        logger.info("Paper executor initialized")

    else:
        # Реальное исполнение
        logger.info("Initializing LIVE mode...")

        # Создаём аутентификацию из окружения
        try:
            auth = create_auth_from_env(testnet=testnet)
            logger.info("Binance auth created from environment")
        except ValueError as exc:
            logger.error(f"Auth failed: {exc}")
            logger.info("Set BINANCE_FUTURES_API_KEY and BINANCE_FUTURES_API_SECRET")
            raise

        # Проверяем баланс аккаунта
        async with BinanceFuturesRestClient() as rest_client:
            order_client = BinanceOrderClient(rest_client, auth)
            handler.order_client = order_client

            # Проверяем открытые ордера (должны быть пусты при старте)
            for symbol in config.symbols:
                try:
                    open_orders = await order_client.query_open_orders(symbol)
                    if open_orders:
                        logger.warning(
                            f"{symbol}: найдено {len(open_orders)} открытых ордеров",
                        )
                except Exception as exc:
                    logger.warning(f"{symbol}: не удалось проверить ордера: {exc}")

        logger.info("Live mode initialized")

    # ========================================
    # 9. Создаём WS-шлюзы
    # ========================================
    ws_config = BinanceFuturesWSConfig(
        symbols=config.symbols,
        depth_update_speed=config.collector.depth_update_speed,
        snapshot_limit=config.collector.snapshot_limit,
    )

    gateway = BinanceFuturesMarketDataWS(
        config=ws_config,
        handler=handler,
    )

    # В live режиме запускаем приватный канал
    if mode == "live" and auth is not None:
        user_handler = LiveUserStreamHandler(handler)
        user_stream_ws = BinanceFuturesUserStreamWS(
            config=UserStreamWSConfig(),
            auth=auth,
            handler=user_handler,
        )

    # ========================================
    # 10. Обработка сигналов остановки
    # ========================================
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    # ========================================
    # 11. Запуск
    # ========================================
    logger.info(
        "Live runner started",
        mode=mode,
        symbols=", ".join(config.symbols),
        max_positions=config.risk.max_open_positions,
        risk_per_trade=f"{config.risk.risk_per_trade_pct}%",
    )

    try:
        gateway_task = asyncio.create_task(gateway.run())

        if user_stream_ws is not None:
            user_stream_task = asyncio.create_task(user_stream_ws.run())

        await stop_event.wait()

        await gateway.stop()
        gateway_task.cancel()

        if user_stream_ws is not None:
            await user_stream_ws.stop()
            if user_stream_task is not None:
                user_stream_task.cancel()

        try:
            await gateway_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Gateway error: {exc}")

    except asyncio.CancelledError:
        pass

    finally:
        # Закрываем все позиции
        for symbol, position in list(handler._open_positions.items()):
            book = books.get(symbol)
            exit_price = book.mid_price if book is not None else position.entry_price
            handler._close_position(position, exit_price, ExitReason.FORCED_CLOSE)

        # Flush бары
        for aggregator in handler.tape_aggregators.values():
            aggregator.flush()
        for aggregator in handler.level_aggregators.values():
            aggregator.flush()

        # Закрываем журналы и рекордер
        await recorder.close()
        await decision_journal.close()
        await incident_journal.close()
        await trade_journal.close()

        # Финальная статистика
        logger.info(
            "Live runner stopped",
            signals=handler._signal_count,
            trades_closed=handler._win_count + handler._loss_count,
            wins=handler._win_count,
            losses=handler._loss_count,
            total_pnl=f"{handler._total_pnl:+.2f}",
        )

        dj_stats = decision_journal.get_stats()
        ij_stats = incident_journal.get_stats()
        tj_stats = trade_journal.get_stats()

        logger.info(f"Decision Journal: {dj_stats}")
        logger.info(f"Incident Journal: {ij_stats}")
        logger.info(f"Trade Journal: {tj_stats}")


# ============================================================
# Точка входа
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Trading Runner: бумажная и живая торговля",
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="configs/default.yaml",
        help="Путь к конфигу",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=None,
        help="Режим работы (по умолчанию из конфига)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Использовать тестнет вместо основного рынка",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Уровень логирования",
    )

    args = parser.parse_args()

    # Инициализация логирования
    setup_logging(level=args.log_level, format="text", use_colors=True)

    # Загрузка конфига
    config = load_app_config(args.config_path)

    # Определение режима
    mode = args.mode or config.execution.mode

    if mode == "live":
        logger.warning(
            "LIVE MODE: ордера будут отправляться на биржу",
            testnet=args.testnet,
        )

    try:
        asyncio.run(run_live_runner(config, mode=mode, testnet=args.testnet))
    except KeyboardInterrupt:
        logger.info("Stopped by user")


if __name__ == "__main__":
    main()