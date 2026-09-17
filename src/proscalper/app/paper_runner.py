"""
Paper Trading Runner: бумажная торговля на живых данных.

Связывает все модули системы в единый поток:
- Рыночные данные (стакан, лента, бары, уровни)
- Анализ стакана (стены, дисбаланс, спуфинг, айсберги)
- Сигнальный слой (пробой уровней)
- Риск-менеджмент (лимиты, сайзинг, стопы)
- Бумажное исполнение (без реальных ордеров)
- Журналирование всех решений

Использование:
    python -m proscalper.app.paper_runner configs/default.yaml

Отличие от collector.py:
- collector: только запись данных на диск
- paper_runner: запись + анализ + сигналы + исполнение + журналы

Использование в связке с:
- core/config.py (конфигурация)
- features/book_analyzer.py (стены, дисбаланс)
- features/spoof_detector.py (фильтрация спуфинга)
- features/iceberg.py (детекция айсбергов)
- features/symbol_metrics.py (метрики инструмента)
- journal/decision_journal.py (журнал решений)
- journal/incident_journal.py (журнал инцидентов)
"""
from __future__ import annotations

import asyncio
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from proscalper.core.config import load_app_config, AppConfig
from proscalper.core.instrument_registry import InstrumentRegistry
from proscalper.core.types import BookSide, OrderSide
from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient
from proscalper.exchange.binance_futures.ws_market import (
    BinanceFuturesMarketDataWS,
    BinanceFuturesWSConfig,
)
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
from proscalper.storage.delta_writer import MarketDataRecorder


# ============================================================
# Позиция в бумажной торговле
# ============================================================

@dataclass
class PaperPosition:
    """Позиция в бумажной торговле."""
    position_id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    stop_price: float
    signal_id: str
    opened_ts_ns: int
    
    # Состояние
    is_open: bool = True
    closed_ts_ns: int = 0
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    exit_reason: str = ""
    
    @property
    def age_ms(self) -> int:
        if self.opened_ts_ns == 0:
            return 0
        now_ns = time.time_ns()
        end_ns = self.closed_ts_ns if not self.is_open else now_ns
        return int((end_ns - self.opened_ts_ns) / 1_000_000)


# ============================================================
# Обработчик событий с торговой логикой
# ============================================================

class PaperRunnerHandler:
    """
    Обработчик рыночных событий с бумажной торговлей.
    
    Расширяет логику коллекционера:
    - Обновляет стаканы, ленту, бары (как коллекционер)
    - Запускает анализ стакана (стены, дисбаланс)
    - Детектирует спуфинг и айсберги
    - Отслеживает уровни и детектирует пробои
    - Симулирует исполнение сигналов
    """
    
    def __init__(
        self,
        config: AppConfig,
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
        tick_sizes: Dict[str, float],
    ) -> None:
        self.config = config
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
        self.tick_sizes = tick_sizes
        
        # Агрегаторы баров (1с для ленты, 5м для уровней)
        self.tape_aggregators: Dict[str, MultiTimeframeAggregator] = {}
        self.level_aggregators: Dict[str, MultiTimeframeAggregator] = {}
        
        # Счётчики
        self._trade_count = 0
        self._book_ticker_count = 0
        self._book_delta_count = 0
        self._tape_bar_count = 0
        self._level_bar_count = 0
        self._signal_count = 0
        self._last_status_ts = time.time()
        
        # Позиции
        self._positions: Dict[str, PaperPosition] = {}
        self._open_positions: Dict[str, PaperPosition] = {}  # symbol -> position
        
        # Трекинг пробоев: (symbol, level_id) -> ts первого пересечения
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
        
        # Лента
        self.tape_manager.on_trade(event)
        
        # Метрики инструмента
        self.metrics_manager.on_trade(event)
        
        # Агрегаторы баров
        tape_agg = self.tape_aggregators.get(event.symbol.upper())
        if tape_agg is not None:
            tape_agg.on_trade(event)
        
        level_agg = self.level_aggregators.get(event.symbol.upper())
        if level_agg is not None:
            level_agg.on_trade(event)
        
        # Запись на диск
        await self.recorder.record_trade(event)
        
        # Проверка пробоев после каждой сделки
        self._check_breakouts(event.symbol, event.price)
    
    async def on_book_ticker(self, event) -> None:
        """Обработка обновления лучших цен."""
        self._book_ticker_count += 1
        
        # Guard для валидации
        self.guard.update(event)
        
        # Метрики инструмента
        self.metrics_manager.on_book_ticker(event)
        
        # Обновление лучших цен в анализаторе стакана
        symbol = event.symbol.upper()
        book_analyzer = self.book_analyzer_manager._analyzers.get(symbol)
        if book_analyzer is not None:
            book_analyzer.update_best_prices(
                bid_price=event.bid_price,
                ask_price=event.ask_price,
                ts_ns=event.ts_local_ns,
            )
        
        # Запись на диск
        await self.recorder.record_book_ticker(event)
    
    async def on_snapshot(self, event) -> None:
        """Применение снапшота стакана."""
        symbol = event.symbol.upper()
        
        # Обновляем локальный стакан
        book = self.books.get(symbol)
        if book is not None:
            book.apply_snapshot(event)
        
        # Обновляем анализатор стакана
        self.book_analyzer_manager.on_snapshot(event)
        
        # Запись на диск
        await self.recorder.record_book_snapshot(event)
    
    async def on_delta_batch(self, event) -> None:
        """Применение пакета дельт стакана."""
        self._book_delta_count += 1
        symbol = event.symbol.upper()
        
        # Обновляем локальный стакан
        book = self.books.get(symbol)
        if book is not None:
            ok = book.apply_delta_batch(event)
            if not ok or book.out_of_sync:
                # Логируем инцидент
                self.incident_journal.report_book_out_of_sync(
                    symbol=symbol,
                    reason=book.out_of_sync_reason or "unknown",
                    details={
                        "last_update_id": book.last_update_id,
                    },
                )
        
        # Обновляем анализатор стакана
        self.book_analyzer_manager.on_delta_batch(event)
        
        # Запись на диск
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
        
        # Уровни НЕ сбрасываем — они загружены из истории
        print(
            f"[RESET] Очищены стаканы, tape, bars "
            f"для {len(self.books)} символов (уровни сохранены)"
        )
    
    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        """Обработка state-событий от gateway."""
        print(f"[STATE] {state} {details}")
        
        if state == "WS_DISCONNECT":
            self.incident_journal.report_ws_disconnect(
                message=str(details),
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
        """Закрытие 1с бара (лента)."""
        self._tape_bar_count += 1
    
    def _on_level_bar_close(self, bar: Bar) -> None:
        """Закрытие 5м бара (уровни)."""
        self._level_bar_count += 1
        
        # Передаём бар в LevelManager
        new_levels = self.level_manager.on_bar(bar)
        
        for level in new_levels:
            print(
                f"[LEVEL] {level.side.name} {level.symbol} "
                f"@ {level.center:.2f} "
                f"(touches={level.touches}, strength={level.strength:.2f})"
            )
    
    # ============================================
    # Детекция пробоев
    # ============================================
    
    def _check_breakouts(self, symbol: str, price: float) -> None:
        """
        Проверяет пробой уровней после каждой сделки.
        
        Упрощённая логика для первой версии:
        1. Берём активные уровни для символа
        2. Если цена пересекла уровень сверху (для сопротивления)
           или снизу (для поддержки) — потенциальный пробой
        3. Проверяем условия подтверждения:
           - Импульс в ленте (нетто-дельта в сторону пробоя)
           - Нет крупной стены на пути (из анализатора)
        4. Если все условия — генерируем сигнал
        """
        symbol = symbol.upper()
        active_levels = self.level_manager.get_active_levels(symbol)
        
        for level in active_levels:
            level_id = f"{symbol}_{level.side.name}_{level.center:.0f}"
            
            # Определяем направление пробоя
            if level.side.name == "RESISTANCE" and price > level.center:
                # Потенциальный пробой сопротивления вверх (ЛОНГ)
                self._handle_breakout_signal(
                    symbol=symbol,
                    level=level,
                    level_id=level_id,
                    side=OrderSide.BUY,
                    price=price,
                )
            
            elif level.side.name == "SUPPORT" and price < level.center:
                # Потенциальный пробой поддержки вниз (ШОРТ)
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
        
        # Проверяем, не открыта ли уже позиция по этому символу
        if symbol in self._open_positions:
            return
        
        # Проверяем лимит позиций
        max_positions = self.config.risk.max_open_positions
        if len(self._open_positions) >= max_positions:
            return
        
        # Дебаунс: не генерируем повторный сигнал по тому же уровню
        if level_id in self._breakout_attempts:
            last_attempt = self._breakout_attempts[level_id]
            if (now_ns - last_attempt) < 30_000_000_000:  # 30 секунд
                return
        
        self._breakout_attempts[level_id] = now_ns
        
        # Проверяем импульс в ленте
        tape_metrics = self.tape_manager.snapshot(symbol)
        if tape_metrics is None:
            return
        
        # Для лонга: положительная дельта, для шорта: отрицательная
        if side == OrderSide.BUY and tape_metrics.net_delta_1s <= 0:
            self.decision_journal.log_signal_rejected(
                symbol=symbol,
                signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                reject_reasons=["NO_POSITIVE_DELTA"],
            )
            return
        
        if side == OrderSide.SELL and tape_metrics.net_delta_1s >= 0:
            self.decision_journal.log_signal_rejected(
                symbol=symbol,
                signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                reject_reasons=["NO_NEGATIVE_DELTA"],
            )
            return
        
        # Проверяем отсутствие крупной стены на пути
        book_analyzer = self.book_analyzer_manager._analyzers.get(symbol)
        if book_analyzer is not None:
            liquidity = book_analyzer.get_level_liquidity(level)
            if liquidity is not None and liquidity.has_significant_walls:
                # Есть значимая стена — не пробиваем
                self.decision_journal.log_signal_rejected(
                    symbol=symbol,
                    signal_id=f"sig_{uuid.uuid4().hex[:8]}",
                    reject_reasons=["WALL_ON_PATH"],
                )
                return
        
        # Все проверки прошли — генерируем сигнал
        signal_id = f"sig_{uuid.uuid4().hex[:8]}"
        self._signal_count += 1
        
        print(
            f"[SIGNAL] {side.name} {symbol} @ {price:.2f} "
            f"(level={level.center:.2f}, delta={tape_metrics.net_delta_1s:+.0f})"
        )
        
        # Логируем сигнал
        self.decision_journal.log_signal_created(
            symbol=symbol,
            signal_id=signal_id,
            reasons=["BREAKOUT", "LEVEL_ACTIVE", "DELTA_CONFIRMED"],
        )
        
        # Симулируем исполнение
        self._execute_paper_trade(
            symbol=symbol,
            signal_id=signal_id,
            side=side,
            entry_price=price,
            level=level,
        )
    
    # ============================================
    # Бумажное исполнение
    # ============================================
    
    def _execute_paper_trade(
        self,
        symbol: str,
        signal_id: str,
        side: OrderSide,
        entry_price: float,
        level: Level,
    ) -> None:
        """Симулирует исполнение бумажной сделки."""
        # Расчёт размера позиции
        capital = self.config.risk.initial_capital
        risk_pct = self.config.risk.risk_per_trade_pct
        
        # Стоп за уровнем с буфером
        tick_size = self.tick_sizes.get(symbol, 0.01)
        stop_buffer = max(
            self.config.risk.stop_buffer_ticks * tick_size,
            level.center * self.config.risk.stop_buffer_pct,
        )
        
        if side == OrderSide.BUY:
            stop_price = level.center - stop_buffer
        else:
            stop_price = level.center + stop_buffer
        
        # Риск на сделку в деньгах
        risk_amount = capital * risk_pct / 100.0
        risk_per_unit = abs(entry_price - stop_price)
        
        if risk_per_unit <= 0:
            return
        
        quantity = risk_amount / risk_per_unit
        
        # Проверяем минимальный/максимальный номинал
        notional = quantity * entry_price
        if notional < self.config.risk.min_notional_per_trade:
            quantity = self.config.risk.min_notional_per_trade / entry_price
        elif notional > self.config.risk.max_notional_per_trade:
            quantity = self.config.risk.max_notional_per_trade / entry_price
        
        # Логируем одобрение риска
        self.decision_journal.log_risk_approved(
            symbol=symbol,
            signal_id=signal_id,
            quantity=quantity,
            notional=quantity * entry_price,
            stop_price=stop_price,
        )
        
        # Создаём позицию
        position = PaperPosition(
            position_id=f"pos_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            signal_id=signal_id,
            opened_ts_ns=time.time_ns(),
        )
        
        self._positions[position.position_id] = position
        self._open_positions[symbol] = position
        
        # Логируем открытие
        self.decision_journal.log_position_opened(
            symbol=symbol,
            position_id=position.position_id,
            signal_id=signal_id,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
        )
        
        print(
            f"[PAPER_OPEN] {side.name} {symbol} "
            f"qty={quantity:.6f} @ {entry_price:.2f} "
            f"stop={stop_price:.2f} "
            f"(notional={notional:.2f})"
        )
    
    def check_stop_exits(self, symbol: str, price: float) -> None:
        """Проверяет срабатывание стопов."""
        symbol = symbol.upper()
        position = self._open_positions.get(symbol)
        
        if position is None or not position.is_open:
            return
        
        should_close = False
        reason = ""
        
        if position.side == OrderSide.BUY:
            # Для лонга стоп срабатывает при цене ниже стопа
            if price <= position.stop_price:
                should_close = True
                reason = "STOP_LOSS"
        else:
            # Для шорта стоп срабатывает при цене выше стопа
            if price >= position.stop_price:
                should_close = True
                reason = "STOP_LOSS"
        
        if should_close:
            self._close_position(position, price, reason)
    
    def _close_position(
        self,
        position: PaperPosition,
        exit_price: float,
        reason: str,
    ) -> None:
        """Закрывает бумажную позицию."""
        # Расчёт PnL
        if position.side == OrderSide.BUY:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Комиссия (тейкер)
        fee_pct = self.config.execution.taker_fee_pct
        fees = abs(exit_price * position.quantity) * fee_pct / 100.0
        
        net_pnl = gross_pnl - fees
        
        # Обновляем позицию
        position.is_open = False
        position.closed_ts_ns = time.time_ns()
        position.exit_price = exit_price
        position.realized_pnl = net_pnl
        position.exit_reason = reason
        
        # Удаляем из открытых
        self._open_positions.pop(position.symbol, None)
        
        # Статистика
        self._total_pnl += net_pnl
        if net_pnl > 0:
            self._win_count += 1
        else:
            self._loss_count += 1
        
        # Логируем закрытие
        self.decision_journal.log_position_closed(
            symbol=position.symbol,
            position_id=position.position_id,
            realized_pnl=net_pnl,
            reasons=[reason],
        )
        
        print(
            f"[PAPER_CLOSE] {position.symbol} "
            f"exit={exit_price:.2f} "
            f"pnl={net_pnl:+.2f} "
            f"({reason}) "
            f"[total={self._total_pnl:+.2f}]"
        )
    
    # ============================================
    # Статус
    # ============================================
    
    def _print_status(self) -> None:
        """Печатает периодический статус."""
        print(
            f"[STATUS] trades={self._trade_count}, "
            f"tickers={self._book_ticker_count}, "
            f"deltas={self._book_delta_count}, "
            f"signals={self._signal_count}, "
            f"positions_open={len(self._open_positions)}, "
            f"pnl={self._total_pnl:+.2f} "
            f"(W{self._win_count}/L{self._loss_count})"
        )
        
        # Метрики по каждому символу
        for symbol in self.books.keys():
            metrics = self.tape_manager.snapshot(symbol)
            if metrics is not None:
                print(
                    f"[TAPE] {symbol}: "
                    f"Δ1s={metrics.net_delta_1s:+.0f}, "
                    f"rate={metrics.trades_per_sec_1s:.1f}/s, "
                    f"price={metrics.last_price:.2f}"
                )
                
                # Проверяем стопы по текущей цене
                self.check_stop_exits(symbol, metrics.last_price)
        
        # Уровни
        for symbol in self.books.keys():
            active_levels = self.level_manager.get_active_levels(symbol)
            if active_levels:
                levels_str = ", ".join(
                    f"{lvl.side.name[0]}{lvl.center:.0f}"
                    for lvl in active_levels[:5]
                )
                print(
                    f"[LEVELS] {symbol}: "
                    f"{len(active_levels)} active ({levels_str})"
                )


# ============================================================
# Основная функция запуска
# ============================================================

async def run_paper_runner(config: AppConfig) -> None:
    """Основная функция paper runner."""
    
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
        
        print(f"Загружено инструментов: {len(registry.all_symbols())}")
        print(f"Рабочие символы: {', '.join(config.symbols)}")
        
        # Загрузка истории для уровней
        print(
            f"Загрузка истории: {config.collector.history_hours}ч "
            f"× {config.collector.history_interval} бары..."
        )
        
        for symbol in config.symbols:
            try:
                klines = await rest_client.get_klines_last_hours(
                    symbol=symbol,
                    interval=config.collector.history_interval,
                    hours=config.collector.history_hours,
                )
                history_klines[symbol] = klines
                print(f"  {symbol}: загружено {len(klines)} свечей")
            except Exception as exc:
                print(f"  [WARN] {symbol}: не удалось загрузить историю: {exc}")
    
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
        books[symbol.upper()] = FastOrderBook(
            symbol=symbol,
            tick_size=tick_size,
        )
    
    # ========================================
    # 3. Создаём основные менеджеры
    # ========================================
    guard = BookTickerGuard(
        config=FastGuardConfig(),
        tick_sizes=tick_sizes,
    )
    
    tape_manager = TapeManager()
    
    # LevelManager + загрузка истории
    level_manager = LevelManager(tick_sizes=tick_sizes)
    
    for symbol, klines in history_klines.items():
        detector = level_manager.get_or_create(symbol)
        level_count = detector.load_from_klines(klines)
        print(
            f"[LEVELS] {symbol}: инициализировано {level_count} уровней "
            f"из {len(klines)} баров истории"
        )
    
    # Анализатор стакана
    book_analyzer_manager = BookAnalyzerManager(tick_sizes=tick_sizes)
    
    # Спуфинг и айсберги
    spoof_manager = SpoofDetectorManager(tick_sizes=tick_sizes)
    iceberg_manager = IcebergDetectorManager(tick_sizes=tick_sizes)
    
    # Метрики инструментов
    metrics_manager = SymbolMetricsManager(tick_sizes=tick_sizes)
    
    # ========================================
    # 4. Создаём журналы
    # ========================================
    decision_journal = DecisionJournal(
        config=DecisionJournalConfig(
            base_dir=config.journal.decision_journal_dir,
        )
    )
    await decision_journal.start()
    
    incident_journal = IncidentJournal(
        config=IncidentJournalConfig(
            base_dir=config.journal.incident_journal_dir,
        )
    )
    await incident_journal.start()
    
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
    handler = PaperRunnerHandler(
        config=config,
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
        tick_sizes=tick_sizes,
    )
    
    # ========================================
    # 7. Создаём агрегаторы баров
    # ========================================
    TAPE_TF_NS = 1_000_000_000        # 1 секунда
    LEVEL_TF_NS = 300_000_000_000     # 5 минут
    
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
    # 8. Создаём WS-шлюз
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
    
    # ========================================
    # 9. Обработка сигналов остановки
    # ========================================
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    
    def signal_handler() -> None:
        print("\nПолучен сигнал остановки")
        stop_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    # ========================================
    # 10. Запуск
    # ========================================
    print("=" * 60)
    print("PAPER RUNNER запущен")
    print(f"Режим: {config.execution.mode}")
    print(f"Символы: {', '.join(config.symbols)}")
    print(f"Макс позиций: {config.risk.max_open_positions}")
    print(f"Риск на сделку: {config.risk.risk_per_trade_pct}%")
    print("=" * 60)
    
    try:
        gateway_task = asyncio.create_task(gateway.run())
        await stop_event.wait()
        
        await gateway.stop()
        gateway_task.cancel()
        
        try:
            await gateway_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"Gateway завершился с ошибкой: {exc}")
    
    except asyncio.CancelledError:
        pass
    
    finally:
        # Закрываем все позиции по последним ценам
        for symbol, position in list(handler._open_positions.items()):
            book = books.get(symbol)
            if book is not None and book.mid_price is not None:
                handler._close_position(
                    position,
                    book.mid_price,
                    reason="FORCED_CLOSE",
                )
        
        # Flush бары
        for aggregator in handler.tape_aggregators.values():
            aggregator.flush()
        for aggregator in handler.level_aggregators.values():
            aggregator.flush()
        
        # Закрываем журналы и рекордер
        await recorder.close()
        await decision_journal.close()
        await incident_journal.close()
        
        # Финальная статистика
        print("\n" + "=" * 60)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        print(f"Сигналов: {handler._signal_count}")
        print(f"Сделок закрыто: {handler._win_count + handler._loss_count}")
        print(f"  Выигрышных: {handler._win_count}")
        print(f"  Проигрышных: {handler._loss_count}")
        if handler._win_count + handler._loss_count > 0:
            win_rate = handler._win_count / (handler._win_count + handler._loss_count)
            print(f"Win rate: {win_rate * 100:.1f}%")
        print(f"Общий PnL: {handler._total_pnl:+.2f} USDT")
        
        # Статистика журналов
        dj_stats = decision_journal.get_stats()
        ij_stats = incident_journal.get_stats()
        print(f"\nDecision Journal: written={dj_stats['written']}, dropped={dj_stats['dropped']}")
        print(f"Incident Journal: written={ij_stats['written']}")
        
        print("\nPaper runner stopped")


def main() -> None:
    """Точка входа."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    
    config = load_app_config(config_path)
    
    # Проверяем режим
    if config.execution.mode != "paper":
        print(f"[WARN] Режим '{config.execution.mode}', но запускаем paper")
    
    try:
        asyncio.run(run_paper_runner(config))
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()