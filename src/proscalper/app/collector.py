"""
Сборщик рыночных данных.

Запускает WebSocket-шлюз, синхронизирует стаканы, анализирует ленту сделок,
агрегирует бары, детектирует уровни и записывает рыночные события на диск.

Использование:
    python -m proscalper.app.collector configs/default.yaml
"""
from __future__ import annotations

import asyncio
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

from proscalper.core.instrument_registry import InstrumentRegistry
from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient
from proscalper.exchange.binance_futures.ws_market import (
    BinanceFuturesMarketDataWS,
    BinanceFuturesWSConfig,
)
from proscalper.market_data.bookticker_guard import (
    BookTickerGuard,
    FastGuardConfig,
)
from proscalper.market_data.orderbook_fast import FastOrderBook
from proscalper.market_data.tape import TapeManager
from proscalper.market_data.bar_aggregator import MultiTimeframeAggregator, Bar
from proscalper.features.levels import LevelManager, LevelDetectorConfig
from proscalper.storage.delta_writer import MarketDataRecorder


@dataclass
class CollectorConfig:
    """Конфигурация сборщика данных."""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    data_dir: str = "./data/binance_futures"
    depth_update_speed: str = "100ms"
    snapshot_limit: int = 1000
    batch_size: int = 1000
    flush_interval_ms: int = 200
    queue_size: int = 50_000


def load_config(config_path: str) -> CollectorConfig:
    """Загрузка конфигурации из YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    collector_data = data.get("collector", {})

    return CollectorConfig(
        symbols=collector_data.get("symbols", ["BTCUSDT", "ETHUSDT"]),
        data_dir=collector_data.get("data_dir", "./data/binance_futures"),
        depth_update_speed=collector_data.get("depth_update_speed", "100ms"),
        snapshot_limit=collector_data.get("snapshot_limit", 1000),
        batch_size=collector_data.get("batch_size", 1000),
        flush_interval_ms=collector_data.get("flush_interval_ms", 200),
        queue_size=collector_data.get("queue_size", 50_000),
    )


class CollectorHandler:
    """
    Обработчик рыночных событий для сборщика.

    Связывает WS gateway с:
    - рекордером (запись на диск)
    - локальными стаканами (FastOrderBook)
    - BookTickerGuard (fast validation)
    - TapeManager (анализ ленты)
    - MultiTimeframeAggregator (построение баров)
    - LevelManager (детектор уровней)
    """

    def __init__(
        self,
        recorder: MarketDataRecorder,
        books: Dict[str, FastOrderBook],
        guard: BookTickerGuard,
        tape_manager: TapeManager,
        bar_aggregators: Dict[str, MultiTimeframeAggregator],
        level_manager: LevelManager,
    ) -> None:
        self.recorder = recorder
        self.books = books
        self.guard = guard
        self.tape_manager = tape_manager
        self.bar_aggregators = bar_aggregators
        self.level_manager = level_manager

        # Счётчики для status logger
        self._trade_count = 0
        self._book_ticker_count = 0
        self._book_delta_count = 0
        self._bar_count = 0
        self._level_count = 0
        self._last_status_ts = time.time()

    def _on_bar_close(self, bar: Bar) -> None:
        """Callback при закрытии бара."""
        self._bar_count += 1

        # Передаём бар в LevelManager
        new_levels = self.level_manager.on_bar(bar)
        self._level_count += len(new_levels)

        # Логируем новые уровни
        for level in new_levels:
            print(
                f"[LEVEL] {level.side.name} {level.symbol} "
                f"@ {level.center:.2f} "
                f"(touches={level.touches}, strength={level.strength:.2f})"
            )

    async def on_trade(self, event) -> None:
        """Обработка сделки."""
        self._trade_count += 1

        # Обновляем TapeAnalyzer
        self.tape_manager.on_trade(event)

        # Обновляем BarAggregator
        aggregator = self.bar_aggregators.get(event.symbol.upper())
        if aggregator is not None:
            aggregator.on_trade(event)

        await self.recorder.record_trade(event)

    async def on_book_ticker(self, event) -> None:
        """Обработка обновления лучших цен."""
        self._book_ticker_count += 1
        self.guard.update(event)
        await self.recorder.record_book_ticker(event)

    async def on_snapshot(self, event) -> None:
        """Применение snapshot стакана."""
        book = self.books.get(event.symbol.upper())
        if book is not None:
            book.apply_snapshot(event)
        await self.recorder.record_book_snapshot(event)

    async def on_delta_batch(self, event) -> None:
        """Применение пакета дельт стакана."""
        self._book_delta_count += 1

        book = self.books.get(event.symbol.upper())

        if book is not None:
            ok = book.apply_delta_batch(event)

            if not ok or book.out_of_sync:
                print(
                    f"[OUT_OF_SYNC] {event.symbol} "
                    f"reason={book.out_of_sync_reason}"
                )

        await self.recorder.record_book_delta(event)

    async def on_snapshot_requested(self, symbol: str) -> None:
        """
        Вызывается перед запросом snapshot через REST.

        Помечает стакан, что snapshot запрошен, чтобы буферизовать
        приходящие depth events до применения snapshot.
        """
        book = self.books.get(symbol.upper())
        if book is not None:
            book.mark_snapshot_requested()

    async def reset_books(self) -> None:
        """
        Сбрасывает все локальные стаканы, tape, bars и levels.

        Вызывается перед каждым новым подключением к WS,
        чтобы старые события не мешали синхронизации.
        """
        for symbol, book in self.books.items():
            book.reset()

        self.tape_manager.reset_all()

        for aggregator in self.bar_aggregators.values():
            aggregator.reset()

        self.level_manager.reset_all()

        print(
            f"[RESET] Очищены стаканы, tape, bars и levels "
            f"для {len(self.books)} символов"
        )

    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        """Обработка state-событий от gateway."""
        print(f"[STATE] {state} {details}")

        # Печатаем статус каждые 10 секунд
        now = time.time()
        if now - self._last_status_ts >= 10.0:
            # Базовые счётчики
            print(
                f"[STATUS] trades={self._trade_count}, "
                f"book_tickers={self._book_ticker_count}, "
                f"book_deltas={self._book_delta_count}, "
                f"bars={self._bar_count}, "
                f"levels={self._level_count}"
            )

            # Tape метрики для каждого символа
            for symbol in self.books.keys():
                metrics = self.tape_manager.snapshot(symbol)
                if metrics is not None:
                    print(
                        f"[TAPE] {symbol}: "
                        f"Δ1s={metrics.net_delta_1s:+.0f}, "
                        f"rate={metrics.trades_per_sec_1s:.1f}/s, "
                        f"imb={metrics.volume_imbalance_1s:+.2f}, "
                        f"price={metrics.last_price:.2f}"
                    )

            # Уровни для каждого символа
            for symbol in self.books.keys():
                active_levels = self.level_manager.get_active_levels(symbol)
                if active_levels:
                    levels_str = ", ".join(
                        f"{lvl.side.name[0]}{lvl.center:.0f}"
                        for lvl in active_levels[:5]  # топ-5
                    )
                    print(
                        f"[LEVELS] {symbol}: "
                        f"{len(active_levels)} active ({levels_str})"
                    )

            self._last_status_ts = now


async def run_collector(config: CollectorConfig) -> None:
    """
    Основная функция сборщика.
    """
    # ========================================
    # 1. Загружаем информацию об инструментах
    # ========================================
    registry = InstrumentRegistry()

    async with BinanceFuturesRestClient() as rest_client:
        instruments = await rest_client.get_exchange_info()
        registry.load_from_exchange_info(instruments)

    # Проверяем, что все символы есть в реестре
    for symbol in config.symbols:
        if registry.get_instrument(symbol) is None:
            raise ValueError(f"Символ {symbol} не найден в exchangeInfo")

    print(f"Загружено инструментов: {len(registry.all_symbols())}")
    print(f"Рабочие символы: {', '.join(config.symbols)}")

    # ========================================
    # 2. Создаём локальные стаканы
    # ========================================
    books: Dict[str, FastOrderBook] = {}

    for symbol in config.symbols:
        tick_size = registry.get_tick_size(symbol)
        if tick_size is None:
            raise ValueError(f"Не найден tick_size для {symbol}")

        books[symbol.upper()] = FastOrderBook(
            symbol=symbol,
            tick_size=tick_size,
        )

    # ========================================
    # 3. Создаём BookTickerGuard
    # ========================================
    tick_sizes = {
        s: registry.get_tick_size(s)
        for s in config.symbols
    }

    guard = BookTickerGuard(
        config=FastGuardConfig(),
        tick_sizes=tick_sizes,
    )

    # ========================================
    # 4. Создаём TapeManager
    # ========================================
    tape_manager = TapeManager()

    # ========================================
    # 5. Создаём LevelManager
    # ========================================
    level_manager = LevelManager(tick_sizes=tick_sizes)

    # ========================================
    # 6. Создаём BarAggregators
    # ========================================
    bar_aggregators: Dict[str, MultiTimeframeAggregator] = {}

    # Временный callback для bar close (будет заменён после создания полного handler)
    def _temp_on_bar_close(bar: Bar) -> None:
        pass

    for symbol in config.symbols:
        aggregator = MultiTimeframeAggregator(
            symbol=symbol,
            timeframes_ns=[
                1_000_000_000,   # 1 second
                5_000_000_000,   # 5 seconds
            ],
            on_bar_close=_temp_on_bar_close,
        )
        bar_aggregators[symbol.upper()] = aggregator

        # Создаём LevelDetector для этого символа
        level_manager.get_or_create(symbol)

    # ========================================
    # 7. Создаём рекордер
    # ========================================
    recorder = MarketDataRecorder(
        base_dir=config.data_dir,
        batch_size=config.batch_size,
        flush_interval_ms=config.flush_interval_ms,
        queue_size=config.queue_size,
    )

    await recorder.start()

    # ========================================
    # 8. Создаём обработчик
    # ========================================
    handler = CollectorHandler(
        recorder=recorder,
        books=books,
        guard=guard,
        tape_manager=tape_manager,
        bar_aggregators=bar_aggregators,
        level_manager=level_manager,
    )

    # Обновляем callback в bar aggregators на реальный метод handler
    for aggregator in bar_aggregators.values():
        for inner_agg in aggregator._aggregators.values():
            inner_agg._on_bar_close = handler._on_bar_close

    # ========================================
    # 9. Создаём WS-шлюз
    # ========================================
    ws_config = BinanceFuturesWSConfig(
        symbols=config.symbols,
        depth_update_speed=config.depth_update_speed,
        snapshot_limit=config.snapshot_limit,
    )

    gateway = BinanceFuturesMarketDataWS(
        config=ws_config,
        handler=handler,
    )

    # ========================================
    # 10. Обработка сигналов остановки
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
            # Windows не поддерживает add_signal_handler для SIGTERM
            pass

    # ========================================
    # 11. Запуск
    # ========================================
    try:
        gateway_task = asyncio.create_task(gateway.run())

        # Ждём сигнала остановки
        await stop_event.wait()

        # Останавливаем шлюз
        await gateway.stop()

        # Быстрая отмена без долгого ожидания
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
        # Flush все открытые бары
        for aggregator in bar_aggregators.values():
            aggregator.flush()

        await recorder.close()

        # Выводим статистику
        stats = recorder.stats()
        print("\nСтатистика записи:")
        for event_type, counts in stats.items():
            print(
                f"  {event_type}: "
                f"written={counts['written']}, "
                f"dropped={counts['dropped']}"
            )

        print(f"\nДополнительно:")
        print(f"  bars: {handler._bar_count}")
        print(f"  levels: {handler._level_count}")

        # Выводим активные уровни
        for symbol in config.symbols:
            active_levels = level_manager.get_active_levels(symbol)
            if active_levels:
                print(f"\n  {symbol} активные уровни:")
                for level in active_levels[:10]:
                    print(
                        f"    {level.side.name:10} @ {level.center:8.2f} "
                        f"(touches={level.touches}, strength={level.strength:.2f})"
                    )

        print("\nCollector stopped")


def main() -> None:
    """Точка входа."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Файл конфигурации не найден: {config_path}")
        sys.exit(1)

    try:
        asyncio.run(run_collector(config))
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()