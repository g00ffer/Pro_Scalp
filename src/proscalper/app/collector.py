"""
Сборщик рыночных данных.

Запускает WebSocket-шлюз, синхронизирует стаканы, анализирует ленту сделок,
агрегирует бары, детектирует уровни и записывает рыночные события на диск.

Обновление:
- Загрузка 24ч истории через REST при старте для инициализации уровней
- Разделение агрегаторов: 1с (лента) и 5м (уровни)
- Детектор уровней работает на 5-минутных барах

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


# ============================================================
# Константы таймфреймов
# ============================================================

TAPE_TF_NS = 1_000_000_000        # 1 секунда — для ленты и компрессии
LEVEL_TF_NS = 300_000_000_000     # 5 минут — для детектора уровней


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
    # Параметры загрузки истории
    history_interval: str = "5m"
    history_hours: float = 24.0


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
        history_interval=collector_data.get("history_interval", "5m"),
        history_hours=collector_data.get("history_hours", 24.0),
    )


class CollectorHandler:
    """
    Обработчик рыночных событий для сборщика.

    Связывает WS gateway с:
    - рекордером (запись на диск)
    - локальными стаканами (FastOrderBook)
    - BookTickerGuard (fast validation)
    - TapeManager (анализ ленты)
    - ленточными агрегаторами (1с бары)
    - уровневыми агрегаторами (5м бары)
    - LevelManager (детектор уровней)
    """

    def __init__(
        self,
        recorder: MarketDataRecorder,
        books: Dict[str, FastOrderBook],
        guard: BookTickerGuard,
        tape_manager: TapeManager,
        tape_aggregators: Dict[str, MultiTimeframeAggregator],
        level_aggregators: Dict[str, MultiTimeframeAggregator],
        level_manager: LevelManager,
    ) -> None:
        self.recorder = recorder
        self.books = books
        self.guard = guard
        self.tape_manager = tape_manager
        self.tape_aggregators = tape_aggregators
        self.level_aggregators = level_aggregators
        self.level_manager = level_manager

        # Счётчики для status logger
        self._trade_count = 0
        self._book_ticker_count = 0
        self._book_delta_count = 0
        self._tape_bar_count = 0
        self._level_bar_count = 0
        self._level_count = 0
        self._last_status_ts = time.time()

    def _on_tape_bar_close(self, bar: Bar) -> None:
        """Callback при закрытии 1с бара (лента/компрессия)."""
        self._tape_bar_count += 1

    def _on_level_bar_close(self, bar: Bar) -> None:
        """Callback при закрытии 5м бара (детектор уровней)."""
        self._level_bar_count += 1

        # Передаём 5м бар в LevelManager
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

        # Обновляем ленточные агрегаторы (1с)
        tape_agg = self.tape_aggregators.get(event.symbol.upper())
        if tape_agg is not None:
            tape_agg.on_trade(event)

        # Обновляем уровневые агрегаторы (5м)
        level_agg = self.level_aggregators.get(event.symbol.upper())
        if level_agg is not None:
            level_agg.on_trade(event)

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

        for aggregator in self.tape_aggregators.values():
            aggregator.reset()

        for aggregator in self.level_aggregators.values():
            aggregator.reset()

        # НЕ сбрасываем level_manager — уровни загружены из истории
        # и должны сохраняться между reconnect'ами

        print(
            f"[RESET] Очищены стаканы, tape, bars "
            f"для {len(self.books)} символов "
            f"(уровни сохранены)"
        )

    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        """Обработка state-событий от gateway."""
        print(f"[STATE] {state} {details}")

        # Печатаем статус каждые 10 секунд
        now = time.time()
        if now - self._last_status_ts >= 10.0:
            print(
                f"[STATUS] trades={self._trade_count}, "
                f"book_tickers={self._book_ticker_count}, "
                f"book_deltas={self._book_delta_count}, "
                f"tape_bars={self._tape_bar_count}, "
                f"level_bars={self._level_bar_count}, "
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
    # 1. Загружаем информацию об инструментах + историю
    # ========================================
    registry = InstrumentRegistry()
    history_klines: Dict[str, List] = {}

    async with BinanceFuturesRestClient() as rest_client:
        instruments = await rest_client.get_exchange_info()
        registry.load_from_exchange_info(instruments)

        # Проверяем, что все символы есть в реестре
        for symbol in config.symbols:
            if registry.get_instrument(symbol) is None:
                raise ValueError(f"Символ {symbol} не найден в exchangeInfo")

        print(f"Загружено инструментов: {len(registry.all_symbols())}")
        print(f"Рабочие символы: {', '.join(config.symbols)}")

        # ============================================================
        # ЗАГРУЗКА ИСТОРИИ ДЛЯ ДЕТЕКТОРА УРОВНЕЙ
        # 24 часа × 5м = 288 баров на символ
        # ============================================================
        print(
            f"Загрузка истории: {config.history_hours}ч "
            f"× {config.history_interval} бары..."
        )

        for symbol in config.symbols:
            try:
                klines = await rest_client.get_klines_last_hours(
                    symbol=symbol,
                    interval=config.history_interval,
                    hours=config.history_hours,
                )
                history_klines[symbol] = klines
                print(
                    f"  {symbol}: загружено {len(klines)} свечей"
                )
            except Exception as exc:
                print(
                    f"  [WARN] {symbol}: не удалось загрузить историю: {exc}"
                )

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
    # 5. Создаём LevelManager + загружаем историю
    # ========================================
    level_manager = LevelManager(tick_sizes=tick_sizes)

    # Инициализация детекторов уровней из истории
    for symbol, klines in history_klines.items():
        detector = level_manager.get_or_create(symbol)
        level_count = detector.load_from_klines(klines)
        print(
            f"[LEVELS] {symbol}: инициализировано {level_count} уровней "
            f"из {len(klines)} баров истории"
        )

    # Для символов без истории — просто создаём детектор
    for symbol in config.symbols:
        level_manager.get_or_create(symbol)

    # ========================================
    # 6. Создаём ленточные агрегаторы (1с)
    # ========================================
    tape_aggregators: Dict[str, MultiTimeframeAggregator] = {}

    def _temp_tape_bar_close(bar: Bar) -> None:
        pass  # будет заменён после создания handler

    for symbol in config.symbols:
        aggregator = MultiTimeframeAggregator(
            symbol=symbol,
            timeframes_ns=[TAPE_TF_NS],  # 1 секунда
            on_bar_close=_temp_tape_bar_close,
        )
        tape_aggregators[symbol.upper()] = aggregator

    # ========================================
    # 7. Создаём уровневые агрегаторы (5м)
    # ========================================
    level_aggregators: Dict[str, MultiTimeframeAggregator] = {}

    def _temp_level_bar_close(bar: Bar) -> None:
        pass  # будет заменён после создания handler

    for symbol in config.symbols:
        aggregator = MultiTimeframeAggregator(
            symbol=symbol,
            timeframes_ns=[LEVEL_TF_NS],  # 5 минут
            on_bar_close=_temp_level_bar_close,
        )
        level_aggregators[symbol.upper()] = aggregator

    # ========================================
    # 8. Создаём рекордер
    # ========================================
    recorder = MarketDataRecorder(
        base_dir=config.data_dir,
        batch_size=config.batch_size,
        flush_interval_ms=config.flush_interval_ms,
        queue_size=config.queue_size,
    )
    await recorder.start()

    # ========================================
    # 9. Создаём обработчик
    # ========================================
    handler = CollectorHandler(
        recorder=recorder,
        books=books,
        guard=guard,
        tape_manager=tape_manager,
        tape_aggregators=tape_aggregators,
        level_aggregators=level_aggregators,
        level_manager=level_manager,
    )

    # Подключаем реальные callback'и к агрегаторам
    for aggregator in tape_aggregators.values():
        for inner_agg in aggregator._aggregators.values():
            inner_agg._on_bar_close = handler._on_tape_bar_close

    for aggregator in level_aggregators.values():
        for inner_agg in aggregator._aggregators.values():
            inner_agg._on_bar_close = handler._on_level_bar_close

    # ========================================
    # 10. Создаём WS-шлюз
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
    # 11. Обработка сигналов остановки
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
            pass  # Windows

    # ========================================
    # 12. Запуск
    # ========================================
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
        # Flush все открытые бары
        for aggregator in tape_aggregators.values():
            aggregator.flush()
        for aggregator in level_aggregators.values():
            aggregator.flush()

        await recorder.close()

        # Статистика записи
        stats = recorder.stats()
        print("\nСтатистика записи:")
        for event_type, counts in stats.items():
            print(
                f"  {event_type}: "
                f"written={counts['written']}, "
                f"dropped={counts['dropped']}"
            )

        print(f"\nДополнительно:")
        print(f"  tape_bars (1s): {handler._tape_bar_count}")
        print(f"  level_bars (5m): {handler._level_bar_count}")
        print(f"  new_levels: {handler._level_count}")

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