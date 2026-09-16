"""
Сборщик рыночных данных.

Запускает WebSocket-шлюз, синхронизирует стаканы и записывает
рыночные события на диск.

Использование:
    python -m proscalper.app.collector configs/default.yaml
"""
from __future__ import annotations

import asyncio
import signal
import sys
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
from proscalper.storage.delta_writer import MarketDataRecorder


@dataclass
class CollectorConfig:
    """Конфигурация сборщика данных."""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    data_dir: str = "./data/binance_futures"
    depth_update_speed: str = "100ms"
    snapshot_limit: int = 100
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
        snapshot_limit=collector_data.get("snapshot_limit", 100),
        batch_size=collector_data.get("batch_size", 1000),
        flush_interval_ms=collector_data.get("flush_interval_ms", 200),
        queue_size=collector_data.get("queue_size", 50_000),
    )


class CollectorHandler:
    """
    Обработчик рыночных событий для сборщика.

    Связывает WS gateway с рекордером и локальными стаканами.
    """

    def __init__(
        self,
        recorder: MarketDataRecorder,
        books: Dict[str, FastOrderBook],
        guard: BookTickerGuard,
    ) -> None:
        self.recorder = recorder
        self.books = books
        self.guard = guard

    async def on_trade(self, event) -> None:
        await self.recorder.record_trade(event)

    async def on_book_ticker(self, event) -> None:
        self.guard.update(event)
        await self.recorder.record_book_ticker(event)

    async def on_snapshot(self, event) -> None:
        book = self.books.get(event.symbol.upper())
        if book is not None:
            book.apply_snapshot(event)
        await self.recorder.record_book_snapshot(event)

    async def on_delta_batch(self, event) -> None:
        book = self.books.get(event.symbol.upper())

        if book is not None:
            ok = book.apply_delta_batch(event)

            if not ok or book.out_of_sync:
                print(
                    f"[OUT_OF_SYNC] {event.symbol} "
                    f"reason={book.out_of_sync_reason}"
                )

        await self.recorder.record_book_delta(event)

    async def on_state(self, state: str, details: Dict[str, Any]) -> None:
        print(f"[STATE] {state} {details}")


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
    # 4. Создаём рекордер
    # ========================================
    recorder = MarketDataRecorder(
        base_dir=config.data_dir,
        batch_size=config.batch_size,
        flush_interval_ms=config.flush_interval_ms,
        queue_size=config.queue_size,
    )

    await recorder.start()

    # ========================================
    # 5. Создаём обработчик
    # ========================================
    handler = CollectorHandler(
        recorder=recorder,
        books=books,
        guard=guard,
    )

    # ========================================
    # 6. Создаём WS-шлюз
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
    # 7. Обработка сигналов остановки
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
    # 8. Запуск
    # ========================================
    try:
        gateway_task = asyncio.create_task(gateway.run())

        # Ждём сигнала остановки
        await stop_event.wait()

        # Останавливаем шлюз
        await gateway.stop()

        # Ждём завершения задачи шлюза
        try:
            await gateway_task
        except asyncio.CancelledError:
            pass

    finally:
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

        print("Collector stopped")


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