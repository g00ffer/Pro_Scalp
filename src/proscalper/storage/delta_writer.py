"""
Фоновая запись рыночных дельт.

Цель:
- не блокировать hot-path;
- писать сырые события батчами;
- поддерживать раздельное хранение по типу события, символу и дате.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import msgspec

from proscalper.core.events import (
    BookDeltaBatchEvent,
    BookSnapshotEvent,
    BookTickerEvent,
    TradeEvent,
)

EventType = Union[
    TradeEvent,
    BookTickerEvent,
    BookDeltaBatchEvent,
    BookSnapshotEvent,
]


class AsyncEventWriter:
    """
    Асинхронный писатель событий в JSONL.

    Для production можно заменить JSONL на msgpack/Arrow,
    но для MVP и аудита JSONL удобен.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        event_type: str,
        batch_size: int = 1000,
        flush_interval_ms: int = 200,
        queue_size: int = 50_000,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._event_type = event_type
        self._batch_size = batch_size
        self._flush_interval_sec = flush_interval_ms / 1000.0
        self._queue: asyncio.Queue[EventType] = asyncio.Queue(maxsize=queue_size)

        self._task: Optional[asyncio.Task] = None
        self._closed = False

        self.written_events = 0
        self.dropped_events = 0

    async def start(self) -> None:
        if self._task is not None:
            return

        self._closed = False
        self._task = asyncio.create_task(self._run())

    async def write(self, event: EventType) -> None:
        if self._closed:
            return

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped_events += 1

    async def close(self) -> None:
        self._closed = True

        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            batch: List[EventType] = []

            try:
                first_event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._flush_interval_sec,
                )
                batch.append(first_event)

            except asyncio.TimeoutError:
                pass

            while len(batch) < self._batch_size:
                try:
                    event = self._queue.get_nowait()
                    batch.append(event)
                except asyncio.QueueEmpty:
                    break

            if batch:
                await asyncio.to_thread(self._flush_batch, batch)
                self.written_events += len(batch)

            if self._closed and self._queue.empty():
                break

    def _flush_batch(self, batch: List[EventType]) -> None:
        grouped: Dict[Path, List[bytes]] = defaultdict(list)

        for event in batch:
            path = self._path_for_event(event)
            encoded = msgspec.json.encode(event)
            grouped[path].append(encoded + b"\n")

        for path, encoded_events in grouped.items():
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "ab") as file:
                file.write(b"".join(encoded_events))

    def _path_for_event(self, event: EventType) -> Path:
        symbol = getattr(event, "symbol", "GLOBAL")
        if not symbol:
            symbol = "GLOBAL"

        symbol = str(symbol).upper()

        ts_ns = getattr(event, "ts_local_ns", 0)
        if ts_ns:
            gmtime = time.gmtime(ts_ns / 1_000_000_000)
        else:
            gmtime = time.gmtime()

        date_str = time.strftime("%Y-%m-%d", gmtime)

        return (
            self._base_dir
            / self._event_type
            / date_str
            / f"{symbol}.jsonl"
        )


class MarketDataRecorder:
    """
    Рекордер рыночных данных.

    Пишет:
    - сделки;
    - bookTicker;
    - дельты стакана;
    - периодические снапшоты как keyframes.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        batch_size: int = 1000,
        flush_interval_ms: int = 200,
        queue_size: int = 50_000,
    ) -> None:
        base_dir = Path(base_dir)

        self._writers = {
            "trade": AsyncEventWriter(
                base_dir=base_dir,
                event_type="trade",
                batch_size=batch_size,
                flush_interval_ms=flush_interval_ms,
                queue_size=queue_size,
            ),
            "book_ticker": AsyncEventWriter(
                base_dir=base_dir,
                event_type="book_ticker",
                batch_size=batch_size,
                flush_interval_ms=flush_interval_ms,
                queue_size=queue_size,
            ),
            "book_delta": AsyncEventWriter(
                base_dir=base_dir,
                event_type="book_delta",
                batch_size=batch_size,
                flush_interval_ms=flush_interval_ms,
                queue_size=queue_size,
            ),
            "book_snapshot": AsyncEventWriter(
                base_dir=base_dir,
                event_type="book_snapshot",
                batch_size=max(1, batch_size // 10),
                flush_interval_ms=flush_interval_ms,
                queue_size=max(100, queue_size // 10),
            ),
        }

    async def start(self) -> None:
        for writer in self._writers.values():
            await writer.start()

    async def close(self) -> None:
        for writer in self._writers.values():
            await writer.close()

    async def record_trade(self, event: TradeEvent) -> None:
        await self._writers["trade"].write(event)

    async def record_book_ticker(self, event: BookTickerEvent) -> None:
        await self._writers["book_ticker"].write(event)

    async def record_book_delta(self, event: BookDeltaBatchEvent) -> None:
        await self._writers["book_delta"].write(event)

    async def record_book_snapshot(self, event: BookSnapshotEvent) -> None:
        await self._writers["book_snapshot"].write(event)

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {
            name: {
                "written": writer.written_events,
                "dropped": writer.dropped_events,
            }
            for name, writer in self._writers.items()
        }