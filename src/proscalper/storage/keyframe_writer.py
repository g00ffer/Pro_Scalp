"""
Писатель keyframes стакана (периодических снимков).

Проблема: писать полный снапшот стакана каждые 100мс — это:
- сотни гигабайт в день
- лишняя нагрузка на диск
- бесполезные данные для replay (дельты уже всё восстанавливают)

Решение:
- пишем дельты (уже делает delta_writer.py)
- периодически (каждые N секунд) пишем keyframe — полный снимок стакана
- при replay: берём ближайший keyframe + применяем дельты

Дополнительные триггеры keyframe:
- после успешного resync
- при старте записи
- при смене UTC-дня
- при рассинхроне (out_of_sync)
- по таймеру (keyframe_interval_sec)

Формат: JSONL (совместимо с delta_writer).
Файл: <base_dir>/<YYYY-MM-DD>/<SYMBOL>.jsonl

Архитектурное решение:
- модуль работает с FastOrderBook
- пишет через общий AsyncEventWriter
- не блокирует hot path
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from proscalper.core.events import BookSnapshotEvent


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class KeyframeConfig:
    """
    Конфигурация писателя keyframes.
    """
    base_dir: str = "./data/binance_futures/book_snapshot"
    # Периодичность записи keyframe
    keyframe_interval_sec: float = 30.0
    # Глубина снимка (сколько уровней)
    depth_levels: int = 50
    # Писать keyframe при старте
    write_on_start: bool = True
    # Писать keyframe после resync стакана
    write_on_resync: bool = True
    # Писать keyframe при out_of_sync
    write_on_out_of_sync: bool = False  # при out_of_sync данные невалидны
    # Писать keyframe при смене UTC-дня
    write_on_day_change: bool = True


# ============================================================
# Протокол для стакана (duck typing)
# ============================================================

class OrderBookLike(Protocol):
    """Минимальный интерфейс FastOrderBook."""
    symbol: str
    tick_size: float

    @property
    def initialized(self) -> bool: ...

    @property
    def out_of_sync(self) -> bool: ...

    @property
    def last_update_id(self) -> int: ...

    @property
    def last_update_ts_ns(self) -> int: ...

    def top_bids(self, levels: int) -> List[tuple]: ...

    def top_asks(self, levels: int) -> List[tuple]: ...


# ============================================================
# Основной класс
# ============================================================

class KeyframeWriter:
    """
    Управляет записью keyframes для набора символов.

    Использование:
        writer = KeyframeWriter(config)
        await writer.start(recorder_or_writer)

        # Регистрация стаканов
        writer.register_book("BTCUSDT", fast_order_book)

        # Периодический вызов (например, раз в секунду из главного цикла)
        await writer.tick(now_ns=time.time_ns())

        # Принудительный keyframe (после resync)
        await writer.force_keyframe("BTCUSDT", reason="resync")

        await writer.close()

    Модуль НЕ владеет записью на диск — он формирует
    BookSnapshotEvent и передаёт его в переданный writer.
    Это позволяет переиспользовать существующий MarketDataRecorder.
    """

    def __init__(self, config: Optional[KeyframeConfig] = None) -> None:
        self._config = config or KeyframeConfig()
        self._books: Dict[str, OrderBookLike] = {}

        # Последнее время keyframe по каждому символу
        self._last_keyframe_ts_ns: Dict[str, int] = {}
        # Последний известный день (UTC) по символу
        self._last_day_str: Dict[str, str] = {}
        # Последний write_id (для отслеживания resync)
        self._last_write_id: Dict[str, int] = {}

        # Ссылка на writer (метод write_snapshot / record_book_snapshot)
        self._snapshot_writer: Optional[Any] = None

        # Счётчики
        self.total_keyframes: int = 0

    # ============================================
    # Регистрация
    # ============================================

    def register_book(self, symbol: str, book: OrderBookLike) -> None:
        """Регистрирует стакан для периодических keyframes."""
        self._books[symbol.upper()] = book

    def unregister_book(self, symbol: str) -> None:
        self._books.pop(symbol.upper(), None)

    # ============================================
    # Жизненный цикл
    # ============================================

    async def start(self, snapshot_writer: Any) -> None:
        """
        Инициализирует writer.

        snapshot_writer — объект с async методом:
            record_book_snapshot(event: BookSnapshotEvent) -> None
        Например, MarketDataRecorder из delta_writer.py.
        """
        self._snapshot_writer = snapshot_writer

        if self._config.write_on_start:
            for symbol in list(self._books.keys()):
                await self._write_keyframe(
                    symbol, reason="start", force=True
                )

    async def close(self) -> None:
        """Завершение — записываем финальный keyframe по каждому символу."""
        # Финальный keyframe полезен для полного replay
        for symbol in list(self._books.keys()):
            try:
                await self._write_keyframe(
                    symbol, reason="close", force=True
                )
            except Exception:
                pass

    # ============================================
    # Основной цикл
    # ============================================

    async def tick(self, now_ns: Optional[int] = None) -> None:
        """
        Вызывается периодически (например, раз в секунду).

        Проверяет каждый стакан:
        - прошёл ли интервал keyframe
        - изменился ли UTC-день
        - был ли resync (по last_update_id)
        """
        if now_ns is None:
            now_ns = time.time_ns()

        interval_ns = int(self._config.keyframe_interval_sec * 1_000_000_000)

        for symbol, book in list(self._books.items()):
            # Проверяем валидность стакана
            if not book.initialized:
                continue

            # Не пишем при рассинхроне, если не включено явно
            if book.out_of_sync and not self._config.write_on_out_of_sync:
                continue

            now_day = time.strftime("%Y-%m-%d", time.gmtime(now_ns / 1_000_000_000))
            last_day = self._last_day_str.get(symbol)
            day_changed = (
                self._config.write_on_day_change
                and last_day is not None
                and now_day != last_day
            )

            # Проверяем resync: если last_update_id сбросился или уменьшился
            resync_detected = False
            if self._config.write_on_resync:
                last_id = self._last_write_id.get(symbol)
                current_id = book.last_update_id
                if last_id is not None and current_id < last_id:
                    resync_detected = True
                elif last_id is None:
                    # первый keyframe
                    resync_detected = True

            # Проверяем интервал
            last_kf_ts = self._last_keyframe_ts_ns.get(symbol, 0)
            interval_passed = (now_ns - last_kf_ts) >= interval_ns

            if day_changed:
                await self._write_keyframe(symbol, reason="day_change", force=True)
            elif resync_detected:
                await self._write_keyframe(symbol, reason="resync", force=True)
            elif interval_passed:
                await self._write_keyframe(symbol, reason="interval", force=False)

    async def force_keyframe(
        self,
        symbol: str,
        reason: str = "manual",
    ) -> None:
        """Принудительная запись keyframe."""
        await self._write_keyframe(symbol, reason=reason, force=True)

    # ============================================
    # Внутреннее
    # ============================================

    async def _write_keyframe(
        self,
        symbol: str,
        reason: str,
        force: bool = False,
    ) -> None:
        if self._snapshot_writer is None:
            return

        book = self._books.get(symbol.upper())
        if book is None:
            return

        if not book.initialized:
            return

        if book.out_of_sync and not self._config.write_on_out_of_sync:
            return

        # Извлекаем top-N
        levels = self._config.depth_levels
        try:
            bids = book.top_bids(levels)
            asks = book.top_asks(levels)
        except Exception:
            return

        if not bids or not asks:
            return

        now_ns = time.time_ns()

        # Собираем BookSnapshotEvent
        from proscalper.core.events import BookLevel

        event = BookSnapshotEvent(
            ts_exchange_ns=book.last_update_ts_ns or now_ns,
            ts_local_ns=now_ns,
            symbol=symbol.upper(),
            bids=[BookLevel(price=float(p), quantity=float(q)) for p, q in bids],
            asks=[BookLevel(price=float(p), quantity=float(q)) for p, q in asks],
            first_update_id=book.last_update_id,
            last_update_id=book.last_update_id,
        )

        # Пишем через внешний writer
        try:
            record = getattr(self._snapshot_writer, "record_book_snapshot", None)
            if record is not None:
                await record(event)
        except Exception:
            pass

        # Обновляем состояние
        self._last_keyframe_ts_ns[symbol.upper()] = now_ns
        self._last_day_str[symbol.upper()] = time.strftime(
            "%Y-%m-%d",
            time.gmtime(now_ns / 1_000_000_000),
        )
        self._last_write_id[symbol.upper()] = book.last_update_id
        self.total_keyframes += 1

    # ============================================
    # Диагностика
    # ============================================

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_keyframes": self.total_keyframes,
            "registered_symbols": list(self._books.keys()),
            "config_interval_sec": self._config.keyframe_interval_sec,
        }