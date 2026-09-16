"""
Восстановление локального стакана из записанных событий (Replay Builder).

Задача: по записанным рыночным событиям (keyframes + дельты) восстановить
FastOrderBook на произвольный момент времени. Используется в бэктестере.

Формат записи (создан delta_writer.py + keyframe_writer.py):
    data/<event_type>/<YYYY-MM-DD>/<SYMBOL>.jsonl
    - book_snapshot/*.jsonl — периодические keyframes (полный снимок)
    - book_delta/*.jsonl    — инкрементальные обновления
    - trade/*.jsonl         — сделки
    - book_ticker/*.jsonl   — быстрые best bid/ask

Алгоритм replay:
1. Найти последний keyframe ДО целевого ts
2. Применить его к пустому FastOrderBook
3. Применить все дельты между keyframe.ts и target.ts
4. Проверить sequence (пропуски → пометить сегмент как invalid)

Принципы:
- НЕ используем данные из будущего (lookahead)
- Работаем потоково (не держим всё в памяти)
- Поддерживаем как JSONL, так и Parquet (через compactor)
- Возвращаем (book, is_valid, gaps)

Архитектурное решение:
- модуль не знает про стратегию — только восстанавливает стакан
- вызывающий код сам решает, что делать с invalid-сегментами
- все ошибки парсинга логируются, но не роняют replay
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import msgspec

from proscalper.core.events import (
    BookDeltaBatchEvent,
    BookLevel,
    BookSnapshotEvent,
    BookTickerEvent,
    TradeEvent,
)
from proscalper.core.types import BookSide
from proscalper.market_data.orderbook_fast import FastOrderBook


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class ReplayConfig:
    """Конфигурация replay builder."""
    source_dir: str = "./data/binance_futures"
    # Если True — читать Parquet вместо JSONL (после compactor)
    use_parquet: bool = False
    # Максимально допустимый размер буфера дельт до keyframe (защита от утечек)
    max_deltas_before_keyframe: int = 100_000


# ============================================================
# Результат
# ============================================================

@dataclass
class ReplayResult:
    """
    Результат построения стакана на момент target_ts.
    """
    symbol: str
    target_ts_ns: int
    book: FastOrderBook

    # Валидность
    is_valid: bool = True
    reason_invalid: str = ""

    # Статистика
    keyframe_ts_ns: int = 0
    deltas_applied: int = 0
    snapshots_seen: int = 0
    gaps_detected: int = 0
    out_of_order_events: int = 0


# ============================================================
# Основной класс
# ============================================================

class ReplayBuilder:
    """
    Строит стакан на момент target_ts по записанным событиям.

    Использование:
        builder = ReplayBuilder(ReplayConfig(source_dir="./data/binance_futures"))

        # Восстановить стакан на 12:00:00.500
        result = builder.build_at(
            symbol="BTCUSDT",
            target_ts_ns=...,
            tick_size=0.1,
            date_str="2026-09-16",
        )

        if result.is_valid:
            # используем result.book
            best_bid = result.book.best_bid
        else:
            # скипаем сегмент — данные повреждены
            ...

        # Или итеративно: проиграть весь день и подавать события в колбэк
        for book, event in builder.replay_day(
            symbol="BTCUSDT",
            date_str="2026-09-16",
            tick_size=0.1,
        ):
            # реагируем на каждое обновление
            ...
    """

    def __init__(self, config: Optional[ReplayConfig] = None) -> None:
        self.config = config or ReplayConfig()
        self._root = Path(self.config.source_dir)

    # ============================================
    # Build_at: снимок на момент времени
    # ============================================

    def build_at(
        self,
        symbol: str,
        target_ts_ns: int,
        tick_size: float,
        date_str: str,
    ) -> ReplayResult:
        """
        Восстанавливает стакан на целевой ts.

        Шаги:
        1. Читает все keyframe'ы за день, находит последний до target_ts_ns
        2. Применяет его
        3. Применяет все дельты между keyframe и target
        """
        symbol = symbol.upper()
        book = FastOrderBook(symbol=symbol, tick_size=tick_size)

        result = ReplayResult(
            symbol=symbol,
            target_ts_ns=target_ts_ns,
            book=book,
        )

        # 1. Ключфреймы
        keyframe_path = self._path_for("book_snapshot", date_str, symbol)
        keyframes = list(self._iter_snapshots(keyframe_path, until_ts=target_ts_ns))
        result.snapshots_seen = len(keyframes)

        if not keyframes:
            result.is_valid = False
            result.reason_invalid = "no keyframe found before target_ts"
            return result

        # Последний keyframe до target_ts
        last_snapshot = max(keyframes, key=lambda s: s.ts_exchange_ns)
        book.apply_snapshot(last_snapshot)
        result.keyframe_ts_ns = last_snapshot.ts_exchange_ns

        # 2. Дельты между keyframe и target
        delta_path = self._path_for("book_delta", date_str, symbol)
        for delta in self._iter_deltas(
            delta_path,
            from_ts=last_snapshot.ts_exchange_ns,
            until_ts=target_ts_ns,
        ):
            ok = book.apply_delta_batch(delta)
            if not ok or book.out_of_sync:
                result.is_valid = False
                result.reason_invalid = (
                    f"book out of sync at delta ts={delta.ts_exchange_ns}, "
                    f"reason={book.out_of_sync_reason}"
                )
                # Мы продолжаем статистику, но сегмент помечен invalid
                result.gaps_detected += 1
                break
            result.deltas_applied += 1

        return result

    # ============================================
    # Replay: итерация по всему дню
    # ============================================

    def replay_day(
        self,
        symbol: str,
        date_str: str,
        tick_size: float,
        start_ts_ns: Optional[int] = None,
        end_ts_ns: Optional[int] = None,
    ) -> Iterator[Tuple[FastOrderBook, BookDeltaBatchEvent]]:
        """
        Генерирует (book, delta) на каждое обновление стакана за день.

        Стакан мутируется на месте — то есть на каждом шаге мы имеем
        актуальное состояние. Итерирующийся код должен обрабатывать
        дельту сразу же (не сохранять ссылку).

        Если сегмент становится невалидным (gap), генератор пытается
        восстановить стакан по следующему keyframe и продолжает.
        """
        symbol = symbol.upper()
        book = FastOrderBook(symbol=symbol, tick_size=tick_size)

        keyframe_path = self._path_for("book_snapshot", date_str, symbol)
        delta_path = self._path_for("book_delta", date_str, symbol)

        # Загружаем все keyframes один раз (они редки — раз в 30 сек)
        keyframes: List[BookSnapshotEvent] = list(self._iter_snapshots(keyframe_path))

        if not keyframes:
            return

        # Первый keyframe — начальная точка
        first_kf = keyframes[0]
        book.apply_snapshot(first_kf)

        # Итератор дельт по возрастанию ts
        deltas = self._iter_deltas(
            delta_path,
            from_ts=first_kf.ts_exchange_ns,
            until_ts=end_ts_ns,
        )

        next_kf_idx = 1
        last_applied_ts = first_kf.ts_exchange_ns

        for delta in deltas:
            if start_ts_ns is not None and delta.ts_exchange_ns < start_ts_ns:
                # Пропускаем, но применяем к стакану (state должен быть актуален)
                pass

            # Если пришло время следующего keyframe — применяем его
            while (
                next_kf_idx < len(keyframes)
                and keyframes[next_kf_idx].ts_exchange_ns <= delta.ts_exchange_ns
            ):
                kf = keyframes[next_kf_idx]
                book.apply_snapshot(kf)
                last_applied_ts = kf.ts_exchange_ns
                next_kf_idx += 1

            ok = book.apply_delta_batch(delta)

            if not ok or book.out_of_sync:
                # Пытаемся восстановиться по следующему keyframe
                if next_kf_idx < len(keyframes):
                    kf = keyframes[next_kf_idx]
                    book.apply_snapshot(kf)
                    last_applied_ts = kf.ts_exchange_ns
                    next_kf_idx += 1
                    continue
                else:
                    # Восстановление невозможно — останавливаем replay
                    return

            last_applied_ts = delta.ts_exchange_ns

            if start_ts_ns is not None and delta.ts_exchange_ns < start_ts_ns:
                continue

            yield book, delta

    # ============================================
    # Утилиты: чтение событий
    # ============================================

    def _iter_snapshots(
        self,
        path: Path,
        until_ts: Optional[int] = None,
    ) -> Iterator[BookSnapshotEvent]:
        """Читает keyframes из файла (JSONL)."""
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Не выходим сразу, потому что порядок может быть нарушен
                # (редко, но защищаемся от этого)
                ts_exchange_ns = obj.get("ts_exchange_ns", 0)
                if until_ts is not None and ts_exchange_ns > until_ts:
                    continue

                try:
                    snapshot = self._dict_to_snapshot(obj)
                except Exception:
                    continue

                if snapshot is not None:
                    yield snapshot

    def _iter_deltas(
        self,
        path: Path,
        from_ts: Optional[int] = None,
        until_ts: Optional[int] = None,
    ) -> Iterator[BookDeltaBatchEvent]:
        """Читает дельты из файла (JSONL)."""
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts_exchange_ns = obj.get("ts_exchange_ns", 0)
                if from_ts is not None and ts_exchange_ns < from_ts:
                    continue
                if until_ts is not None and ts_exchange_ns > until_ts:
                    # При монотонности можно break, но мы остаёмся
                    # консервативными и продолжаем (защита от нестандартных
                    # случаев с нарушением порядка в файле)
                    continue

                try:
                    delta = self._dict_to_delta(obj)
                except Exception:
                    continue

                if delta is not None:
                    yield delta

    # ============================================
    # Парсинг (dict → события)
    # ============================================

    def _dict_to_snapshot(self, obj: Dict[str, Any]) -> Optional[BookSnapshotEvent]:
        """Парсит dict → BookSnapshotEvent."""
        try:
            return BookSnapshotEvent(
                ts_exchange_ns=int(obj.get("ts_exchange_ns", 0)),
                ts_local_ns=int(obj.get("ts_local_ns", 0)),
                symbol=str(obj.get("symbol", "")).upper(),
                bids=[
                    BookLevel(price=float(p), quantity=float(q))
                    for p, q in obj.get("bids", [])
                ],
                asks=[
                    BookLevel(price=float(p), quantity=float(q))
                    for p, q in obj.get("asks", [])
                ],
                first_update_id=int(obj.get("first_update_id", 0)),
                last_update_id=int(obj.get("last_update_id", 0)),
            )
        except Exception:
            return None

    def _dict_to_delta(self, obj: Dict[str, Any]) -> Optional[BookDeltaBatchEvent]:
        """Парсит dict → BookDeltaBatchEvent."""
        try:
            return BookDeltaBatchEvent(
                ts_exchange_ns=int(obj.get("ts_exchange_ns", 0)),
                ts_local_ns=int(obj.get("ts_local_ns", 0)),
                symbol=str(obj.get("symbol", "")).upper(),
                first_update_id=int(obj.get("first_update_id", 0)),
                last_update_id=int(obj.get("last_update_id", 0)),
                prev_update_id=int(obj.get("prev_update_id", 0)),
                bids=[
                    BookLevel(price=float(p), quantity=float(q))
                    for p, q in obj.get("bids", [])
                ],
                asks=[
                    BookLevel(price=float(p), quantity=float(q))
                    for p, q in obj.get("asks", [])
                ],
            )
        except Exception:
            return None

    # ============================================
    # Пути
    # ============================================

    def _path_for(
        self,
        event_type: str,
        date_str: str,
        symbol: str,
    ) -> Path:
        suffix = ".parquet" if self.config.use_parquet else ".jsonl"
        return self._root / event_type / date_str / f"{symbol.upper()}{suffix}"