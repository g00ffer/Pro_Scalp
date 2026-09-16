"""
Быстрый локальный стакан для скальпинга.

Хранение:
- цены как integer ticks;
- SortedDict для быстрого доступа к лучшим уровням;
- дельты применяются батчами;
- есть защита от нарушения последовательности update id.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    from sortedcontainers import SortedDict
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Нужно установить sortedcontainers: pip install sortedcontainers"
    ) from exc

from proscalper.core.events import (
    BookDeltaBatchEvent,
    BookLevel,
    BookSnapshotEvent,
)
from proscalper.core.types import BookSide


class FastOrderBook:
    """
    Локальный стакан с быстрой синхронизацией.

    Особенности:
    - ключ уровня: integer tick;
    - best bid/ask берутся за O(1);
    - дельты буферизуются до snapshot;
    - при рассинхронизации требуется новый snapshot.
    """

    def __init__(
        self,
        symbol: str,
        tick_size: float,
        max_pending_batches: int = 2000,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size должен быть положительным")

        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self._inv_tick_size = 1.0 / tick_size
        self._max_pending_batches = max_pending_batches

        self._bids: SortedDict[int, float] = SortedDict()
        self._asks: SortedDict[int, float] = SortedDict()

        self._initialized = False
        self._out_of_sync = False
        self._out_of_sync_reason: Optional[str] = None

        self._last_update_id = 0
        self._pending_batches: List[BookDeltaBatchEvent] = []

        self._last_update_ts_ns = 0
        self._last_local_ts_ns = 0

    # =========================
    # Properties
    # =========================

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def out_of_sync(self) -> bool:
        return self._out_of_sync

    @property
    def out_of_sync_reason(self) -> Optional[str]:
        return self._out_of_sync_reason

    @property
    def requires_resync(self) -> bool:
        return not self._initialized or self._out_of_sync

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def last_update_ts_ns(self) -> int:
        return self._last_update_ts_ns

    @property
    def best_bid(self) -> Optional[Tuple[float, float]]:
        """
        Лучший bid: (price, quantity).
        """
        if not self._bids:
            return None

        tick, qty = self._bids.peekitem(-1)
        price = self._ticks_to_price(tick)
        return price, qty

    @property
    def best_ask(self) -> Optional[Tuple[float, float]]:
        """
        Лучший ask: (price, quantity).
        """
        if not self._asks:
            return None

        tick, qty = self._asks.peekitem(0)
        price = self._ticks_to_price(tick)
        return price, qty

    @property
    def spread(self) -> Optional[float]:
        best_bid = self.best_bid
        best_ask = self.best_ask

        if best_bid is None or best_ask is None:
            return None

        return best_ask[0] - best_bid[0]

    @property
    def spread_ticks(self) -> Optional[int]:
        spread = self.spread
        if spread is None:
            return None
        return int(round(spread * self._inv_tick_size))

    @property
    def mid_price(self) -> Optional[float]:
        best_bid = self.best_bid
        best_ask = self.best_ask

        if best_bid is None or best_ask is None:
            return None

        return (best_bid[0] + best_ask[0]) / 2.0

    # =========================
    # Public API
    # =========================

    def reset(self) -> None:
        """
        Полный сброс состояния.
        """
        self._bids.clear()
        self._asks.clear()
        self._initialized = False
        self._out_of_sync = False
        self._out_of_sync_reason = None
        self._last_update_id = 0
        self._pending_batches.clear()
        self._last_update_ts_ns = 0
        self._last_local_ts_ns = 0

    def apply_snapshot(self, snapshot: BookSnapshotEvent) -> bool:
        """
        Применить полный срез стакана.

        Если до снапшота уже пришли дельты, они буферизуются
        и применяются сразу после снапшота.
        """
        if snapshot.symbol.upper() != self.symbol:
            return False

        # Сохраняем дельты, которые могли прийти до снапшота.
        pending = self._pending_batches

        # Очищаем текущее состояние.
        self._bids.clear()
        self._asks.clear()
        self._initialized = False
        self._out_of_sync = False
        self._out_of_sync_reason = None
        self._last_update_id = 0
        self._pending_batches.clear()

        # Применяем снапшот.
        for level in snapshot.bids:
            self._update_level(BookSide.BID, level.price, level.quantity)

        for level in snapshot.asks:
            self._update_level(BookSide.ASK, level.price, level.quantity)

        self._last_update_id = snapshot.last_update_id
        self._last_update_ts_ns = snapshot.ts_exchange_ns
        self._last_local_ts_ns = snapshot.ts_local_ns
        self._initialized = True

        # Применяем буферизированные дельты.
        for delta_batch in pending:
            if not self._apply_delta_batch_internal(delta_batch):
                return False

        return True

    def apply_delta_batch(self, delta_batch: BookDeltaBatchEvent) -> bool:
        """
        Применить батч изменений стакана.
        """
        if delta_batch.symbol.upper() != self.symbol:
            return False

        if not self._initialized:
            if len(self._pending_batches) >= self._max_pending_batches:
                self._mark_out_of_sync("PENDING_BATCH_OVERFLOW")
                return False

            self._pending_batches.append(delta_batch)
            return True

        if self._out_of_sync:
            return False

        return self._apply_delta_batch_internal(delta_batch)

    def top_bids(self, levels: int = 20) -> List[Tuple[float, float]]:
        """
        Топовые биды: [(price, quantity), ...]
        """
        result: List[Tuple[float, float]] = []
        count = min(levels, len(self._bids))

        for i in range(count):
            tick, qty = self._bids.peekitem(-i - 1)
            price = self._ticks_to_price(tick)
            result.append((price, qty))

        return result

    def top_asks(self, levels: int = 20) -> List[Tuple[float, float]]:
        """
        Топовые аски: [(price, quantity), ...]
        """
        result: List[Tuple[float, float]] = []
        count = min(levels, len(self._asks))

        for i in range(count):
            tick, qty = self._asks.peekitem(i)
            price = self._ticks_to_price(tick)
            result.append((price, qty))

        return result

    def top_depth_notional(
        self,
        side: BookSide,
        levels: int = 20,
    ) -> float:
        """
        Суммарный номинал в топ-N уровнях.
        """
        if side == BookSide.BID:
            top_levels = self.top_bids(levels)
        else:
            top_levels = self.top_asks(levels)

        return sum(price * qty for price, qty in top_levels)

    def lag_ms(self, now_ms: int) -> int:
        """
        Насколько стакан отстал по локальному времени.
        """
        if self._last_local_ts_ns == 0:
            return 999_999

        local_update_ms = self._last_local_ts_ns // 1_000_000
        return max(0, now_ms - local_update_ms)

    # =========================
    # Internal
    # =========================

    def _apply_delta_batch_internal(
        self,
        delta_batch: BookDeltaBatchEvent,
    ) -> bool:
        """
        Внутреннее применение батча дельт.
        """
        if delta_batch.symbol.upper() != self.symbol:
            return False

        # Старые события игнорируем.
        if delta_batch.last_update_id <= self._last_update_id:
            return True

        if not self._is_sequence_valid(delta_batch):
            self._mark_out_of_sync("SEQUENCE_MISMATCH")
            return False

        for level in delta_batch.bids:
            self._update_level(BookSide.BID, level.price, level.quantity)

        for level in delta_batch.asks:
            self._update_level(BookSide.ASK, level.price, level.quantity)

        self._last_update_id = delta_batch.last_update_id
        self._last_update_ts_ns = delta_batch.ts_exchange_ns
        self._last_local_ts_ns = delta_batch.ts_local_ns

        return True

    def _is_sequence_valid(self, delta_batch: BookDeltaBatchEvent) -> bool:
        """
        Проверка последовательности Binance Futures depthUpdate.

        После снапшота первое событие может пересекаться:
            U <= lastUpdateId + 1 <= u

        Дальше ожидаем:
            pu == previous lastUpdateId
        """
        if delta_batch.prev_update_id == self._last_update_id:
            return True

        if (
            delta_batch.first_update_id <= self._last_update_id + 1
            <= delta_batch.last_update_id
        ):
            return True

        return False

    def _update_level(
        self,
        side: BookSide,
        price: float,
        quantity: float,
    ) -> None:
        """
        Обновить один уровень стакана.
        """
        tick = self._price_to_ticks(price)

        if side == BookSide.BID:
            book = self._bids
        else:
            book = self._asks

        if quantity <= 0.0:
            book.pop(tick, None)
        else:
            book[tick] = quantity

    def _price_to_ticks(self, price: float) -> int:
        return int(round(price * self._inv_tick_size))

    def _ticks_to_price(self, tick: int) -> float:
        return round(tick * self.tick_size, 12)

    def _mark_out_of_sync(self, reason: str) -> None:
        self._out_of_sync = True
        self._out_of_sync_reason = reason