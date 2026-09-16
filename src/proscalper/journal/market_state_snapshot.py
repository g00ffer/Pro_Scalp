"""
Структура снимка состояния рынка (market state snapshot).

Используется во всех журналах (Decision Journal, Incident Journal)
и в логировании решений. Хранит полный контекст момента:
- стакан (bid/ask, top-N)
- лента (дельта, объём, скорость)
- уровень (id, цена, сила, касания)
- ликвидность (стены, дисбаланс, спуфинг, айсберги)
- тайминги (exchange, local, lag)

Принцип: снимок неизменяемый (frozen) и самодостаточный.
По нему можно полностью восстановить картину, при которой было
принято решение.

Источник данных:
- FastOrderBook → book
- TapeAnalyzer → tape
- Level + LevelTracker → level
- BookAnalyzer + WallRegistry + SpoofDetector + IcebergDetector → liquidity
- BookTickerGuard → fast_guard
- Timestamps → timings
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import msgspec


# ============================================================
# Секции снимка (frozen — только для чтения)
# ============================================================

class BookSnapshotSection(msgspec.Struct, frozen=True):
    """
    Секция снимка стакана.

    Хранит top-N bid/ask и агрегированные метрики на момент снимка.
    """
    ts_ns: int
    best_bid: float
    best_bid_qty: float
    best_ask: float
    best_ask_qty: float
    spread_ticks: int
    mid_price: float
    top_bids: List[Tuple[float, float]] = field(default_factory=list)
    top_asks: List[Tuple[float, float]] = field(default_factory=list)
    top_depth_notional: float = 0.0
    is_initialized: bool = False
    out_of_sync: bool = False


class TapeSnapshotSection(msgspec.Struct, frozen=True):
    """
    Секция снимка ленты сделок.

    Все метрики rolling-окон (см. market_data/tape.py).
    """
    ts_ns: int
    trades_per_sec: float = 0.0
    buy_volume_1s: float = 0.0
    sell_volume_1s: float = 0.0
    net_delta_1s: float = 0.0
    total_volume_1s: float = 0.0
    volume_imbalance_1s: float = 0.0
    aggressor_volume_1s: float = 0.0
    last_price: float = 0.0
    price_change_1s: float = 0.0
    trades_per_sec_median: float = 0.0
    buy_volume_median_1s: float = 0.0
    sell_volume_median_1s: float = 0.0


class LevelSnapshotSection(msgspec.Struct, frozen=True):
    """
    Секция снимка уровня.

    Хранит статические и полустатические атрибуты уровня,
    на котором основан сигнал.
    """
    level_id: str
    symbol: str
    side: str                       # "SUPPORT" | "RESISTANCE"
    state: str                      # "FORMING" | "ACTIVE" | ...
    price: float
    zone_lower: float
    zone_upper: float
    strength: float = 0.0
    touches: int = 0
    age_sec: float = 0.0
    round_bonus: float = 0.0
    fakeout_count: int = 0


class LiquiditySnapshotSection(msgspec.Struct, frozen=True):
    """
    Секция снимка ликвидности вокруг уровня.

    Агрегирует данные BookAnalyzer, WallRegistry, SpoofDetector,
    IcebergDetector, ImbalanceCalculator.
    """
    ts_ns: int
    imbalance: float = 0.0
    weighted_imbalance: float = 0.0

    bid_wall_price: Optional[float] = None
    bid_wall_notional: float = 0.0
    bid_wall_consumed: bool = False
    bid_wall_spoof_score: float = 0.0
    bid_wall_iceberg_score: float = 0.0

    ask_wall_price: Optional[float] = None
    ask_wall_notional: float = 0.0
    ask_wall_consumed: bool = False
    ask_wall_spoof_score: float = 0.0
    ask_wall_iceberg_score: float = 0.0

    cross_confirmation_score: float = 0.0
    cross_spoof_penalty: float = 0.0


class TimingsSnapshotSection(msgspec.Struct, frozen=True):
    """
    Секция таймингов.

    Все значения в миллисекундах/наносекундах для точной диагностики.
    """
    signal_created_ts_ns: int = 0
    decision_ts_ns: int = 0
    order_submit_ts_ns: int = 0
    exchange_ack_ts_ns: int = 0
    fill_ts_ns: int = 0

    book_lag_ms: int = 0
    bookticker_age_ms: int = 0
    signal_processing_ms: float = 0.0
    order_queue_latency_ms: float = 0.0
    exchange_ack_latency_ms: float = 0.0
    fill_latency_ms: float = 0.0


# ============================================================
# Объединённый снимок
# ============================================================

class MarketStateSnapshot(msgspec.Struct, frozen=True):
    """
    Полный снимок состояния рынка в момент решения.

    Все секции frozen (immutable). После создания объект
    безопасен для параллельного чтения.
    """
    snapshot_id: str
    symbol: str
    ts_ns: int

    book: Optional[BookSnapshotSection] = None
    tape: Optional[TapeSnapshotSection] = None
    level: Optional[LevelSnapshotSection] = None
    liquidity: Optional[LiquiditySnapshotSection] = None
    timings: Optional[TimingsSnapshotSection] = None

    # Дополнительный произвольный контекст (reason codes и т.п.)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """
        Преобразование в плоский dict для JSONL-записи.

        Сохраняет вложенность секций для простоты разбора.
        """
        return {
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "ts_ns": self.ts_ns,
            "book": _section_to_dict(self.book),
            "tape": _section_to_dict(self.tape),
            "level": _section_to_dict(self.level),
            "liquidity": _section_to_dict(self.liquidity),
            "timings": _section_to_dict(self.timings),
            "extra": dict(self.extra) if self.extra else {},
        }


# ============================================================
# Билдер — упрощает создание снимков из живых объектов
# ============================================================

class MarketStateSnapshotBuilder:
    """
    Билдер снимка из живых объектов системы.

    Использование:
        builder = MarketStateSnapshotBuilder(snapshot_id_factory)
        snapshot = builder.build(
            symbol="BTCUSDT",
            book=fast_order_book,
            tape=tape_analyzer,
            level=level,
            level_age_fn=lambda lvl: lvl.age_sec(now_ns),
            liquidity=level_liquidity,
            cross_analysis=cross_market_analysis,
            timings=TimingsSnapshotSection(...),
            extra={"reason_codes": ["LEVEL_ACTIVE", "COMPRESSION"]},
        )

    Принцип: builder НЕ хранит состояние, только собирает.
    Все вычисления — простые преобразования.
    """

    def __init__(self, snapshot_id_factory: callable) -> None:
        """
        snapshot_id_factory() → str
        Например, lambda: uuid.uuid4().hex
        """
        self._id_factory = snapshot_id_factory

    def build(
        self,
        symbol: str,
        book: Any = None,
        tape: Any = None,
        level: Any = None,
        liquidity: Any = None,
        cross_analysis: Any = None,
        timings: Optional[TimingsSnapshotSection] = None,
        level_age_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        ts_ns: Optional[int] = None,
    ) -> MarketStateSnapshot:
        """Собирает полный снимок."""
        now_ns = ts_ns or time.time_ns()

        book_section = self._build_book(book)
        tape_section = self._build_tape(tape)
        level_section = self._build_level(level, level_age_sec, now_ns)
        liquidity_section = self._build_liquidity(liquidity, cross_analysis, now_ns)

        return MarketStateSnapshot(
            snapshot_id=self._id_factory(),
            symbol=symbol.upper(),
            ts_ns=now_ns,
            book=book_section,
            tape=tape_section,
            level=level_section,
            liquidity=liquidity_section,
            timings=timings,
            extra=extra or {},
        )

    # --------------------------------------------------------
    # Внутренние строители секций
    # --------------------------------------------------------

    def _build_book(self, book: Any) -> Optional[BookSnapshotSection]:
        if book is None:
            return None

        # Работаем с duck typing: ожидаем атрибуты FastOrderBook
        best_bid = getattr(book, "best_bid", None)
        best_ask = getattr(book, "best_ask", None)

        bid_price, bid_qty = (best_bid if best_bid else (0.0, 0.0))
        ask_price, ask_qty = (best_ask if best_ask else (0.0, 0.0))

        mid_price = getattr(book, "mid_price", None) or 0.0
        spread_ticks = getattr(book, "spread_ticks", None) or 0

        try:
            top_bids = book.top_bids(20)
            top_asks = book.top_asks(20)
        except Exception:
            top_bids = []
            top_asks = []

        # top_depth_notional: считаем суммой через метод
        try:
            # Пытаемся взять у самого объекта, если есть
            top_depth_notional = (
                book.top_depth_notional(book._bids, 20)  # type: ignore
                if False else sum(p * q for p, q in top_bids + top_asks)
            )
        except Exception:
            top_depth_notional = sum(p * q for p, q in top_bids + top_asks)

        return BookSnapshotSection(
            ts_ns=getattr(book, "last_update_ts_ns", 0) or 0,
            best_bid=float(bid_price),
            best_bid_qty=float(bid_qty),
            best_ask=float(ask_price),
            best_ask_qty=float(ask_qty),
            spread_ticks=int(spread_ticks),
            mid_price=float(mid_price),
            top_bids=[(float(p), float(q)) for p, q in top_bids],
            top_asks=[(float(p), float(q)) for p, q in top_asks],
            top_depth_notional=float(top_depth_notional),
            is_initialized=bool(getattr(book, "initialized", False)),
            out_of_sync=bool(getattr(book, "out_of_sync", False)),
        )

    def _build_tape(self, tape: Any) -> Optional[TapeSnapshotSection]:
        if tape is None:
            return None

        # Ожидаем объект с методами snapshot() или готовые атрибуты
        try:
            metrics = tape.snapshot() if hasattr(tape, "snapshot") else tape
        except Exception:
            return None

        return TapeSnapshotSection(
            ts_ns=getattr(metrics, "ts_ns", 0) or 0,
            trades_per_sec=float(getattr(metrics, "trades_per_sec_1s", 0.0) or 0.0),
            buy_volume_1s=float(getattr(metrics, "buy_volume_1s", 0.0) or 0.0),
            sell_volume_1s=float(getattr(metrics, "sell_volume_1s", 0.0) or 0.0),
            net_delta_1s=float(getattr(metrics, "net_delta_1s", 0.0) or 0.0),
            total_volume_1s=float(
                (getattr(metrics, "buy_volume_1s", 0.0) or 0.0)
                + (getattr(metrics, "sell_volume_1s", 0.0) or 0.0)
            ),
            volume_imbalance_1s=float(
                getattr(metrics, "volume_imbalance_1s", 0.0) or 0.0
            ),
            aggressor_volume_1s=float(
                getattr(metrics, "aggressor_volume_1s", 0.0) or 0.0
            ),
            last_price=float(getattr(metrics, "last_price", 0.0) or 0.0),
            price_change_1s=float(
                getattr(metrics, "price_change_1s", 0.0) or 0.0
            ),
            trades_per_sec_median=float(
                getattr(metrics, "trades_per_sec_median", 0.0) or 0.0
            ),
            buy_volume_median_1s=float(
                getattr(metrics, "buy_volume_median_1s", 0.0) or 0.0
            ),
            sell_volume_median_1s=float(
                getattr(metrics, "sell_volume_median_1s", 0.0) or 0.0
            ),
        )

    def _build_level(
        self,
        level: Any,
        age_sec: Optional[float],
        now_ns: int,
    ) -> Optional[LevelSnapshotSection]:
        if level is None:
            return None

        # side может быть enum с .name или .value
        side = getattr(level, "side", None)
        side_str = getattr(side, "name", None) or str(side)

        state = getattr(level, "state", None)
        state_str = getattr(state, "name", None) or str(state)

        zone_lower = getattr(level, "zone_lower", 0.0)
        zone_upper = getattr(level, "zone_upper", 0.0)

        if age_sec is None:
            try:
                age_sec = level.age_sec(now_ns)
            except Exception:
                age_sec = 0.0

        return LevelSnapshotSection(
            level_id=str(getattr(level, "id", "")),
            symbol=str(getattr(level, "symbol", "")),
            side=side_str,
            state=state_str,
            price=float(getattr(level, "center", 0.0)),
            zone_lower=float(zone_lower),
            zone_upper=float(zone_upper),
            strength=float(getattr(level, "strength", 0.0) or 0.0),
            touches=int(getattr(level, "touches", 0) or 0),
            age_sec=float(age_sec or 0.0),
            round_bonus=float(getattr(level, "round_bonus", 0.0) or 0.0),
            fakeout_count=int(getattr(level, "fakeout_count", 0) or 0),
        )

    def _build_liquidity(
        self,
        liquidity: Any,
        cross_analysis: Any,
        now_ns: int,
    ) -> Optional[LiquiditySnapshotSection]:
        # Может быть None, если BookAnalyzer не заполнен
        if liquidity is None and cross_analysis is None:
            return None

        section = LiquiditySnapshotSection(ts_ns=now_ns)

        if liquidity is not None:
            # Извлекаем поля через getattr (duck typing)
            bid_wall = getattr(liquidity, "bid_wall", None)
            ask_wall = getattr(liquidity, "ask_wall", None)

            section = LiquiditySnapshotSection(
                ts_ns=now_ns,
                imbalance=float(getattr(liquidity, "imbalance", 0.0) or 0.0),
                weighted_imbalance=float(
                    getattr(liquidity, "weighted_imbalance", 0.0) or 0.0
                ),
                bid_wall_price=(
                    float(bid_wall.price) if bid_wall is not None else None
                ),
                bid_wall_notional=float(
                    getattr(bid_wall, "current_notional", 0.0) or 0.0
                ) if bid_wall is not None else 0.0,
                bid_wall_consumed=bool(
                    getattr(liquidity, "bid_wall_consumed", False)
                ),
                bid_wall_spoof_score=float(
                    getattr(bid_wall, "spoof_score", 0.0) or 0.0
                ) if bid_wall is not None else 0.0,
                bid_wall_iceberg_score=float(
                    getattr(bid_wall, "iceberg_score", 0.0) or 0.0
                ) if bid_wall is not None else 0.0,
                ask_wall_price=(
                    float(ask_wall.price) if ask_wall is not None else None
                ),
                ask_wall_notional=float(
                    getattr(ask_wall, "current_notional", 0.0) or 0.0
                ) if ask_wall is not None else 0.0,
                ask_wall_consumed=bool(
                    getattr(liquidity, "ask_wall_consumed", False)
                ),
                ask_wall_spoof_score=float(
                    getattr(ask_wall, "spoof_score", 0.0) or 0.0
                ) if ask_wall is not None else 0.0,
                ask_wall_iceberg_score=float(
                    getattr(ask_wall, "iceberg_score", 0.0) or 0.0
                ) if ask_wall is not None else 0.0,
            )

        if cross_analysis is not None:
            # Иммутабельный Struct — создаём копию с добавлением полей
            section = LiquiditySnapshotSection(
                ts_ns=section.ts_ns,
                imbalance=section.imbalance,
                weighted_imbalance=section.weighted_imbalance,
                bid_wall_price=section.bid_wall_price,
                bid_wall_notional=section.bid_wall_notional,
                bid_wall_consumed=section.bid_wall_consumed,
                bid_wall_spoof_score=section.bid_wall_spoof_score,
                bid_wall_iceberg_score=section.bid_wall_iceberg_score,
                ask_wall_price=section.ask_wall_price,
                ask_wall_notional=section.ask_wall_notional,
                ask_wall_consumed=section.ask_wall_consumed,
                ask_wall_spoof_score=section.ask_wall_spoof_score,
                ask_wall_iceberg_score=section.ask_wall_iceberg_score,
                cross_confirmation_score=float(
                    getattr(cross_analysis, "cross_confirmation_score", 0.0) or 0.0
                ),
                cross_spoof_penalty=float(
                    getattr(cross_analysis, "spoof_penalty", 0.0) or 0.0
                ),
            )

        return section


# ============================================================
# Утилиты
# ============================================================

def _section_to_dict(section: Any) -> Optional[Dict[str, Any]]:
    """Плоское представление секции снимка для JSONL."""
    if section is None:
        return None

    # msgspec.Struct → dict
    try:
        return msgspec.structs.asdict(section)
    except Exception:
        # Fallback: ручной обход
        result: Dict[str, Any] = {}
        for name in dir(section):
            if name.startswith("_"):
                continue
            value = getattr(section, name)
            if callable(value):
                continue
            result[name] = value
        return result


def make_timings_section(
    signal_created_ts_ns: int = 0,
    decision_ts_ns: int = 0,
    order_submit_ts_ns: int = 0,
    exchange_ack_ts_ns: int = 0,
    fill_ts_ns: int = 0,
    book_lag_ms: int = 0,
    bookticker_age_ms: int = 0,
) -> TimingsSnapshotSection:
    """
    Утилита для быстрого создания секции таймингов.

    Автоматически считает производные задержки (processing, ack, fill).
    """
    def _delta_ms(a: int, b: int) -> float:
        if a <= 0 or b <= 0:
            return 0.0
        return (b - a) / 1_000_000

    return TimingsSnapshotSection(
        signal_created_ts_ns=signal_created_ts_ns,
        decision_ts_ns=decision_ts_ns,
        order_submit_ts_ns=order_submit_ts_ns,
        exchange_ack_ts_ns=exchange_ack_ts_ns,
        fill_ts_ns=fill_ts_ns,
        book_lag_ms=book_lag_ms,
        bookticker_age_ms=bookticker_age_ms,
        signal_processing_ms=_delta_ms(signal_created_ts_ns, decision_ts_ns),
        exchange_ack_latency_ms=_delta_ms(order_submit_ts_ns, exchange_ack_ts_ns),
        fill_latency_ms=_delta_ms(order_submit_ts_ns, fill_ts_ns),
    )