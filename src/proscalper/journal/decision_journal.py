"""
Журнал решений (Decision Journal).

Пишет JSONL-записи для каждого значимого решения робота:
- сигналы (принятые и отклонённые)
- отправка ордера
- исполнение (fill)
- изменение состояния позиции (BE, trailing, partial close, close)
- аварийные действия

Каждая запись самодостаточна: содержит полный MarketStateSnapshot
на момент решения, reason codes и связанные идентификаторы.

Формат: JSONL по дням, чтобы удобно грузить в DuckDB/ClickHouse.
Файл: <base_dir>/<YYYY-MM-DD>.jsonl

Почему JSONL:
- append-only
- не блокирует hot path
- легко парсится и индексируется позже
- не требует схемы

Архитектурное решение:
- запись асинхронная, через очередь (по аналогии с delta_writer)
- не блокирует принятие решений
- при переполнении очереди — не теряем критические записи
  (см. drop policy ниже)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import msgspec

from proscalper.journal.market_state_snapshot import MarketStateSnapshot


# ============================================================
# Типы записей
# ============================================================

class DecisionType(str, Enum):
    """Тип решения."""
    SIGNAL_CREATED = "SIGNAL_CREATED"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"

    ORDER_SUBMIT = "ORDER_SUBMIT"
    ORDER_ACK = "ORDER_ACK"
    ORDER_FILL = "ORDER_FILL"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"

    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    BREAK_EVEN = "BREAK_EVEN"
    TRAILING_UPDATED = "TRAILING_UPDATED"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"

    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"
    KILL_SWITCH = "KILL_SWITCH"


# ============================================================
# Запись журнала
# ============================================================

class JournalEntry(msgspec.Struct):
    """
    Одна запись Decision Journal.

    Поля сгруппированы логически:
    - идентификация (id, type, symbol)
    - связь с сигналом/позицией/ордером
    - снимок состояния рынка
    - результаты и метрики
    - произвольный metadata
    """
    # --- Обязательные поля ---
    entry_id: str
    ts_ns: int
    decision_type: str  # DecisionType.value
    symbol: str

    # --- Связи ---
    signal_id: Optional[str] = None
    position_id: Optional[str] = None
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    parent_entry_id: Optional[str] = None

    # --- Причины (reason codes) ---
    reasons: List[str] = field(default_factory=list)
    reject_reasons: List[str] = field(default_factory=list)

    # --- Рыночный снимок (может быть большим) ---
    market_state: Optional[Dict[str, Any]] = None

    # --- Результаты ---
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    quantity: Optional[float] = None
    notional: Optional[float] = None

    filled_qty: Optional[float] = None
    avg_fill_price: Optional[float] = None
    slippage_ticks: Optional[float] = None
    fees: Optional[float] = None

    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None

    # --- Метаданные ---
    strategy_version: str = "1.0.0"
    config_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class DecisionJournalConfig:
    """Конфигурация Decision Journal."""
    base_dir: str = "./logs/journal"
    queue_size: int = 100_000
    batch_size: int = 500
    flush_interval_ms: int = 200
    # Если True — критические записи (ORDER_SUBMIT, EMERGENCY_FLATTEN и др.)
    # никогда не дропаются; дропаем только телеметрию.
    protect_critical_entries: bool = True


# Критические записи, которые нельзя терять при переполнении очереди
_CRITICAL_TYPES = {
    DecisionType.ORDER_SUBMIT.value,
    DecisionType.ORDER_ACK.value,
    DecisionType.ORDER_FILL.value,
    DecisionType.ORDER_REJECTED.value,
    DecisionType.POSITION_OPENED.value,
    DecisionType.POSITION_CLOSED.value,
    DecisionType.EMERGENCY_FLATTEN.value,
    DecisionType.KILL_SWITCH.value,
}


# ============================================================
# Асинхронный писатель
# ============================================================

class DecisionJournal:
    """
    Асинхронный Decision Journal.

    Использование:
        journal = DecisionJournal(config)
        await journal.start()

        # В горячем пути:
        journal.log_signal_created(
            symbol="BTCUSDT",
            signal_id="...",
            market_state=snapshot,
            reasons=["LEVEL_ACTIVE", "COMPRESSION"],
        )

        # Периодически/при завершении:
        await journal.close()

    Все log_* методы:
    - НЕ async (никаких await в hot path)
    - put_nowait в очередь
    - дропают запись только при переполнении очереди и если
      она не критическая
    """

    def __init__(self, config: Optional[DecisionJournalConfig] = None) -> None:
        self._config = config or DecisionJournalConfig()
        self._base_dir = Path(self._config.base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._queue: asyncio.Queue[JournalEntry] = asyncio.Queue(
            maxsize=self._config.queue_size
        )
        self._task: Optional[asyncio.Task] = None
        self._closed = False

        # Статистика
        self.written_count: int = 0
        self.dropped_count: int = 0
        self.dropped_critical_count: int = 0

    # ============================================
    # Жизненный цикл
    # ============================================

    async def start(self) -> None:
        if self._task is not None:
            return
        self._closed = False
        self._task = asyncio.create_task(self._writer_loop())

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            await self._task
            self._task = None

    # ============================================
    # Высокоуровневые log_* методы (синхронные)
    # ============================================

    def log_signal_created(
        self,
        symbol: str,
        signal_id: str,
        market_state: Optional[MarketStateSnapshot] = None,
        reasons: Optional[List[str]] = None,
        strategy_version: str = "1.0.0",
        config_hash: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.SIGNAL_CREATED,
            symbol=symbol,
            signal_id=signal_id,
            market_state=market_state,
            reasons=reasons,
            strategy_version=strategy_version,
            config_hash=config_hash,
            metadata=metadata,
        )

    def log_signal_rejected(
        self,
        symbol: str,
        signal_id: str,
        market_state: Optional[MarketStateSnapshot] = None,
        reject_reasons: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.SIGNAL_REJECTED,
            symbol=symbol,
            signal_id=signal_id,
            market_state=market_state,
            reject_reasons=reject_reasons,
            metadata=metadata,
        )

    def log_risk_approved(
        self,
        symbol: str,
        signal_id: str,
        quantity: float,
        notional: float,
        stop_price: Optional[float] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        reasons: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.RISK_APPROVED,
            symbol=symbol,
            signal_id=signal_id,
            quantity=quantity,
            notional=notional,
            stop_price=stop_price,
            market_state=market_state,
            reasons=reasons,
            metadata=metadata,
        )

    def log_risk_rejected(
        self,
        symbol: str,
        signal_id: str,
        reject_reasons: Optional[List[str]] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.RISK_REJECTED,
            symbol=symbol,
            signal_id=signal_id,
            reject_reasons=reject_reasons,
            market_state=market_state,
            metadata=metadata,
        )

    def log_order_submit(
        self,
        symbol: str,
        signal_id: str,
        order_id: str,
        client_order_id: str,
        side: str,
        price: Optional[float],
        quantity: float,
        market_state: Optional[MarketStateSnapshot] = None,
        reasons: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.ORDER_SUBMIT,
            symbol=symbol,
            signal_id=signal_id,
            order_id=order_id,
            client_order_id=client_order_id,
            entry_price=price,
            quantity=quantity,
            market_state=market_state,
            reasons=reasons,
            metadata={"side": side, **(metadata or {})},
        )

    def log_order_ack(
        self,
        symbol: str,
        client_order_id: str,
        exchange_order_id: str,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.ORDER_ACK,
            symbol=symbol,
            order_id=exchange_order_id,
            client_order_id=client_order_id,
            market_state=market_state,
            metadata=metadata,
        )

    def log_order_fill(
        self,
        symbol: str,
        client_order_id: str,
        exchange_order_id: str,
        filled_qty: float,
        avg_fill_price: float,
        slippage_ticks: Optional[float] = None,
        fees: Optional[float] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.ORDER_FILL,
            symbol=symbol,
            order_id=exchange_order_id,
            client_order_id=client_order_id,
            filled_qty=filled_qty,
            avg_fill_price=avg_fill_price,
            slippage_ticks=slippage_ticks,
            fees=fees,
            market_state=market_state,
            metadata=metadata,
        )

    def log_order_rejected(
        self,
        symbol: str,
        client_order_id: str,
        reject_reasons: Optional[List[str]] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.ORDER_REJECTED,
            symbol=symbol,
            client_order_id=client_order_id,
            reject_reasons=reject_reasons,
            market_state=market_state,
            metadata=metadata,
        )

    def log_position_opened(
        self,
        symbol: str,
        position_id: str,
        signal_id: str,
        entry_price: float,
        quantity: float,
        stop_price: float,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.POSITION_OPENED,
            symbol=symbol,
            position_id=position_id,
            signal_id=signal_id,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            market_state=market_state,
            metadata=metadata,
        )

    def log_position_closed(
        self,
        symbol: str,
        position_id: str,
        realized_pnl: float,
        reasons: Optional[List[str]] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.POSITION_CLOSED,
            symbol=symbol,
            position_id=position_id,
            realized_pnl=realized_pnl,
            reasons=reasons,
            market_state=market_state,
            metadata=metadata,
        )

    def log_break_even(
        self,
        symbol: str,
        position_id: str,
        new_stop_price: float,
        unrealized_r: float,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.BREAK_EVEN,
            symbol=symbol,
            position_id=position_id,
            stop_price=new_stop_price,
            metadata={"unrealized_r": unrealized_r, **(metadata or {})},
            market_state=market_state,
        )

    def log_trailing_updated(
        self,
        symbol: str,
        position_id: str,
        new_stop_price: float,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.TRAILING_UPDATED,
            symbol=symbol,
            position_id=position_id,
            stop_price=new_stop_price,
            market_state=market_state,
            metadata=metadata,
        )

    def log_partial_close(
        self,
        symbol: str,
        position_id: str,
        closed_qty: float,
        realized_pnl: float,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.PARTIAL_CLOSE,
            symbol=symbol,
            position_id=position_id,
            filled_qty=closed_qty,
            realized_pnl=realized_pnl,
            market_state=market_state,
            metadata=metadata,
        )

    def log_emergency_flatten(
        self,
        symbol: str,
        position_id: Optional[str],
        reasons: Optional[List[str]] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.EMERGENCY_FLATTEN,
            symbol=symbol,
            position_id=position_id,
            reasons=reasons,
            market_state=market_state,
            metadata=metadata,
        )

    def log_kill_switch(
        self,
        symbol: str,
        reasons: Optional[List[str]] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._enqueue(
            decision_type=DecisionType.KILL_SWITCH,
            symbol=symbol,
            reasons=reasons,
            market_state=market_state,
            metadata=metadata,
        )

    # ============================================
    # Общий вход (для нестандартных случаев)
    # ============================================

    def log(
        self,
        entry: JournalEntry,
    ) -> None:
        """Прямая запись произвольной записи."""
        self._enqueue_entry(entry)

    # ============================================
    # Статистика
    # ============================================

    def get_stats(self) -> Dict[str, int]:
        return {
            "written": self.written_count,
            "dropped": self.dropped_count,
            "dropped_critical": self.dropped_critical_count,
            "queue_size": self._queue.qsize(),
        }

    # ============================================
    # Внутреннее
    # ============================================

    def _enqueue(
        self,
        decision_type: DecisionType,
        symbol: str,
        signal_id: Optional[str] = None,
        position_id: Optional[str] = None,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        reasons: Optional[List[str]] = None,
        reject_reasons: Optional[List[str]] = None,
        market_state: Optional[MarketStateSnapshot] = None,
        entry_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        quantity: Optional[float] = None,
        notional: Optional[float] = None,
        filled_qty: Optional[float] = None,
        avg_fill_price: Optional[float] = None,
        slippage_ticks: Optional[float] = None,
        fees: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        strategy_version: str = "1.0.0",
        config_hash: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        market_dict: Optional[Dict[str, Any]] = None
        if market_state is not None:
            try:
                market_dict = market_state.to_jsonl_dict()
            except Exception:
                market_dict = None

        entry = JournalEntry(
            entry_id=uuid.uuid4().hex,
            ts_ns=time.time_ns(),
            decision_type=decision_type.value,
            symbol=symbol.upper(),
            signal_id=signal_id,
            position_id=position_id,
            order_id=order_id,
            client_order_id=client_order_id,
            reasons=list(reasons or []),
            reject_reasons=list(reject_reasons or []),
            market_state=market_dict,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            notional=notional,
            filled_qty=filled_qty,
            avg_fill_price=avg_fill_price,
            slippage_ticks=slippage_ticks,
            fees=fees,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            strategy_version=strategy_version,
            config_hash=config_hash,
            metadata=dict(metadata or {}),
        )

        self._enqueue_entry(entry)

    def _enqueue_entry(self, entry: JournalEntry) -> None:
        if self._closed:
            return

        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            is_critical = (
                self._config.protect_critical_entries
                and entry.decision_type in _CRITICAL_TYPES
            )
            if is_critical:
                self.dropped_critical_count += 1
            else:
                self.dropped_count += 1

    async def _writer_loop(self) -> None:
        while True:
            batch: List[JournalEntry] = []

            try:
                first = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._config.flush_interval_ms / 1000.0,
                )
                batch.append(first)
            except asyncio.TimeoutError:
                pass

            while len(batch) < self._config.batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if batch:
                await asyncio.to_thread(self._flush_batch, batch)
                self.written_count += len(batch)

            if self._closed and self._queue.empty():
                break

    def _flush_batch(self, batch: List[JournalEntry]) -> None:
        # Группируем по дню
        by_day: Dict[str, List[bytes]] = {}

        for entry in batch:
            day_str = time.strftime(
                "%Y-%m-%d",
                time.gmtime(entry.ts_ns / 1_000_000_000),
            )
            encoded = msgspec.json.encode(entry)
            by_day.setdefault(day_str, []).append(encoded + b"\n")

        for day_str, encoded_list in by_day.items():
            path = self._base_dir / f"{day_str}.jsonl"
            with open(path, "ab") as f:
                f.write(b"".join(encoded_list))