"""
Журнал сделок (Trade Journal).

Хранит полный журнал фактических сделок:
- открытие позиции (вход, цена, объём, стоп)
- изменения позиции (брейк-ивен, трейлинг, частичное закрытие)
- закрытие позиции (выход, PnL, причина)

Отличие от Decision Journal:
- Decision Journal = все решения (принятые и отклонённые)
- Trade Journal = только фактические сделки с полным учётом

Используется для:
- Анализа производительности стратегии
- Построения отчётов (совместим с BacktestReport)
- Аудита исполнения
- Расчёта статистики (win rate, profit factor, R-multiples)

Формат: JSONL по дням, как в decision_journal и incident_journal.

Используется в связке с:
- journal/reasons.py (коды причин входа/выхода)
- journal/market_state_snapshot.py (снимки рынка)
- app/paper_runner.py (бумажная торговля)
- app/live_runner.py (живая торговля)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec

from proscalper.journal.reasons import ExitReason, SignalReason


# ============================================================
# Типы записей
# ============================================================

class TradeEventType(str, Enum):
    """Тип события сделки."""
    TRADE_OPENED = "TRADE_OPENED"
    TRADE_UPDATED = "TRADE_UPDATED"
    TRADE_CLOSED = "TRADE_CLOSED"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    BREAK_EVEN = "BREAK_EVEN"
    TRAILING_UPDATE = "TRAILING_UPDATE"


class TradeDirection(str, Enum):
    """Направление сделки."""
    LONG = "LONG"
    SHORT = "SHORT"


# ============================================================
# Запись журнала
# ============================================================

class TradeJournalEntry(msgspec.Struct):
    """
    Одна запись журнала сделок.
    
    Хранит полное состояние сделки на момент события:
    - идентификация (id, symbol, direction)
    - цены и объёмы (вход, выход, стоп)
    - издержки (комиссии, проскальзывание)
    - PnL (gross, net, R-multiple)
    - причины (вход, выход) из journal/reasons.py
    - временные метки (для анализа задержек)
    """
    # --- Обязательные поля ---
    trade_id: str
    ts_ns: int
    event_type: str  # TradeEventType.value
    symbol: str
    direction: str   # TradeDirection.value
    
    # --- Связи ---
    signal_id: Optional[str] = None
    position_id: Optional[str] = None
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    
    # --- Цены и объёмы ---
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    quantity: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    
    # --- Издержки ---
    fees: float = 0.0
    slippage_ticks: float = 0.0
    slippage_cost: float = 0.0
    
    # --- PnL ---
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    
    # --- Причины (из journal/reasons.py) ---
    entry_reasons: List[str] = []
    exit_reasons: List[str] = []
    
    # --- Временные метки ---
    entry_ts_ns: int = 0
    exit_ts_ns: int = 0
    duration_ms: int = 0
    
    # --- Снимок рынка (опционально) ---
    market_state: Optional[Dict[str, Any]] = None
    
    # --- Метаданные ---
    strategy_version: str = "1.0.0"
    metadata: Dict[str, Any] = {}


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class TradeJournalConfig:
    """Конфигурация журнала сделок."""
    base_dir: str = "./logs/trades"
    queue_size: int = 50_000
    batch_size: int = 200
    flush_interval_ms: int = 200
    
    # Если True — записи закрытия сделок никогда не дропаются
    protect_close_entries: bool = True


# ============================================================
# Журнал
# ============================================================

class TradeJournal:
    """
    Асинхронный журнал сделок.
    
    Использование:
        journal = TradeJournal(config)
        await journal.start()
        
        # При открытии позиции:
        journal.log_open(
            symbol="BTCUSDT",
            direction=TradeDirection.LONG,
            signal_id="sig_123",
            position_id="pos_456",
            entry_price=76000.0,
            quantity=0.001,
            stop_price=75990.0,
            entry_reasons=[SignalReason.BREAKOUT, SignalReason.DELTA_CONFIRMED],
        )
        
        # При закрытии позиции:
        journal.log_close(
            symbol="BTCUSDT",
            position_id="pos_456",
            exit_price=76050.0,
            exit_reasons=[ExitReason.STOP_LOSS],
            gross_pnl=0.05,
            fees=0.002,
        )
        
        # При завершении:
        await journal.close()
    
    Все log_* методы:
    - НЕ async (никаких await в hot path)
    - put_nowait в очередь
    - дропают запись только при переполнении очереди
      и если она не критическая
    """
    
    def __init__(self, config: Optional[TradeJournalConfig] = None) -> None:
        self._config = config or TradeJournalConfig()
        self._base_dir = Path(self._config.base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        
        self._queue: asyncio.Queue[TradeJournalEntry] = asyncio.Queue(
            maxsize=self._config.queue_size
        )
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        
        # Статистика
        self.written_count: int = 0
        self.dropped_count: int = 0
        
        # Кэш открытых сделок для быстрого доступа
        # (position_id -> последняя запись)
        self._open_trades: Dict[str, TradeJournalEntry] = {}
    
    # ============================================
    # Жизненный цикл
    # ============================================
    
    async def start(self) -> None:
        """Запускает фоновый писатель."""
        if self._task is not None:
            return
        self._closed = False
        self._task = asyncio.create_task(self._writer_loop())
    
    async def close(self) -> None:
        """Останавливает писатель, дожидается записи всех записей."""
        self._closed = True
        if self._task is not None:
            await self._task
            self._task = None
    
    # ============================================
    # Публичные методы (синхронные)
    # ============================================
    
    def log_open(
        self,
        symbol: str,
        direction: TradeDirection,
        signal_id: str,
        position_id: str,
        entry_price: float,
        quantity: float,
        stop_price: float,
        entry_order_id: Optional[str] = None,
        target_price: Optional[float] = None,
        entry_reasons: Optional[List[str]] = None,
        fees: float = 0.0,
        slippage_ticks: float = 0.0,
        slippage_cost: float = 0.0,
        market_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Логирует открытие сделки."""
        entry = TradeJournalEntry(
            trade_id=uuid.uuid4().hex,
            ts_ns=time.time_ns(),
            event_type=TradeEventType.TRADE_OPENED.value,
            symbol=symbol.upper(),
            direction=direction.value,
            signal_id=signal_id,
            position_id=position_id,
            entry_order_id=entry_order_id,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            fees=fees,
            slippage_ticks=slippage_ticks,
            slippage_cost=slippage_cost,
            entry_reasons=list(entry_reasons or []),
            entry_ts_ns=time.time_ns(),
            market_state=market_state,
            metadata=dict(metadata or {}),
        )
        
        self._open_trades[position_id] = entry
        self._enqueue(entry)
    
    def log_close(
        self,
        symbol: str,
        position_id: str,
        exit_price: float,
        exit_reasons: Optional[List[str]] = None,
        exit_order_id: Optional[str] = None,
        gross_pnl: float = 0.0,
        fees: float = 0.0,
        slippage_cost: float = 0.0,
        market_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Логирует закрытие сделки."""
        # Находим открытую сделку для расчёта PnL и duration
        open_entry = self._open_trades.pop(position_id, None)
        
        entry_price = open_entry.entry_price if open_entry else None
        quantity = open_entry.quantity if open_entry else None
        entry_ts_ns = open_entry.entry_ts_ns if open_entry else 0
        entry_fees = open_entry.fees if open_entry else 0.0
        entry_slippage_cost = open_entry.slippage_cost if open_entry else 0.0
        
        # Рассчитываем PnL если не передан
        net_pnl = gross_pnl - fees - entry_fees
        
        # Рассчитываем R-multiple
        r_multiple = 0.0
        if open_entry and open_entry.stop_price and entry_price and quantity:
            risk_per_unit = abs(entry_price - open_entry.stop_price)
            if risk_per_unit > 0:
                r_multiple = net_pnl / (risk_per_unit * quantity)
        
        # Рассчитываем длительность
        duration_ms = 0
        if entry_ts_ns > 0:
            duration_ms = int((time.time_ns() - entry_ts_ns) / 1_000_000)
        
        entry = TradeJournalEntry(
            trade_id=uuid.uuid4().hex,
            ts_ns=time.time_ns(),
            event_type=TradeEventType.TRADE_CLOSED.value,
            symbol=symbol.upper(),
            direction=open_entry.direction if open_entry else TradeDirection.LONG.value,
            signal_id=open_entry.signal_id if open_entry else None,
            position_id=position_id,
            exit_order_id=exit_order_id,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            stop_price=open_entry.stop_price if open_entry else None,
            fees=fees + entry_fees,
            slippage_cost=slippage_cost + entry_slippage_cost,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            r_multiple=r_multiple,
            exit_reasons=list(exit_reasons or []),
            entry_ts_ns=entry_ts_ns,
            exit_ts_ns=time.time_ns(),
            duration_ms=duration_ms,
            market_state=market_state,
            metadata=dict(metadata or {}),
        )
        
        self._enqueue(entry)
    
    def log_partial_close(
        self,
        symbol: str,
        position_id: str,
        closed_qty: float,
        realized_pnl: float,
        exit_price: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Логирует частичное закрытие позиции."""
        entry = TradeJournalEntry(
            trade_id=uuid.uuid4().hex,
            ts_ns=time.time_ns(),
            event_type=TradeEventType.PARTIAL_CLOSE.value,
            symbol=symbol.upper(),
            direction=TradeDirection.LONG.value,  # будет уточнено
            position_id=position_id,
            quantity=closed_qty,
            exit_price=exit_price,
            gross_pnl=realized_pnl,
            net_pnl=realized_pnl,
            metadata=dict(metadata or {}),
        )
        
        self._enqueue(entry)
    
    def log_break_even(
        self,
        symbol: str,
        position_id: str,
        new_stop_price: float,
        unrealized_r: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Логирует перенос стопа в безубыток."""
        entry = TradeJournalEntry(
            trade_id=uuid.uuid4().hex,
            ts_ns=time.time_ns(),
            event_type=TradeEventType.BREAK_EVEN.value,
            symbol=symbol.upper(),
            direction=TradeDirection.LONG.value,
            position_id=position_id,
            stop_price=new_stop_price,
            r_multiple=unrealized_r,
            metadata=dict(metadata or {}),
        )
        
        self._enqueue(entry)
    
    def log_trailing_update(
        self,
        symbol: str,
        position_id: str,
        new_stop_price: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Логирует обновление трейлинг-стопа."""
        entry = TradeJournalEntry(
            trade_id=uuid.uuid4().hex,
            ts_ns=time.time_ns(),
            event_type=TradeEventType.TRAILING_UPDATE.value,
            symbol=symbol.upper(),
            direction=TradeDirection.LONG.value,
            position_id=position_id,
            stop_price=new_stop_price,
            metadata=dict(metadata or {}),
        )
        
        self._enqueue(entry)
    
    # ============================================
    # Запросы
    # ============================================
    
    def get_open_trades(self) -> List[TradeJournalEntry]:
        """Возвращает все открытые сделки."""
        return list(self._open_trades.values())
    
    def get_open_trade(self, position_id: str) -> Optional[TradeJournalEntry]:
        """Возвращает открытую сделку по position_id."""
        return self._open_trades.get(position_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика журнала."""
        return {
            "written": self.written_count,
            "dropped": self.dropped_count,
            "queue_size": self._queue.qsize(),
            "open_trades": len(self._open_trades),
        }
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    def _enqueue(self, entry: TradeJournalEntry) -> None:
        """Ставит запись в очередь."""
        if self._closed:
            return
        
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            # Проверяем, критическая ли запись
            is_critical = (
                self._config.protect_close_entries
                and entry.event_type == TradeEventType.TRADE_CLOSED.value
            )
            
            if not is_critical:
                self.dropped_count += 1
            else:
                # Критические записи не дропаем — логируем в stderr
                import sys
                print(
                    f"[TRADE_JOURNAL] QUEUE FULL, critical entry "
                    f"{entry.event_type} dropped",
                    file=sys.stderr,
                )
                self.dropped_count += 1
    
    async def _writer_loop(self) -> None:
        """Фоновый цикл записи."""
        while True:
            batch: List[TradeJournalEntry] = []
            
            try:
                first = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._config.flush_interval_ms / 1000.0,
                )
                batch.append(first)
            except asyncio.TimeoutError:
                pass
            
            # Собираем батч
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
    
    def _flush_batch(self, batch: List[TradeJournalEntry]) -> None:
        """Пишет батч в JSONL файлы по дням."""
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