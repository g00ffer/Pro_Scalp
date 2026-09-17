"""
Сверка локального состояния позиций с биржей (Reconciliation).

Проблема: между локальным состоянием (позиции, стопы в памяти
робота) и реальным состоянием на бирже может возникнуть
расхождение:
- робот перезапустился, а позиции остались
- ордер стопа был отклонён, но робот не заметил
- сетевой сбой → ордер ушёл, а ack не пришёл
- частичный филл не учтён локально

Решение: периодическая сверка локального состояния с биржей
через sync.py и принятие корректирующих действий.

Уровни расхождений:
1. MISSING_STOP: локально позиция есть, на бирже стопа нет
   → КРИТИЧНО → экстренно поставить стоп или закрыть позицию
2. POSITION_MISMATCH: объём на бирже ≠ локальный
   → КРИТИЧНО → корректировка или аварийное закрытие
3. ORPHAN_ORDER: на бирже есть ордер, локально его нет
   → ВНИМАНИЕ → отмена или принятие
4. ORPHAN_POSITION: на бирже позиция, локально нет
   → ВНИМАНИЕ → принятие или закрытие
5. SYNCED: всё совпадает

Используется в связке с:
- exchange/binance_futures/sync.py (данные с биржи)
- journal/incident_journal.py (логирование расхождений)
- journal/reasons.py (коды причин)
- app/live_runner.py (периодический вызов)

Принципы:
- НЕ изменяет состояние автоматически по умолчанию
- сообщает о расхождениях через callback / журнал
- в режиме `auto_fix` может принимать корректирующие действия
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from proscalper.exchange.binance_futures.sync import (
    BinanceSyncer,
    SyncResult,
    PositionSnapshot,
    OpenOrderSnapshot,
)
from proscalper.core.logging import get_logger
from proscalper.journal.reasons import ProtectionReason

logger = get_logger(__name__)


# ============================================================
# Типы расхождений
# ============================================================

class DiscrepancyType(str, Enum):
    """Тип расхождения."""
    MISSING_STOP = "MISSING_STOP"
    POSITION_MISMATCH = "POSITION_MISMATCH"
    ORPHAN_ORDER = "ORPHAN_ORDER"
    ORPHAN_POSITION = "ORPHAN_POSITION"
    STALE_DATA = "STALE_DATA"
    SYNCED = "SYNCED"


class DiscrepancySeverity(str, Enum):
    """Критичность расхождения."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class Discrepancy:
    """Одно обнаруженное расхождение."""
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    symbol: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    ts_ns: int = field(default_factory=time.time_ns)

    @property
    def is_critical(self) -> bool:
        return self.severity in (
            DiscrepancySeverity.ERROR,
            DiscrepancySeverity.CRITICAL,
        )


@dataclass
class ReconciliationResult:
    """Результат сверки."""
    ts_ns: int
    success: bool = False
    discrepancies: List[Discrepancy] = field(default_factory=list)
    symbols_checked: int = 0
    positions_checked: int = 0

    @property
    def has_critical(self) -> bool:
        return any(d.is_critical for d in self.discrepancies)

    @property
    def has_warnings(self) -> bool:
        return any(
            d.severity == DiscrepancySeverity.WARNING
            for d in self.discrepancies
        )

    @property
    def is_clean(self) -> bool:
        return len(self.discrepancies) == 0


# ============================================================
# Локальное состояние (для сравнения)
# ============================================================

@dataclass
class LocalPosition:
    """Локальная позиция для сравнения."""
    symbol: str
    side: str              # "LONG" / "SHORT"
    quantity: float
    entry_price: float
    stop_order_id: Optional[str] = None
    signal_id: Optional[str] = None


# ============================================================
# Реконсилятор
# ============================================================

class Reconciliator:
    """
    Сверка локального состояния с биржей.

    Использование:
        reconciliator = Reconciliator(syncer)

        # Регистрируем локальные позиции
        reconciliator.set_local_positions([
            LocalPosition("BTCUSDT", "LONG", 0.001, 76000.0, "stop_123"),
        ])

        # Периодическая сверка
        result = await reconciliator.reconcile()
        if result.has_critical:
            for d in result.discrepancies:
                if d.is_critical:
                    print(f"CRITICAL: {d.message}")

        # Callback для инцидентов
        reconciliator.on_discrepancy(my_incident_handler)
    """

    def __init__(
        self,
        syncer: BinanceSyncer,
        auto_fix: bool = False,
    ) -> None:
        self._syncer = syncer
        self._auto_fix = auto_fix

        # Локальное состояние
        self._local_positions: Dict[str, LocalPosition] = {}
        self._local_stop_ids: Dict[str, str] = {}  # symbol -> stop_order_id

        # Callback для расхождений
        self._on_discrepancy: Optional[
            Callable[[Discrepancy], None]
        ] = None

        # Счётчики
        self._total_reconciliations: int = 0
        self._total_discrepancies: int = 0
        self._last_reconciliation_ts_ns: int = 0

    # ============================================
    # Настройка
    # ============================================

    def set_local_positions(
        self, positions: List[LocalPosition]
    ) -> None:
        """Устанавливает локальные позиции для сверки."""
        self._local_positions = {
            p.symbol.upper(): p for p in positions
        }
        self._local_stop_ids = {
            p.symbol.upper(): p.stop_order_id
            for p in positions
            if p.stop_order_id
        }

    def update_local_position(self, position: LocalPosition) -> None:
        """Обновляет одну локальную позицию."""
        symbol = position.symbol.upper()
        self._local_positions[symbol] = position
        if position.stop_order_id:
            self._local_stop_ids[symbol] = position.stop_order_id
        elif symbol in self._local_stop_ids:
            del self._local_stop_ids[symbol]

    def remove_local_position(self, symbol: str) -> None:
        """Удаляет локальную позицию (после закрытия)."""
        symbol = symbol.upper()
        self._local_positions.pop(symbol, None)
        self._local_stop_ids.pop(symbol, None)

    def on_discrepancy(
        self, callback: Callable[[Discrepancy], None]
    ) -> None:
        """Регистрирует обработчик расхождений."""
        self._on_discrepancy = callback

    # ============================================
    # Сверка
    # ============================================

    async def reconcile(
        self,
        symbols: Optional[List[str]] = None,
    ) -> ReconciliationResult:
        """
        Выполняет полную сверку.

        1. Запрашивает состояние с биржи через syncer
        2. Сравнивает с локальными позициями
        3. Возвращает список расхождений
        """
        self._total_reconciliations += 1
        self._last_reconciliation_ts_ns = time.time_ns()

        result = ReconciliationResult(
            ts_ns=time.time_ns(),
        )

        # Запрашиваем данные с биржи
        sync_result = await self._syncer.full_sync(symbols)

        if not sync_result.success:
            result.discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.STALE_DATA,
                severity=DiscrepancySeverity.ERROR,
                symbol="ALL",
                message=f"Sync failed: {sync_result.error}",
            ))
            return result

        result.success = True

        # Сравниваем позиции
        symbols_to_check = (
            [s.upper() for s in symbols]
            if symbols
            else set(
                list(self._local_positions.keys())
                + [p.symbol for p in sync_result.positions if p.is_open]
            )
        )

        result.symbols_checked = len(symbols_to_check)

        for symbol in symbols_to_check:
            self._check_symbol(
                symbol=symbol,
                sync_result=sync_result,
                result=result,
            )

        # Проверяем сиротские ордера
        self._check_orphan_orders(sync_result, result)

        # Обновляем счётчики
        self._total_discrepancies += len(result.discrepancies)

        # Вызываем callback для каждого расхождения
        if self._on_discrepancy is not None:
            for d in result.discrepancies:
                try:
                    self._on_discrepancy(d)
                except Exception:
                    pass

        # Логируем результат
        if result.is_clean:
            logger.debug(
                "Reconciliation clean",
                symbols=result.symbols_checked,
            )
        else:
            critical_count = sum(
                1 for d in result.discrepancies if d.is_critical
            )
            logger.warning(
                "Reconciliation found issues",
                total=len(result.discrepancies),
                critical=critical_count,
            )

        return result

    # ============================================
    # Проверки
    # ============================================

    def _check_symbol(
        self,
        symbol: str,
        sync_result: SyncResult,
        result: ReconciliationResult,
    ) -> None:
        """Проверяет один символ."""
        local = self._local_positions.get(symbol)
        exchange = sync_result.get_position(symbol)

        result.positions_checked += 1

        # Случай 1: локально есть позиция, на бирже нет
        if local is not None and exchange is None:
            result.discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.POSITION_MISMATCH,
                severity=DiscrepancySeverity.CRITICAL,
                symbol=symbol,
                message=(
                    f"Local position exists but not on exchange: "
                    f"{local.side} {local.quantity}"
                ),
                details={
                    "local_side": local.side,
                    "local_qty": local.quantity,
                    "exchange": None,
                },
            ))
            return

        # Случай 2: на бирже есть позиция, локально нет
        if local is None and exchange is not None:
            result.discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.ORPHAN_POSITION,
                severity=DiscrepancySeverity.WARNING,
                symbol=symbol,
                message=(
                    f"Exchange position not tracked locally: "
                    f"{'LONG' if exchange.is_long else 'SHORT'} "
                    f"{abs(exchange.position_amt)}"
                ),
                details={
                    "exchange_side": (
                        "LONG" if exchange.is_long else "SHORT"
                    ),
                    "exchange_qty": abs(exchange.position_amt),
                    "exchange_entry": exchange.entry_price,
                },
            ))
            return

        # Оба есть — сравниваем
        if local is not None and exchange is not None:
            # Проверка стороны
            exchange_side = "LONG" if exchange.is_long else "SHORT"
            if local.side != exchange_side:
                result.discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.POSITION_MISMATCH,
                    severity=DiscrepancySeverity.CRITICAL,
                    symbol=symbol,
                    message=(
                        f"Side mismatch: local={local.side}, "
                        f"exchange={exchange_side}"
                    ),
                    details={
                        "local_side": local.side,
                        "exchange_side": exchange_side,
                    },
                ))

            # Проверка объёма
            local_qty = abs(local.quantity)
            exchange_qty = abs(exchange.position_amt)
            if abs(local_qty - exchange_qty) > 1e-8:
                result.discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.POSITION_MISMATCH,
                    severity=DiscrepancySeverity.CRITICAL,
                    symbol=symbol,
                    message=(
                        f"Quantity mismatch: local={local_qty}, "
                        f"exchange={exchange_qty}"
                    ),
                    details={
                        "local_qty": local_qty,
                        "exchange_qty": exchange_qty,
                        "diff": exchange_qty - local_qty,
                    },
                ))

            # Проверка стопа
            self._check_stop_exists(
                symbol=symbol,
                sync_result=sync_result,
                result=result,
            )

    def _check_stop_exists(
        self,
        symbol: str,
        sync_result: SyncResult,
        result: ReconciliationResult,
    ) -> None:
        """Проверяет, что стоп-ордер существует на бирже."""
        expected_stop_id = self._local_stop_ids.get(symbol)

        if expected_stop_id is None:
            # Локально стоп не ожидается — ок
            return

        # Ищем стоп среди открытых ордеров
        open_orders = sync_result.get_open_orders(symbol)
        stop_orders = [
            o for o in open_orders
            if o.order_type in (
                "STOP_MARKET", "STOP",
                "TAKE_PROFIT_MARKET", "TAKE_PROFIT",
            )
        ]

        if len(stop_orders) == 0:
            # Стоп отсутствует на бирже — КРИТИЧНО
            result.discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.MISSING_STOP,
                severity=DiscrepancySeverity.CRITICAL,
                symbol=symbol,
                message=(
                    f"Stop order missing on exchange! "
                    f"Expected: {expected_stop_id}"
                ),
                details={
                    "expected_stop_id": expected_stop_id,
                    "open_orders_count": len(open_orders),
                },
            ))
        else:
            # Проверяем, что наш стоп среди них
            found = any(
                o.client_order_id == expected_stop_id
                or str(o.order_id) == expected_stop_id
                for o in stop_orders
            )
            if not found:
                result.discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.MISSING_STOP,
                    severity=DiscrepancySeverity.ERROR,
                    symbol=symbol,
                    message=(
                        f"Stop order ID mismatch: expected "
                        f"{expected_stop_id}, found "
                        f"{[o.client_order_id for o in stop_orders]}"
                    ),
                    details={
                        "expected": expected_stop_id,
                        "found": [
                            o.client_order_id for o in stop_orders
                        ],
                    },
                ))

    def _check_orphan_orders(
        self,
        sync_result: SyncResult,
        result: ReconciliationResult,
    ) -> None:
        """Ищет ордера на бирже, которых нет локально."""
        known_stop_ids = set(self._local_stop_ids.values())

        for order in sync_result.open_orders:
            if order.order_type in (
                "STOP_MARKET", "STOP",
                "TAKE_PROFIT_MARKET", "TAKE_PROFIT",
            ):
                if (
                    order.client_order_id not in known_stop_ids
                    and str(order.order_id) not in known_stop_ids
                ):
                    result.discrepancies.append(Discrepancy(
                        discrepancy_type=DiscrepancyType.ORPHAN_ORDER,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=order.symbol,
                        message=(
                            f"Unknown stop order on exchange: "
                            f"{order.client_order_id} "
                            f"({order.order_type} {order.side})"
                        ),
                        details={
                            "order_id": order.order_id,
                            "client_order_id": order.client_order_id,
                            "order_type": order.order_type,
                            "side": order.side,
                        },
                    ))

    # ============================================
    # Статистика
    # ============================================

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_reconciliations": self._total_reconciliations,
            "total_discrepancies": self._total_discrepancies,
            "last_reconciliation_ts_ns": self._last_reconciliation_ts_ns,
            "local_positions": len(self._local_positions),
            "auto_fix": self._auto_fix,
        }