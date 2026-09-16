"""
Matching Engine для бэктеста.

Симулирует исполнение ордеров на основе снимка стакана:
- MARKET → проходим по уровням, считаем средневзвешенную цену
- MARKETABLE_LIMIT_IOC → ограничиваем цену, не исполняем что не влезло
- LIMIT → исполняем, если цена касается или хуже
- STOP_MARKET → триггерится при пересечении stop-price
- Частичное исполнение, если глубины не хватает
- Учёт комиссии и проскальзывания через CostCalculator

Принципы:
- НЕТ lookahead: используем только снимок стакана на текущий ts
- Детерминированность (результат зависит только от входа)
- Возврат FillResult с полной разбивкой
- Если нужно — модель latency применяется снаружи (engine)

Используется в связке с:
- backtest/costs.py (комиссии + slippage)
- backtest/engine.py (событийный цикл)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from proscalper.backtest.costs import CostCalculator, CostBreakdown


# ============================================================
# Типы
# ============================================================

class MatchStatus(str, Enum):
    """Статус matching."""
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"       # не исполнился вообще
    SKIPPED = "SKIPPED"         # нет снимка стакана / условия не выполнены


@dataclass
class FillResult:
    """
    Результат попытки матчинга одного ордера.
    """
    status: MatchStatus
    requested_qty: float = 0.0
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    notional: float = 0.0
    fees: float = 0.0
    slippage_ticks: float = 0.0
    slippage_cost: float = 0.0
    total_cost: float = 0.0
    reason: str = ""

    # Заполнено при ошибке / отказе
    reject_reason: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.status == MatchStatus.FILLED

    @property
    def is_partial(self) -> bool:
        return self.status == MatchStatus.PARTIALLY_FILLED


# ============================================================
# Matching Engine
# ============================================================

class MatchingEngine:
    """
    Симулятор исполнения ордера по снимку стакана.

    Использование:
        engine = MatchingEngine(tick_size=0.1, cost_calculator=costs)

        # MARKET
        result = engine.match_market(
            order_qty=0.001,
            bids=book.top_bids(50),
            asks=book.top_asks(50),
            is_buy=True,
        )

        # MARKETABLE LIMIT IOC
        result = engine.match_marketable_limit_ioc(
            order_qty=0.001,
            limit_price=76050.0,
            bids=bids,
            asks=asks,
            is_buy=True,
        )

        # STOP-MARKET (проверка триггера + исполнение как market)
        result = engine.match_stop_market(
            order_qty=0.001,
            stop_price=75990.0,
            best_bid=bid,
            best_ask=ask,
            bids=bids,
            asks=asks,
            is_buy=True,   # направление срабатывания
        )
    """

    def __init__(
        self,
        tick_size: float,
        cost_calculator: Optional[CostCalculator] = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size должен быть положительным")

        self.tick_size = tick_size
        self.costs = cost_calculator or CostCalculator(tick_size=tick_size)

    # ============================================
    # MARKET
    # ============================================

    def match_market(
        self,
        order_qty: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        is_buy: bool,
    ) -> FillResult:
        """
        Симуляция MARKET-ордера.

        Проходим по уровням стакана с противоположной стороны,
        собираем средневзвешенную цену и объём.
        """
        side_book = asks if is_buy else bids
        if not side_book:
            return FillResult(
                status=MatchStatus.SKIPPED,
                requested_qty=order_qty,
                reject_reason="EMPTY_BOOK_SIDE",
            )

        filled_qty, avg_price, remaining = self._walk_book(
            side_book, order_qty
        )

        if filled_qty <= 0:
            return FillResult(
                status=MatchStatus.REJECTED,
                requested_qty=order_qty,
                reject_reason="NO_LIQUIDITY",
            )

        # Издержки через CostCalculator (книга уже учтена в avg_price,
        # поэтому считаем только fees; slippage_ticks считаем отдельно)
        best_price = side_book[0][0]
        raw_slippage_price = abs(avg_price - best_price)
        slippage_ticks = raw_slippage_price / self.tick_size
        slippage_cost = raw_slippage_price * filled_qty

        fees = self.costs.fee_model.taker_fee(avg_price * filled_qty)

        status = (
            MatchStatus.FILLED
            if remaining <= 1e-12
            else MatchStatus.PARTIALLY_FILLED
        )

        return FillResult(
            status=status,
            requested_qty=order_qty,
            filled_qty=filled_qty,
            avg_fill_price=avg_price,
            notional=filled_qty * avg_price,
            fees=fees,
            slippage_ticks=slippage_ticks,
            slippage_cost=slippage_cost,
            total_cost=fees + slippage_cost,
        )

    # ============================================
    # MARKETABLE LIMIT IOC
    # ============================================

    def match_marketable_limit_ioc(
        self,
        order_qty: float,
        limit_price: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        is_buy: bool,
    ) -> FillResult:
        """
        IOC-ордер: исполняется по ценам не хуже limit_price.
        Всё, что не влезло по цене — отменяется (без retry).

        Для BUY: берём asks с price <= limit_price.
        Для SELL: берём bids с price >= limit_price.
        """
        side_book = asks if is_buy else bids
        if not side_book:
            return FillResult(
                status=MatchStatus.SKIPPED,
                requested_qty=order_qty,
                reject_reason="EMPTY_BOOK_SIDE",
            )

        # Фильтруем уровни по лимиту
        eligible: List[Tuple[float, float]] = []
        for price, qty in side_book:
            if is_buy:
                if price <= limit_price:
                    eligible.append((price, qty))
                else:
                    break
            else:
                if price >= limit_price:
                    eligible.append((price, qty))
                else:
                    break

        if not eligible:
            return FillResult(
                status=MatchStatus.REJECTED,
                requested_qty=order_qty,
                reject_reason="PRICE_NOT_REACHED",
            )

        filled_qty, avg_price, remaining = self._walk_book(eligible, order_qty)

        if filled_qty <= 0:
            return FillResult(
                status=MatchStatus.REJECTED,
                requested_qty=order_qty,
                reject_reason="NO_LIQUIDITY_WITHIN_LIMIT",
            )

        best_price = side_book[0][0]
        raw_slippage_price = abs(avg_price - best_price)
        slippage_ticks = raw_slippage_price / self.tick_size
        slippage_cost = raw_slippage_price * filled_qty

        fees = self.costs.fee_model.taker_fee(avg_price * filled_qty)

        status = (
            MatchStatus.FILLED
            if remaining <= 1e-12
            else MatchStatus.PARTIALLY_FILLED
        )

        return FillResult(
            status=status,
            requested_qty=order_qty,
            filled_qty=filled_qty,
            avg_fill_price=avg_price,
            notional=filled_qty * avg_price,
            fees=fees,
            slippage_ticks=slippage_ticks,
            slippage_cost=slippage_cost,
            total_cost=fees + slippage_cost,
        )

    # ============================================
    # STOP-MARKET
    # ============================================

    def match_stop_market(
        self,
        order_qty: float,
        stop_price: float,
        best_bid: float,
        best_ask: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        is_buy: bool,
    ) -> FillResult:
        """
        STOP_MARKET: срабатывает при пересечении stop_price.

        Для BUY (закрытие шорта, стоп сверху):
            триггер: best_ask >= stop_price
        Для SELL (закрытие лонга, стоп снизу):
            триггер: best_bid <= stop_price

        После триггера — исполняем как MARKET по противоположной стороне.
        """
        if is_buy:
            triggered = best_ask >= stop_price
            side_book = asks
        else:
            triggered = best_bid <= stop_price
            side_book = bids

        if not triggered:
            return FillResult(
                status=MatchStatus.SKIPPED,
                requested_qty=order_qty,
                reject_reason="STOP_NOT_TRIGGERED",
            )

        return self.match_market(
            order_qty=order_qty,
            bids=bids,
            asks=asks,
            is_buy=is_buy,
        )

    # ============================================
    # LIMIT (для мейкера)
    # ============================================

    def match_limit(
        self,
        order_qty: float,
        limit_price: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        is_buy: bool,
    ) -> FillResult:
        """
        LIMIT-ордер (мейкер): исполняется только если цена хуже или равна
        лучшей. Комиссия — мейкерская.

        ВАЖНО: это упрощённая модель. В реальности лимитная заявка
        ждёт своей очереди и может исполниться через N секунд.
        Здесь предполагаем мгновенное исполнение по всей доступной
        ликвидности на уровне и ниже/выше.

        Для BUY: уровень asks[0][0] <= limit_price.
        Для SELL: уровень bids[0][0] >= limit_price.
        """
        if is_buy:
            if not asks or asks[0][0] > limit_price:
                return FillResult(
                    status=MatchStatus.SKIPPED,
                    requested_qty=order_qty,
                    reject_reason="LIMIT_NOT_HIT",
                )
        else:
            if not bids or bids[0][0] < limit_price:
                return FillResult(
                    status=MatchStatus.SKIPPED,
                    requested_qty=order_qty,
                    reject_reason="LIMIT_NOT_HIT",
                )

        side_book = asks if is_buy else bids

        # Мейкер: без проскальзывания, цена всегда = limit_price
        # Для упрощения ограничиваем объём доступным по этой цене
        available = sum(
            qty for price, qty in side_book
            if (price <= limit_price if is_buy else price >= limit_price)
        )

        filled_qty = min(order_qty, available)
        if filled_qty <= 0:
            return FillResult(
                status=MatchStatus.SKIPPED,
                requested_qty=order_qty,
                reject_reason="NO_LIQUIDITY_AT_LIMIT",
            )

        fees = self.costs.fee_model.maker_fee(limit_price * filled_qty)

        status = (
            MatchStatus.FILLED
            if filled_qty >= order_qty - 1e-12
            else MatchStatus.PARTIALLY_FILLED
        )

        return FillResult(
            status=status,
            requested_qty=order_qty,
            filled_qty=filled_qty,
            avg_fill_price=limit_price,
            notional=filled_qty * limit_price,
            fees=fees,
            slippage_ticks=0.0,
            slippage_cost=0.0,
            total_cost=fees,
        )

    # ============================================
    # Вспомогательное
    # ============================================

    def _walk_book(
        self,
        side_book: List[Tuple[float, float]],
        order_qty: float,
    ) -> Tuple[float, float, float]:
        """
        Проходит по уровням стакана, собирая заполнение.

        Returns:
            (filled_qty, avg_price, remaining_qty)
        """
        remaining = order_qty
        filled_qty = 0.0
        notional = 0.0

        for price, qty in side_book:
            if qty <= 0 or price <= 0:
                continue

            take = min(remaining, qty)
            filled_qty += take
            notional += take * price
            remaining -= take

            if remaining <= 1e-12:
                break

        avg_price = notional / filled_qty if filled_qty > 0 else 0.0
        return filled_qty, avg_price, max(0.0, remaining)

    def set_tick_size(self, tick_size: float) -> None:
        """Обновление tick_size (редко нужно)."""
        self.tick_size = tick_size