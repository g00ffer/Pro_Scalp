"""
Модель издержек бэктеста: комиссии + проскальзывание.

Комиссии на Binance Futures / Bybit — фиксированные в % от номинала.
Проскальзывание — динамическое, зависит от:
- размера ордера относительно глубины стакана
- спреда
- волатильности

Модуль предоставляет:
- FeeModel: комиссии для мейкера/тейкера
- SlippageModel: оценка проскальзывания по снимку стакана
- CostCalculator: объединяет оба

Принципы:
- детерминированность (один вход → один выход)
- никаких случайных величин (по умолчанию)
- опционально — seed для стохастической модели slippage

Используется в связке с:
- backtest/matching.py (для расчёта итоговой цены fill)
- backtest/engine.py (учёт издержек в PnL)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ============================================================
# Комиссии
# ============================================================

@dataclass(frozen=True)
class FeeConfig:
    """
    Конфигурация комиссий.

    Значения по умолчанию — для Binance Futures USDT-M (VIP0):
    - taker: 0.04% (0.0004)
    - maker: 0.02% (0.0002)
    """
    taker_fee_pct: float = 0.04    # в процентах
    maker_fee_pct: float = 0.02    # в процентах


class FeeModel:
    """
    Модель комиссий.

    Возвращает комиссию в USDT (или quote currency) для сделки.
    """

    def __init__(self, config: Optional[FeeConfig] = None) -> None:
        self.config = config or FeeConfig()

    def taker_fee(self, notional: float) -> float:
        """Комиссия тейкера для указанного номинала."""
        return abs(notional) * self.config.taker_fee_pct / 100.0

    def maker_fee(self, notional: float) -> float:
        """Комиссия мейкера."""
        return abs(notional) * self.config.maker_fee_pct / 100.0

    def fee(self, notional: float, is_maker: bool) -> float:
        """Комиссия с учётом роли."""
        return self.maker_fee(notional) if is_maker else self.taker_fee(notional)

    def round_trip_fee(self, notional: float, both_taker: bool = True) -> float:
        """
        Полная комиссия открытия + закрытия для указанного номинала.

        both_taker=True (по умолчанию) — обе ноги тейкер.
        """
        if both_taker:
            return 2.0 * self.taker_fee(notional)
        return self.maker_fee(notional) + self.taker_fee(notional)


# ============================================================
# Проскальзывание
# ============================================================

@dataclass(frozen=True)
class SlippageConfig:
    """
    Конфигурация модели проскальзывания.
    """
    # Абсолютный минимум проскальзывания в тиках (шум)
    base_slippage_ticks: int = 1
    # Множитель на размер ордера относительно глубины топа
    depth_impact_mult: float = 1.0
    # Если True, добавляется случайный шум (для Монте-Карло)
    stochastic: bool = False
    random_seed: int = 42


class SlippageModel:
    """
    Модель проскальзывания на основе снимка стакана.

    Идея:
    - Мейкер: slippage = 0 (или отрицательный — мы даём ликвидность)
    - Тейкер: проходим по уровням стакана до нужного объёма,
      worst_price сравниваем с best_price, разница — проскальзывание в тиках.

    Если depth недостаточно, добавляем "виртуальное" проскальзывание
    сверх последнего уровня.
    """

    def __init__(
        self,
        tick_size: float,
        config: Optional[SlippageConfig] = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size должен быть положительным")

        self.tick_size = tick_size
        self.config = config or SlippageConfig()

        if self.config.stochastic:
            self._rng = random.Random(self.config.random_seed)
        else:
            self._rng = None

    def estimate_ticks(
        self,
        order_qty: float,
        book_side: List[Tuple[float, float]],
        is_buy: bool,
        is_maker: bool = False,
    ) -> float:
        """
        Оценивает проскальзывание в тиках.

        Args:
            order_qty: объём ордера в базовой валюте
            book_side: список (price, qty) на соответствующей стороне стакана.
                      Для BUY — asks, для SELL — bids.
            is_buy: направление (BUY/SELL)
            is_maker: если True — проскальзывание = 0 (мы даём ликвидность)

        Returns:
            Проскальзывание в тиках (≥ 0). Может быть дробным.
        """
        if is_maker:
            return 0.0

        if order_qty <= 0 or not book_side:
            # Нет данных — минимальное проскальзывание
            return float(self.config.base_slippage_ticks)

        best_price = book_side[0][0]
        remaining = order_qty
        worst_price = best_price
        levels_consumed = 0

        for price, qty in book_side:
            if qty <= 0:
                continue
            fill_qty = min(remaining, qty)
            remaining -= fill_qty
            worst_price = price
            levels_consumed += 1

            if remaining <= 0:
                break

        # Если не хватило глубины — экстраполируем:
        # предполагаем линейный рост цены на уровне last_level step
        if remaining > 0 and levels_consumed > 1:
            # Средний шаг между уровнями
            prev_price = book_side[levels_consumed - 2][0] if levels_consumed >= 2 else best_price
            last_price = book_side[levels_consumed - 1][0]
            step = abs(last_price - prev_price) or self.tick_size
            extra_ticks = (remaining / max(order_qty, 1e-9)) * 5.0
            if is_buy:
                worst_price = last_price + step * extra_ticks
            else:
                worst_price = last_price - step * extra_ticks

        raw_slippage = abs(worst_price - best_price)
        ticks = raw_slippage / self.tick_size

        # Добавляем базу
        ticks = max(ticks, float(self.config.base_slippage_ticks))

        # Стохастика (опционально)
        if self._rng is not None:
            noise = self._rng.uniform(0.0, 1.0)
            ticks += noise

        return ticks

    def slippage_price(
        self,
        order_qty: float,
        book_side: List[Tuple[float, float]],
        is_buy: bool,
        is_maker: bool = False,
    ) -> float:
        """
        Возвращает цену fill с учётом проскальзывания.
        """
        if not book_side:
            raise ValueError("book_side пуст")

        best_price = book_side[0][0]
        ticks = self.estimate_ticks(order_qty, book_side, is_buy, is_maker)
        delta = ticks * self.tick_size

        return best_price + delta if is_buy else best_price - delta


# ============================================================
# Калькулятор издержек
# ============================================================

@dataclass
class CostBreakdown:
    """Разбивка издержек одной сделки."""
    notional: float = 0.0
    fee: float = 0.0
    slippage_ticks: float = 0.0
    slippage_price: float = 0.0
    slippage_cost: float = 0.0
    total_cost: float = 0.0
    fill_price: float = 0.0


class CostCalculator:
    """
    Объединяет FeeModel и SlippageModel.

    Использование:
        calc = CostCalculator(tick_size=0.1)
        breakdown = calc.compute(
            order_qty=0.001,
            reference_price=76000.0,
            book_side=asks,
            is_buy=True,
            is_maker=False,
        )
        # breakdown.total_cost — суммарные издержки в USDT
        # breakdown.fill_price — цена fill с учётом проскальзывания
    """

    def __init__(
        self,
        tick_size: float,
        fee_config: Optional[FeeConfig] = None,
        slippage_config: Optional[SlippageConfig] = None,
    ) -> None:
        self.fee_model = FeeModel(fee_config)
        self.slippage_model = SlippageModel(tick_size, slippage_config)

    def compute(
        self,
        order_qty: float,
        reference_price: float,
        book_side: Optional[List[Tuple[float, float]]] = None,
        is_buy: bool = True,
        is_maker: bool = False,
    ) -> CostBreakdown:
        """
        Считает издержки и итоговую цену fill.

        Если book_side не передан — проскальзывание = base_slippage_ticks.
        """
        breakdown = CostBreakdown()
        breakdown.notional = order_qty * reference_price

        if book_side:
            breakdown.slippage_ticks = self.slippage_model.estimate_ticks(
                order_qty, book_side, is_buy, is_maker
            )
            breakdown.fill_price = self.slippage_model.slippage_price(
                order_qty, book_side, is_buy, is_maker
            )
        else:
            breakdown.slippage_ticks = (
                0.0 if is_maker else float(self.slippage_model.config.base_slippage_ticks)
            )
            delta = breakdown.slippage_ticks * self.slippage_model.tick_size
            breakdown.fill_price = reference_price + delta if is_buy else reference_price - delta

        breakdown.slippage_price = abs(breakdown.fill_price - reference_price)
        breakdown.slippage_cost = breakdown.slippage_price * order_qty

        breakdown.fee = self.fee_model.fee(
            notional=breakdown.fill_price * order_qty,
            is_maker=is_maker,
        )

        breakdown.total_cost = breakdown.fee + breakdown.slippage_cost
        return breakdown