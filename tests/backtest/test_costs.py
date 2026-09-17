"""
Тесты для proscalper.backtest.costs

Запуск:
    python -m pytest tests/backtest/test_costs.py -v
"""
from __future__ import annotations

import pytest

from proscalper.backtest.costs import (
    CostCalculator,
    CostBreakdown,
    FeeConfig,
    FeeModel,
    SlippageConfig,
    SlippageModel,
)


# ============================================================
# FeeModel
# ============================================================

class TestFeeModel:
    """Тесты базовой модели комиссий."""

    def test_default_binance_settings(self):
        """По умолчанию — Binance Futures VIP0."""
        model = FeeModel()
        assert model.config.taker_fee_pct == pytest.approx(0.04)
        assert model.config.maker_fee_pct == pytest.approx(0.02)

    def test_taker_fee(self):
        """Тейкер платит 0.04% от номинала."""
        model = FeeModel()
        fee = model.taker_fee(10_000.0)
        assert fee == pytest.approx(4.0)

    def test_maker_fee(self):
        """Мейкер платит 0.02% от номинала."""
        model = FeeModel()
        fee = model.maker_fee(10_000.0)
        assert fee == pytest.approx(2.0)

    def test_fee_with_role(self):
        """fee() переключает по роли."""
        model = FeeModel()
        assert model.fee(10_000.0, is_maker=True) == pytest.approx(2.0)
        assert model.fee(10_000.0, is_maker=False) == pytest.approx(4.0)

    def test_round_trip_both_taker(self):
        """Полный цикл с обеими тейкерами = 2 * тейкер."""
        model = FeeModel()
        total = model.round_trip_fee(10_000.0, both_taker=True)
        assert total == pytest.approx(8.0)

    def test_round_trip_maker_taker(self):
        """Полный цикл мейкер + тейкер."""
        model = FeeModel()
        total = model.round_trip_fee(10_000.0, both_taker=False)
        assert total == pytest.approx(6.0)

    def test_negative_notional_uses_abs(self):
        """Комиссия всегда положительная."""
        model = FeeModel()
        fee = model.taker_fee(-10_000.0)
        assert fee == pytest.approx(4.0)

    def test_custom_config(self):
        """Кастомные ставки комиссий."""
        config = FeeConfig(taker_fee_pct=0.05, maker_fee_pct=0.01)
        model = FeeModel(config)
        assert model.taker_fee(10_000.0) == pytest.approx(5.0)
        assert model.maker_fee(10_000.0) == pytest.approx(1.0)

    def test_zero_notional(self):
        """Нулевой номинал → нулевая комиссия."""
        model = FeeModel()
        assert model.taker_fee(0.0) == 0.0
        assert model.maker_fee(0.0) == 0.0


# ============================================================
# SlippageModel
# ============================================================

class TestSlippageModel:
    """Тесты модели проскальзывания."""

    def test_invalid_tick_size(self):
        """Неположительный tick_size → ошибка."""
        with pytest.raises(ValueError):
            SlippageModel(tick_size=0.0)
        with pytest.raises(ValueError):
            SlippageModel(tick_size=-0.1)

    def test_maker_no_slippage(self):
        """Мейкер всегда получает 0 проскальзывания."""
        model = SlippageModel(tick_size=0.1)
        book = [(76000.0, 1.0), (76001.0, 1.0)]
        ticks = model.estimate_ticks(0.5, book, is_buy=True, is_maker=True)
        assert ticks == 0.0

    def test_empty_book_base_slippage(self):
        """Пустая книга → базовое проскальзывание."""
        model = SlippageModel(tick_size=0.1)
        ticks = model.estimate_ticks(0.5, [], is_buy=True)
        assert ticks == pytest.approx(1.0)

    def test_zero_qty_base_slippage(self):
        """Нулевой объём → базовое проскальзывание."""
        model = SlippageModel(tick_size=0.1)
        book = [(76000.0, 1.0)]
        ticks = model.estimate_ticks(0.0, book, is_buy=True)
        assert ticks == pytest.approx(1.0)

    def test_single_level_base_slippage(self):
        """Ордер влезает в один уровень → проскальзывание = база."""
        model = SlippageModel(tick_size=0.1)
        book = [(76000.0, 5.0)]
        ticks = model.estimate_ticks(1.0, book, is_buy=True)
        assert ticks == pytest.approx(1.0)

    def test_multiple_levels_slippage(self):
        """Ордер проходит несколько уровней → проскальзывание в тиках."""
        config = SlippageConfig(base_slippage_ticks=0)
        model = SlippageModel(tick_size=0.1, config=config)
        book = [(76000.0, 1.0), (76001.0, 1.0)]
        ticks = model.estimate_ticks(1.5, book, is_buy=True)
        # худшая цена = 76001, лучшая = 76000 → разница 1.0 → 10 тиков
        assert ticks == pytest.approx(10.0)

    def test_slippage_price_buy(self):
        """Цена слиппеджа для покупки выше лучшей."""
        config = SlippageConfig(base_slippage_ticks=0)
        model = SlippageModel(tick_size=0.1, config=config)
        book = [(76000.0, 1.0), (76001.0, 1.0)]
        price = model.slippage_price(1.5, book, is_buy=True)
        assert price == pytest.approx(76001.0)

    def test_slippage_price_sell(self):
        """Цена слиппеджа для продажи ниже лучшей."""
        config = SlippageConfig(base_slippage_ticks=0)
        model = SlippageModel(tick_size=0.1, config=config)
        book = [(76000.0, 1.0), (75999.0, 1.0)]
        price = model.slippage_price(1.5, book, is_buy=False)
        assert price == pytest.approx(75999.0)

    def test_stochastic_with_seed_deterministic(self):
        """Стохастическая модель детерминирована при фиксированном сиде."""
        config = SlippageConfig(stochastic=True, random_seed=42)
        model_1 = SlippageModel(tick_size=0.1, config=config)
        model_2 = SlippageModel(tick_size=0.1, config=config)
        book = [(76000.0, 1.0)]
        ticks_1 = model_1.estimate_ticks(0.5, book, is_buy=True)
        ticks_2 = model_2.estimate_ticks(0.5, book, is_buy=True)
        assert ticks_1 == ticks_2

    def test_slippage_price_empty_book_raises(self):
        """Пустая книга в slippage_price → ошибка."""
        model = SlippageModel(tick_size=0.1)
        with pytest.raises(ValueError):
            model.slippage_price(1.0, [], is_buy=True)

    def test_extrapolation_when_depth_insufficient(self):
        """При нехватке глубины проскальзывание экстраполируется."""
        config = SlippageConfig(base_slippage_ticks=0)
        model = SlippageModel(tick_size=0.1, config=config)
        # Книга с малой глубиной: ордер больше доступного
        book = [(76000.0, 0.1), (76001.0, 0.1)]
        ticks = model.estimate_ticks(10.0, book, is_buy=True)
        # Должно быть больше, чем просто разница между уровнями
        assert ticks > 10.0


# ============================================================
# CostCalculator
# ============================================================

class TestCostCalculator:
    """Тесты объединённого калькулятора издержек."""

    def test_taker_with_book(self):
        """Полный расчёт для тейкера с книгой."""
        calc = CostCalculator(tick_size=0.1)
        book = [(76000.0, 1.0), (76001.0, 1.0)]
        breakdown = calc.compute(
            order_qty=1.5,
            reference_price=76000.0,
            book_side=book,
            is_buy=True,
            is_maker=False,
        )
        assert breakdown.notional == pytest.approx(1.5 * 76000.0)
        assert breakdown.slippage_ticks > 0
        assert breakdown.fill_price >= 76000.0
        assert breakdown.slippage_cost > 0
        assert breakdown.fee > 0
        assert breakdown.total_cost == pytest.approx(
            breakdown.fee + breakdown.slippage_cost
        )

    def test_maker_no_slippage(self):
        """Мейкер не платит проскальзывание."""
        calc = CostCalculator(tick_size=0.1)
        book = [(76000.0, 1.0), (76001.0, 1.0)]
        breakdown = calc.compute(
            order_qty=1.5,
            reference_price=76000.0,
            book_side=book,
            is_buy=True,
            is_maker=True,
        )
        assert breakdown.slippage_ticks == 0.0
        assert breakdown.slippage_cost == 0.0

    def test_no_book_base_slippage(self):
        """Без книги используется базовое проскальзывание."""
        calc = CostCalculator(tick_size=0.1)
        breakdown = calc.compute(
            order_qty=1.0,
            reference_price=76000.0,
            book_side=None,
            is_buy=True,
            is_maker=False,
        )
        assert breakdown.slippage_ticks == pytest.approx(1.0)
        assert breakdown.fill_price == pytest.approx(76000.1)

    def test_no_book_maker(self):
        """Мейкер без книги — нулевое проскальзывание."""
        calc = CostCalculator(tick_size=0.1)
        breakdown = calc.compute(
            order_qty=1.0,
            reference_price=76000.0,
            book_side=None,
            is_buy=True,
            is_maker=True,
        )
        assert breakdown.slippage_ticks == 0.0
        assert breakdown.fill_price == pytest.approx(76000.0)

    def test_sell_no_book(self):
        """Для продажи без книги цена ниже референса."""
        calc = CostCalculator(tick_size=0.1)
        breakdown = calc.compute(
            order_qty=1.0,
            reference_price=76000.0,
            book_side=None,
            is_buy=False,
            is_maker=False,
        )
        assert breakdown.fill_price == pytest.approx(75999.9)

    def test_custom_fees(self):
        """Кастомные ставки комиссий проходят в калькулятор."""
        fee_config = FeeConfig(taker_fee_pct=0.10, maker_fee_pct=0.05)
        calc = CostCalculator(
            tick_size=0.1,
            fee_config=fee_config,
        )
        breakdown = calc.compute(
            order_qty=1.0,
            reference_price=10_000.0,
            book_side=None,
            is_buy=True,
            is_maker=False,
        )
        # номинал ~ 10_000, комиссия 0.10% ≈ 10.0
        # (чуть больше из-за проскальзывания в цене филла)
        assert breakdown.fee > 9.0

    def test_breakdown_fields_consistency(self):
        """Все поля CostBreakdown согласованы."""
        calc = CostCalculator(tick_size=0.1)
        breakdown = calc.compute(
            order_qty=1.0,
            reference_price=76000.0,
            book_side=None,
            is_buy=True,
            is_maker=False,
        )
        # slippage_price = |fill_price - reference_price|
        expected_slippage_price = abs(
            breakdown.fill_price - 76000.0
        )
        assert breakdown.slippage_price == pytest.approx(
            expected_slippage_price
        )
        # slippage_cost = slippage_price * qty
        assert breakdown.slippage_cost == pytest.approx(
            breakdown.slippage_price * 1.0
        )
