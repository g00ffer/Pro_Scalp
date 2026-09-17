"""
Тесты для proscalper.backtest.matching

Запуск:
    python -m pytest tests/backtest/test_matching.py -v
"""
from __future__ import annotations

import pytest

from proscalper.backtest.costs import CostCalculator, FeeConfig
from proscalper.backtest.matching import (
    FillResult,
    MatchStatus,
    MatchingEngine,
)


# ============================================================
# Фикстуры
# ============================================================

@pytest.fixture
def engine() -> MatchingEngine:
    """Движок с tick_size=1.0 для простых расчётов."""
    return MatchingEngine(tick_size=1.0)


@pytest.fixture
def engine_fine() -> MatchingEngine:
    """Движок с мелким тиком (как у BTC)."""
    return MatchingEngine(tick_size=0.1)


# ============================================================
# Инициализация
# ============================================================

class TestMatchingEngineInit:
    """Тесты конструктора."""

    def test_invalid_tick_size_zero(self):
        with pytest.raises(ValueError):
            MatchingEngine(tick_size=0.0)

    def test_invalid_tick_size_negative(self):
        with pytest.raises(ValueError):
            MatchingEngine(tick_size=-0.5)

    def test_default_cost_calculator_created(self):
        """Если cost_calculator не передан — создаётся дефолтный."""
        engine = MatchingEngine(tick_size=0.1)
        assert engine.costs is not None
        assert isinstance(engine.costs, CostCalculator)

    def test_custom_cost_calculator(self):
        """Переданный cost_calculator используется."""
        costs = CostCalculator(
            tick_size=1.0,
            fee_config=FeeConfig(taker_fee_pct=0.10),
        )
        engine = MatchingEngine(tick_size=1.0, cost_calculator=costs)
        assert engine.costs is costs

    def test_set_tick_size(self):
        engine = MatchingEngine(tick_size=1.0)
        engine.set_tick_size(0.5)
        assert engine.tick_size == 0.5


# ============================================================
# match_market
# ============================================================

class TestMatchMarket:
    """Тесты MARKET-ордеров."""

    def test_buy_empty_asks_skipped(self, engine):
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 1.0)],
            asks=[],
            is_buy=True,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "EMPTY_BOOK_SIDE"

    def test_sell_empty_bids_skipped(self, engine):
        result = engine.match_market(
            order_qty=1.0,
            bids=[],
            asks=[(101.0, 1.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "EMPTY_BOOK_SIDE"

    def test_buy_single_level_full_fill(self, engine):
        """Ордер полностью влезает в один уровень."""
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED
        assert result.is_filled
        assert result.filled_qty == pytest.approx(1.0)
        assert result.avg_fill_price == pytest.approx(100.0)
        assert result.requested_qty == pytest.approx(1.0)

    def test_buy_multi_level_weighted_avg(self, engine):
        """Ордер проходит несколько уровней — средневзвешенная цена."""
        result = engine.match_market(
            order_qty=1.5,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 1.0), (102.0, 1.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED
        # avg = (1.0 * 100 + 0.5 * 102) / 1.5 = 151 / 1.5
        expected_avg = (1.0 * 100.0 + 0.5 * 102.0) / 1.5
        assert result.avg_fill_price == pytest.approx(expected_avg)
        assert result.filled_qty == pytest.approx(1.5)

    def test_buy_partial_fill(self, engine):
        """Не хватает глубины → частичное исполнение."""
        result = engine.match_market(
            order_qty=2.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 0.5)],
            is_buy=True,
        )
        assert result.status == MatchStatus.PARTIALLY_FILLED
        assert result.is_partial
        assert result.filled_qty == pytest.approx(0.5)
        assert result.requested_qty == pytest.approx(2.0)

    def test_sell_uses_bids(self, engine):
        """SELL берёт ликвидность из bids."""
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 2.0)],
            asks=[(101.0, 2.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.FILLED
        assert result.avg_fill_price == pytest.approx(99.0)

    def test_no_liquidity_rejected(self, engine):
        """Все уровни с нулевым объёмом → REJECTED."""
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 0.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.REJECTED
        assert result.reject_reason == "NO_LIQUIDITY"

    def test_fees_taker(self, engine):
        """Комиссия тейкера = 0.04% от номинала."""
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        # notional = 1.0 * 100.0 = 100.0
        # fees = 100.0 * 0.04 / 100 = 0.04
        assert result.fees == pytest.approx(0.04)
        assert result.notional == pytest.approx(100.0)

    def test_slippage_ticks_zero_single_level(self, engine):
        """Один уровень → нулевое проскальзывание."""
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.slippage_ticks == pytest.approx(0.0)
        assert result.slippage_cost == pytest.approx(0.0)

    def test_slippage_ticks_multi_level(self, engine):
        """Несколько уровней → проскальзывание в тиках."""
        result = engine.match_market(
            order_qty=1.5,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 1.0), (105.0, 1.0)],
            is_buy=True,
        )
        # avg = (1.0 * 100 + 0.5 * 105) / 1.5 = 152.5 / 1.5 = 101.6667
        expected_avg = (1.0 * 100.0 + 0.5 * 105.0) / 1.5
        expected_slippage_ticks = (expected_avg - 100.0) / 1.0
        assert result.slippage_ticks == pytest.approx(expected_slippage_ticks)
        assert result.slippage_cost == pytest.approx(
            (expected_avg - 100.0) * 1.5
        )

    def test_total_cost_is_fees_plus_slippage(self, engine):
        """total_cost = fees + slippage_cost."""
        result = engine.match_market(
            order_qty=1.5,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 1.0), (102.0, 1.0)],
            is_buy=True,
        )
        assert result.total_cost == pytest.approx(
            result.fees + result.slippage_cost
        )


# ============================================================
# match_marketable_limit_ioc
# ============================================================

class TestMatchMarketableLimitIOC:
    """Тесты IOC-ордеров с ограничением цены."""

    def test_empty_book_skipped(self, engine):
        result = engine.match_marketable_limit_ioc(
            order_qty=1.0,
            limit_price=100.0,
            bids=[],
            asks=[],
            is_buy=True,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "EMPTY_BOOK_SIDE"

    def test_buy_within_limit_full_fill(self, engine):
        """Лимит выше лучших цен → полное исполнение."""
        result = engine.match_marketable_limit_ioc(
            order_qty=1.0,
            limit_price=100.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED
        assert result.filled_qty == pytest.approx(1.0)
        assert result.avg_fill_price == pytest.approx(100.0)

    def test_buy_price_not_reached_rejected(self, engine):
        """Лимит ниже лучших цен → не исполняется."""
        result = engine.match_marketable_limit_ioc(
            order_qty=1.0,
            limit_price=99.0,
            bids=[(98.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.REJECTED
        assert result.reject_reason == "PRICE_NOT_REACHED"

    def test_buy_partial_within_limit(self, engine):
        """Уровни за лимитом не берутся → частичное исполнение."""
        result = engine.match_marketable_limit_ioc(
            order_qty=2.0,
            limit_price=100.5,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 1.0), (101.0, 1.0)],
            is_buy=True,
        )
        # Берём только уровень 100 (101 > 100.5)
        assert result.status == MatchStatus.PARTIALLY_FILLED
        assert result.filled_qty == pytest.approx(1.0)
        assert result.avg_fill_price == pytest.approx(100.0)

    def test_sell_within_limit(self, engine):
        """SELL: bids >= limit."""
        result = engine.match_marketable_limit_ioc(
            order_qty=1.0,
            limit_price=99.0,
            bids=[(99.0, 2.0)],
            asks=[(101.0, 1.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.FILLED
        assert result.filled_qty == pytest.approx(1.0)
        assert result.avg_fill_price == pytest.approx(99.0)

    def test_sell_price_not_reached(self, engine):
        """SELL: лимит выше всех bids → не исполняется."""
        result = engine.match_marketable_limit_ioc(
            order_qty=1.0,
            limit_price=100.0,
            bids=[(99.0, 2.0)],
            asks=[(101.0, 1.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.REJECTED
        assert result.reject_reason == "PRICE_NOT_REACHED"


# ============================================================
# match_stop_market
# ============================================================

class TestMatchStopMarket:
    """Тесты STOP_MARKET-ордеров."""

    def test_buy_stop_triggered(self, engine):
        """BUY стоп: best_ask >= stop_price → триггер."""
        result = engine.match_stop_market(
            order_qty=1.0,
            stop_price=100.0,
            best_bid=99.0,
            best_ask=100.5,
            bids=[(99.0, 1.0)],
            asks=[(100.5, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED
        assert result.filled_qty == pytest.approx(1.0)

    def test_buy_stop_not_triggered(self, engine):
        """BUY стоп: best_ask < stop_price → SKIPPED."""
        result = engine.match_stop_market(
            order_qty=1.0,
            stop_price=101.0,
            best_bid=99.0,
            best_ask=100.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "STOP_NOT_TRIGGERED"

    def test_sell_stop_triggered(self, engine):
        """SELL стоп: best_bid <= stop_price → триггер."""
        result = engine.match_stop_market(
            order_qty=1.0,
            stop_price=99.0,
            best_bid=98.5,
            best_ask=100.0,
            bids=[(98.5, 2.0)],
            asks=[(100.0, 1.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.FILLED
        assert result.avg_fill_price == pytest.approx(98.5)

    def test_sell_stop_not_triggered(self, engine):
        """SELL стоп: best_bid > stop_price → SKIPPED."""
        result = engine.match_stop_market(
            order_qty=1.0,
            stop_price=98.0,
            best_bid=99.0,
            best_ask=100.0,
            bids=[(99.0, 2.0)],
            asks=[(100.0, 1.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "STOP_NOT_TRIGGERED"

    def test_stop_exact_boundary_triggers(self, engine):
        """Точное совпадение с триггером → срабатывает."""
        result = engine.match_stop_market(
            order_qty=1.0,
            stop_price=100.0,
            best_bid=99.0,
            best_ask=100.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED


# ============================================================
# match_limit (мейкер)
# ============================================================

class TestMatchLimit:
    """Тесты LIMIT-ордеров (мейкер)."""

    def test_buy_limit_hit(self, engine):
        """BUY limit >= best ask → исполнение."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=100.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED
        assert result.avg_fill_price == pytest.approx(100.0)
        assert result.filled_qty == pytest.approx(1.0)

    def test_buy_limit_not_hit_skipped(self, engine):
        """BUY limit < best ask → SKIPPED."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=99.0,
            bids=[(98.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "LIMIT_NOT_HIT"

    def test_sell_limit_hit(self, engine):
        """SELL limit <= best bid → исполнение."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=99.0,
            bids=[(99.0, 2.0)],
            asks=[(101.0, 1.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.FILLED
        assert result.avg_fill_price == pytest.approx(99.0)

    def test_sell_limit_not_hit_skipped(self, engine):
        """SELL limit > best bid → SKIPPED."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=100.0,
            bids=[(99.0, 2.0)],
            asks=[(101.0, 1.0)],
            is_buy=False,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "LIMIT_NOT_HIT"

    def test_limit_maker_fee(self, engine):
        """Мейкерская комиссия = 0.02%."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=100.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        # notional = 1.0 * 100.0 = 100.0
        # maker fee = 100.0 * 0.02 / 100 = 0.02
        assert result.fees == pytest.approx(0.02)

    def test_limit_no_slippage(self, engine):
        """Мейкер без проскальзывания."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=100.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        assert result.slippage_ticks == 0.0
        assert result.slippage_cost == 0.0

    def test_limit_partial_fill(self, engine):
        """Недостаточно ликвидности по лимиту → частично."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=100.5,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 0.5), (101.0, 1.0)],
            is_buy=True,
        )
        # Только уровень 100 <= 100.5 → доступно 0.5
        assert result.status == MatchStatus.PARTIALLY_FILLED
        assert result.filled_qty == pytest.approx(0.5)
        # Цена всегда = limit (мейкер)
        assert result.avg_fill_price == pytest.approx(100.5)

    def test_limit_empty_book_skipped(self, engine):
        """Пустая книга → SKIPPED."""
        result = engine.match_limit(
            order_qty=1.0,
            limit_price=100.0,
            bids=[],
            asks=[],
            is_buy=True,
        )
        assert result.status == MatchStatus.SKIPPED
        assert result.reject_reason == "LIMIT_NOT_HIT"


# ============================================================
# FillResult свойства
# ============================================================

class TestFillResultProperties:
    """Тесты свойств FillResult."""

    def test_is_filled_true(self):
        result = FillResult(status=MatchStatus.FILLED)
        assert result.is_filled
        assert not result.is_partial

    def test_is_partial_true(self):
        result = FillResult(status=MatchStatus.PARTIALLY_FILLED)
        assert result.is_partial
        assert not result.is_filled

    def test_rejected_not_filled(self):
        result = FillResult(status=MatchStatus.REJECTED)
        assert not result.is_filled
        assert not result.is_partial

    def test_skipped_not_filled(self):
        result = FillResult(status=MatchStatus.SKIPPED)
        assert not result.is_filled
        assert not result.is_partial


# ============================================================
# Вспомогательное: _walk_book через публичные методы
# ============================================================

class TestWalkBookEdgeCases:
    """Граничные случаи прохода по книге."""

    def test_skips_zero_qty_levels(self, engine):
        """Уровни с нулевым объёмом пропускаются."""
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 0.0), (101.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED
        assert result.avg_fill_price == pytest.approx(101.0)

    def test_skips_zero_price_levels(self, engine):
        """Уровни с нулевой ценой пропускаются."""
        result = engine.match_market(
            order_qty=1.0,
            bids=[(99.0, 1.0)],
            asks=[(0.0, 1.0), (100.0, 2.0)],
            is_buy=True,
        )
        assert result.status == MatchStatus.FILLED
        assert result.avg_fill_price == pytest.approx(100.0)

    def test_zero_order_qty_rejected(self, engine):
        """Нулевой объём ордера → не исполняется."""
        result = engine.match_market(
            order_qty=0.0,
            bids=[(99.0, 1.0)],
            asks=[(100.0, 2.0)],
            is_buy=True,
        )
        # filled_qty = 0 → REJECTED с NO_LIQUIDITY
        assert result.status == MatchStatus.REJECTED
        assert result.filled_qty == 0.0
