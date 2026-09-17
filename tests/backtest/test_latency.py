"""
Тесты для proscalper.backtest.latency

Запуск:
    python -m pytest tests/backtest/test_latency.py -v
"""
from __future__ import annotations

import pytest

from proscalper.backtest.latency import (
    ConstantLatency,
    LatencyBreakdown,
    LatencyModel,
    ProfileLatency,
    Region,
    StochasticLatency,
    StochasticLatencyConfig,
    apply_latency_to_submit_ts,
    make_latency_model,
    ms_to_ns,
)


# ============================================================
# LatencyBreakdown
# ============================================================

class TestLatencyBreakdown:
    """Тесты разбивки задержек."""

    def test_submit_to_ack(self):
        """submit_to_ack = roundtrip/2 + ack."""
        breakdown = LatencyBreakdown(
            signal_processing_ms=0.5,
            network_roundtrip_ms=3.0,
            exchange_matching_ms=1.0,
            ack_ms=2.0,
        )
        # 3.0 / 2 + 2.0 = 3.5
        assert breakdown.submit_to_ack_ms == pytest.approx(3.5)

    def test_submit_to_fill(self):
        """submit_to_fill = roundtrip + matching."""
        breakdown = LatencyBreakdown(
            signal_processing_ms=0.5,
            network_roundtrip_ms=3.0,
            exchange_matching_ms=1.0,
            ack_ms=2.0,
        )
        # 3.0 + 1.0 = 4.0
        assert breakdown.submit_to_fill_ms == pytest.approx(4.0)

    def test_decision_to_fill(self):
        """decision_to_fill = processing + roundtrip + matching."""
        breakdown = LatencyBreakdown(
            signal_processing_ms=0.5,
            network_roundtrip_ms=3.0,
            exchange_matching_ms=1.0,
            ack_ms=2.0,
        )
        # 0.5 + 3.0 + 1.0 = 4.5
        assert breakdown.decision_to_fill_ms == pytest.approx(4.5)

    def test_to_dict(self):
        """to_dict содержит все поля."""
        breakdown = LatencyBreakdown(
            signal_processing_ms=0.5,
            network_roundtrip_ms=3.0,
            exchange_matching_ms=1.0,
            ack_ms=2.0,
        )
        d = breakdown.to_dict()
        assert d["signal_processing_ms"] == pytest.approx(0.5)
        assert d["network_roundtrip_ms"] == pytest.approx(3.0)
        assert d["exchange_matching_ms"] == pytest.approx(1.0)
        assert d["ack_ms"] == pytest.approx(2.0)
        assert d["submit_to_ack_ms"] == pytest.approx(3.5)
        assert d["submit_to_fill_ms"] == pytest.approx(4.0)
        assert d["decision_to_fill_ms"] == pytest.approx(4.5)

    def test_frozen(self):
        """LatencyBreakdown иммутабелен."""
        breakdown = LatencyBreakdown(
            signal_processing_ms=0.5,
            network_roundtrip_ms=3.0,
            exchange_matching_ms=1.0,
            ack_ms=2.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            breakdown.signal_processing_ms = 1.0


# ============================================================
# ConstantLatency
# ============================================================

class TestConstantLatency:
    """Тесты детерминированной модели."""

    def test_default_values(self):
        """Значения по умолчанию."""
        model = ConstantLatency()
        breakdown = model.sample()
        assert breakdown.signal_processing_ms == pytest.approx(0.5)
        assert breakdown.network_roundtrip_ms == pytest.approx(3.0)
        assert breakdown.exchange_matching_ms == pytest.approx(1.0)
        assert breakdown.ack_ms == pytest.approx(2.0)

    def test_custom_values(self):
        """Кастомные значения."""
        model = ConstantLatency(
            signal_processing_ms=1.0,
            network_roundtrip_ms=10.0,
            exchange_matching_ms=2.0,
            ack_ms=5.0,
        )
        breakdown = model.sample()
        assert breakdown.signal_processing_ms == pytest.approx(1.0)
        assert breakdown.network_roundtrip_ms == pytest.approx(10.0)
        assert breakdown.exchange_matching_ms == pytest.approx(2.0)
        assert breakdown.ack_ms == pytest.approx(5.0)

    def test_deterministic(self):
        """Один и тот же результат при повторных вызовах."""
        model = ConstantLatency()
        b1 = model.sample()
        b2 = model.sample()
        assert b1 == b2


# ============================================================
# ProfileLatency
# ============================================================

class TestProfileLatency:
    """Тесты профильной модели."""

    def test_tokyo_profile(self):
        """Профиль Токио (быстрый)."""
        model = ProfileLatency(region=Region.TOKYO)
        breakdown = model.sample()
        assert breakdown.signal_processing_ms == pytest.approx(0.5)
        assert breakdown.network_roundtrip_ms == pytest.approx(3.0)
        assert breakdown.exchange_matching_ms == pytest.approx(1.0)
        assert breakdown.ack_ms == pytest.approx(2.0)

    def test_home_profile_slower_than_tokyo(self):
        """Домашний интернет медленнее Токио."""
        tokyo = ProfileLatency(region=Region.TOKYO).sample()
        home = ProfileLatency(region=Region.HOME).sample()
        assert home.network_roundtrip_ms > tokyo.network_roundtrip_ms

    def test_region_property(self):
        """Свойство region."""
        model = ProfileLatency(region=Region.FRANKFURT)
        assert model.region == Region.FRANKFURT

    def test_all_regions_available(self):
        """Все регионы доступны и валидны."""
        for region in Region:
            model = ProfileLatency(region=region)
            breakdown = model.sample()
            assert breakdown.signal_processing_ms >= 0
            assert breakdown.network_roundtrip_ms >= 0
            assert breakdown.exchange_matching_ms >= 0
            assert breakdown.ack_ms >= 0

    def test_default_region_tokyo(self):
        """По умолчанию — Токио."""
        model = ProfileLatency()
        assert model.region == Region.TOKYO

    def test_deterministic(self):
        """Один и тот же результат при повторных вызовах."""
        model = ProfileLatency(region=Region.TOKYO)
        b1 = model.sample()
        b2 = model.sample()
        assert b1 == b2


# ============================================================
# StochasticLatency
# ============================================================

class TestStochasticLatency:
    """Тесты стохастической модели."""

    def test_deterministic_with_seed(self):
        """Один и тот же сид → одинаковые результаты."""
        model_1 = StochasticLatency(StochasticLatencyConfig(random_seed=42))
        model_2 = StochasticLatency(StochasticLatencyConfig(random_seed=42))
        b1 = model_1.sample()
        b2 = model_2.sample()
        assert b1.signal_processing_ms == pytest.approx(b2.signal_processing_ms)
        assert b1.network_roundtrip_ms == pytest.approx(b2.network_roundtrip_ms)
        assert b1.exchange_matching_ms == pytest.approx(b2.exchange_matching_ms)
        assert b1.ack_ms == pytest.approx(b2.ack_ms)

    def test_different_seeds_different_results(self):
        """Разные сиды → разные результаты."""
        model_1 = StochasticLatency(StochasticLatencyConfig(random_seed=42))
        model_2 = StochasticLatency(StochasticLatencyConfig(random_seed=123))
        b1 = model_1.sample()
        b2 = model_2.sample()
        differs = (
            b1.signal_processing_ms != b2.signal_processing_ms
            or b1.network_roundtrip_ms != b2.network_roundtrip_ms
            or b1.exchange_matching_ms != b2.exchange_matching_ms
            or b1.ack_ms != b2.ack_ms
        )
        assert differs

    def test_non_negative_values(self):
        """Все задержки неотрицательны даже при большом разбросе."""
        model = StochasticLatency(
            StochasticLatencyConfig(jitter_pct=0.9, random_seed=7)
        )
        for _ in range(100):
            breakdown = model.sample()
            assert breakdown.signal_processing_ms >= 0.0
            assert breakdown.network_roundtrip_ms >= 0.0
            assert breakdown.exchange_matching_ms >= 0.0
            assert breakdown.ack_ms >= 0.0

    def test_zero_median_gives_zero(self):
        """Нулевая медиана → всегда 0."""
        config = StochasticLatencyConfig(
            signal_processing_ms=0.0,
            network_roundtrip_ms=0.0,
            exchange_matching_ms=0.0,
            ack_ms=0.0,
        )
        model = StochasticLatency(config)
        for _ in range(10):
            breakdown = model.sample()
            assert breakdown.signal_processing_ms == 0.0
            assert breakdown.network_roundtrip_ms == 0.0
            assert breakdown.exchange_matching_ms == 0.0
            assert breakdown.ack_ms == 0.0

    def test_jitter_affects_variance(self):
        """Больший разброс → больше вариация."""
        low_jitter = StochasticLatency(
            StochasticLatencyConfig(jitter_pct=0.01, random_seed=1)
        )
        high_jitter = StochasticLatency(
            StochasticLatencyConfig(jitter_pct=0.9, random_seed=1)
        )

        low_samples = [
            low_jitter.sample().network_roundtrip_ms for _ in range(50)
        ]
        high_samples = [
            high_jitter.sample().network_roundtrip_ms for _ in range(50)
        ]

        low_mean = sum(low_samples) / 50
        high_mean = sum(high_samples) / 50

        low_variance = sum((x - low_mean) ** 2 for x in low_samples) / 50
        high_variance = sum((x - high_mean) ** 2 for x in high_samples) / 50

        assert high_variance > low_variance


# ============================================================
# Фабрика
# ============================================================

class TestMakeLatencyModel:
    """Тесты фабрики моделей."""

    def test_profile_mode(self):
        model = make_latency_model(mode="profile", region=Region.TOKYO)
        assert isinstance(model, ProfileLatency)

    def test_constant_mode(self):
        model = make_latency_model(mode="constant")
        assert isinstance(model, ConstantLatency)

    def test_stochastic_mode(self):
        model = make_latency_model(mode="stochastic")
        assert isinstance(model, StochasticLatency)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            make_latency_model(mode="unknown")


# ============================================================
# Утилиты
# ============================================================

class TestUtils:
    """Тесты утилит."""

    def test_ms_to_ns(self):
        assert ms_to_ns(1.0) == 1_000_000
        assert ms_to_ns(0.5) == 500_000
        assert ms_to_ns(0.0) == 0
        assert ms_to_ns(100.0) == 100_000_000

    def test_apply_latency_to_submit_ts(self):
        """Сдвиг ts на submit_to_fill_ms."""
        breakdown = LatencyBreakdown(
            signal_processing_ms=0.5,
            network_roundtrip_ms=3.0,
            exchange_matching_ms=1.0,
            ack_ms=2.0,
        )
        # submit_to_fill = 3.0 + 1.0 = 4.0 ms = 4_000_000 ns
        submit_ts = 1_000_000_000
        fill_ts = apply_latency_to_submit_ts(submit_ts, breakdown)
        assert fill_ts == 1_000_000_000 + 4_000_000

    def test_apply_latency_zero(self):
        """Нулевые задержки не сдвигают."""
        breakdown = LatencyBreakdown(
            signal_processing_ms=0.0,
            network_roundtrip_ms=0.0,
            exchange_matching_ms=0.0,
            ack_ms=0.0,
        )
        submit_ts = 1_000_000_000
        fill_ts = apply_latency_to_submit_ts(submit_ts, breakdown)
        assert fill_ts == submit_ts
