"""
Модель задержек бэктеста (Latency Model).

В live-торговле между решением робота и фактическим исполнением
проходит время:
- обработка сигнала внутри робота (микросекунды)
- отправка ордера на биржу (сеть, VPS→exchange)
- постановка в очередь матчинг-движка
- ack от биржи (подтверждение получения)
- fill (исполнение)

В скальпинге это критично: 50-150мс могут превратить
прибыльную сделку в убыточную из-за проскальзывания.

Модуль предоставляет:
- LatencyModel: интерфейс
- ConstantLatency: фиксированные задержки
- ProfileLatency: разные задержки по регионам (Tokyo, Frankfurt, ...)
- StochasticLatency: распределение (для Монте-Карло)

Принципы:
- детерминированность (по умолчанию)
- конфигурируемые профили
- интеграция с engine для учёта ts сдвигов

Используется в связке с:
- backtest/matching.py (учёт latency в момент fill)
- backtest/engine.py (сдвиг ts событий)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


# ============================================================
# Региональные профили
# ============================================================

class Region(str, Enum):
    """Регион, где размещён VPS."""
    TOKYO = "TOKYO"         # ap-northeast-1, оптимально для Binance
    SINGAPORE = "SINGAPORE"
    FRANKFURT = "FRANKFURT"
    VIRGINIA = "VIRGINIA"
    HOME = "HOME"           # домашний интернет (для разработки)


# Профили задержек в миллисекундах (round-trip one-way)
# Значения — ориентировочные, реальные надо замерять на VPS.
_REGION_PROFILES: Dict[Region, Dict[str, float]] = {
    Region.TOKYO: {
        "signal_processing_ms": 0.5,
        "network_roundtrip_ms": 3.0,
        "exchange_matching_ms": 1.0,
        "ack_ms": 2.0,
    },
    Region.SINGAPORE: {
        "signal_processing_ms": 0.5,
        "network_roundtrip_ms": 8.0,
        "exchange_matching_ms": 1.0,
        "ack_ms": 3.0,
    },
    Region.FRANKFURT: {
        "signal_processing_ms": 0.7,
        "network_roundtrip_ms": 25.0,
        "exchange_matching_ms": 1.5,
        "ack_ms": 5.0,
    },
    Region.VIRGINIA: {
        "signal_processing_ms": 0.7,
        "network_roundtrip_ms": 120.0,
        "exchange_matching_ms": 2.0,
        "ack_ms": 8.0,
    },
    Region.HOME: {
        "signal_processing_ms": 1.0,
        "network_roundtrip_ms": 45.0,
        "exchange_matching_ms": 2.0,
        "ack_ms": 10.0,
    },
}


# ============================================================
# Снимок задержек для одной операции
# ============================================================

@dataclass(frozen=True)
class LatencyBreakdown:
    """Разбивка задержек одной операции."""
    signal_processing_ms: float
    network_roundtrip_ms: float
    exchange_matching_ms: float
    ack_ms: float

    @property
    def submit_to_ack_ms(self) -> float:
        """Время от отправки ордера до ack биржи."""
        return self.network_roundtrip_ms / 2.0 + self.ack_ms

    @property
    def submit_to_fill_ms(self) -> float:
        """Время от отправки ордера до фактического fill."""
        return (
            self.network_roundtrip_ms
            + self.exchange_matching_ms
        )

    @property
    def decision_to_fill_ms(self) -> float:
        """Полное время от решения робота до fill."""
        return (
            self.signal_processing_ms
            + self.network_roundtrip_ms
            + self.exchange_matching_ms
        )

    def to_dict(self) -> dict:
        return {
            "signal_processing_ms": self.signal_processing_ms,
            "network_roundtrip_ms": self.network_roundtrip_ms,
            "exchange_matching_ms": self.exchange_matching_ms,
            "ack_ms": self.ack_ms,
            "submit_to_ack_ms": self.submit_to_ack_ms,
            "submit_to_fill_ms": self.submit_to_fill_ms,
            "decision_to_fill_ms": self.decision_to_fill_ms,
        }


# ============================================================
# Модели
# ============================================================

class LatencyModel:
    """
    Абстрактная модель задержек.

    Все наследники должны реализовать sample().
    """

    def sample(self) -> LatencyBreakdown:
        raise NotImplementedError


class ConstantLatency(LatencyModel):
    """
    Детерминированная модель: всегда одна и та же задержка.
    """

    def __init__(
        self,
        signal_processing_ms: float = 0.5,
        network_roundtrip_ms: float = 3.0,
        exchange_matching_ms: float = 1.0,
        ack_ms: float = 2.0,
    ) -> None:
        self._breakdown = LatencyBreakdown(
            signal_processing_ms=signal_processing_ms,
            network_roundtrip_ms=network_roundtrip_ms,
            exchange_matching_ms=exchange_matching_ms,
            ack_ms=ack_ms,
        )

    def sample(self) -> LatencyBreakdown:
        return self._breakdown


class ProfileLatency(LatencyModel):
    """
    Профильная модель: берёт задержки из предопределённых профилей.
    """

    def __init__(self, region: Region = Region.TOKYO) -> None:
        profile = _REGION_PROFILES.get(region)
        if profile is None:
            raise ValueError(f"Неизвестный регион: {region}")

        self._region = region
        self._breakdown = LatencyBreakdown(
            signal_processing_ms=profile["signal_processing_ms"],
            network_roundtrip_ms=profile["network_roundtrip_ms"],
            exchange_matching_ms=profile["exchange_matching_ms"],
            ack_ms=profile["ack_ms"],
        )

    @property
    def region(self) -> Region:
        return self._region

    def sample(self) -> LatencyBreakdown:
        return self._breakdown


@dataclass(frozen=True)
class StochasticLatencyConfig:
    """Конфигурация стохастической модели."""
    # Медианы (базовые задержки)
    signal_processing_ms: float = 0.5
    network_roundtrip_ms: float = 3.0
    exchange_matching_ms: float = 1.0
    ack_ms: float = 2.0

    # Относительный разброс (std = median * jitter_pct)
    jitter_pct: float = 0.3   # 30% разброс
    random_seed: int = 42


class StochasticLatency(LatencyModel):
    """
    Стохастическая модель с гауссовым разбросом.

    Используется для симуляции Монте-Карло и оценки устойчивости
    стратегии к реальному джиттеру сети.

    Гарантия: все задержки ≥ 0.
    """

    def __init__(self, config: Optional[StochasticLatencyConfig] = None) -> None:
        self.config = config or StochasticLatencyConfig()
        self._rng = random.Random(self.config.random_seed)

    def sample(self) -> LatencyBreakdown:
        cfg = self.config

        def _sample(median: float) -> float:
            if median <= 0:
                return 0.0
            std = median * cfg.jitter_pct
            value = self._rng.gauss(median, std)
            return max(0.0, value)

        return LatencyBreakdown(
            signal_processing_ms=_sample(cfg.signal_processing_ms),
            network_roundtrip_ms=_sample(cfg.network_roundtrip_ms),
            exchange_matching_ms=_sample(cfg.exchange_matching_ms),
            ack_ms=_sample(cfg.ack_ms),
        )


# ============================================================
# Фабрика
# ============================================================

def make_latency_model(
    mode: str = "profile",
    region: Region = Region.TOKYO,
    config: Optional[StochasticLatencyConfig] = None,
) -> LatencyModel:
    """
    Фабрика моделей.

    mode:
        "profile" — по региону
        "constant" — фиксированная
        "stochastic" — распределение (для Монте-Карло)
    """
    if mode == "profile":
        return ProfileLatency(region=region)
    if mode == "stochastic":
        return StochasticLatency(config=config)
    if mode == "constant":
        return ConstantLatency()
    raise ValueError(f"Неизвестный mode: {mode}")


# ============================================================
# Утилиты
# ============================================================

def ms_to_ns(ms: float) -> int:
    """Переводит миллисекунды в наносекунды."""
    return int(ms * 1_000_000)


def apply_latency_to_submit_ts(
    submit_ts_ns: int,
    breakdown: LatencyBreakdown,
) -> int:
    """
    Возвращает ts (в ns), когда ордер будет фактически исполнен на бирже
    (или, точнее, момент, когда fill-событие вернётся к нам).

    Применяется как engine для сдвига времени fill-события.
    """
    return submit_ts_ns + ms_to_ns(breakdown.submit_to_fill_ms)