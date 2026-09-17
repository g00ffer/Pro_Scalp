"""
Телеметрия системы (метрики для мониторинга).

Простой встроенный сборщик метрик без внешних зависимостей.
Для продакшена можно подключить экспорт в Prometheus, но для
разработки и бэктеста достаточно встроенного.

Типы метрик:
- Counter: монотонно растущий счётчик
- Gauge: текущее значение (может расти и падать)
- Histogram: распределение значений

Использование:
    from proscalper.core.telemetry import telemetry

    # Счётчик
    telemetry.inc("orders.submitted", tags={"symbol": "BTCUSDT"})
    telemetry.inc("orders.filled")

    # Gauge
    telemetry.set("positions.open", 2)
    telemetry.set("latency.last_ms", 4.2)

    # Histogram
    telemetry.observe("fill.latency_ms", 3.5)
    telemetry.observe("fill.latency_ms", 4.1)

    # Дамп в файл
    telemetry.dump("logs/metrics/latest.json")

    # Сводка в консоль
    print(telemetry.summary())

Принципы:
- потокобезопасность через threading.Lock
- минимальные аллокации в hot path
- не блокирует вызывающий код
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Типы метрик
# ============================================================

@dataclass
class Counter:
    """Монотонно растущий счётчик."""
    name: str
    value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    created_ts_ns: int = field(default_factory=time.time_ns)
    last_updated_ts_ns: int = field(default_factory=time.time_ns)

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount
        self.last_updated_ts_ns = time.time_ns()


@dataclass
class Gauge:
    """Текущее значение."""
    name: str
    value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    created_ts_ns: int = field(default_factory=time.time_ns)
    last_updated_ts_ns: int = field(default_factory=time.time_ns)

    def set(self, value: float) -> None:
        self.value = value
        self.last_updated_ts_ns = time.time_ns()


@dataclass
class Histogram:
    """Распределение значений."""
    name: str
    values: List[float] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    created_ts_ns: int = field(default_factory=time.time_ns)
    last_updated_ts_ns: int = field(default_factory=time.time_ns)

    # Ограничение на количество значений (защита от утечки)
    max_values: int = 10_000

    def observe(self, value: float) -> None:
        self.values.append(value)
        self.last_updated_ts_ns = time.time_ns()
        # Ограничиваем размер
        if len(self.values) > self.max_values:
            self.values = self.values[-self.max_values:]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def sum(self) -> float:
        return sum(self.values)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self.sum / len(self.values)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    def percentile(self, p: float) -> float:
        """Перцентиль (0-100)."""
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100.0)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def reset(self) -> None:
        self.values.clear()


# ============================================================
# Сборщик телеметрии
# ============================================================

class Telemetry:
    """
    Центральный сборщик метрик.

    Потокобезопасный, не блокирует вызывающий код.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._created_ts_ns = time.time_ns()

    # ============================================
    # Counter
    # ============================================

    def inc(
        self,
        name: str,
        amount: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Инкрементирует счётчик."""
        key = self._key(name, tags)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(
                    name=name, tags=dict(tags or {})
                )
            self._counters[key].inc(amount)

    def get_counter(
        self, name: str, tags: Optional[Dict[str, str]] = None
    ) -> float:
        """Возвращает значение счётчика."""
        key = self._key(name, tags)
        with self._lock:
            counter = self._counters.get(key)
            return counter.value if counter else 0.0

    # ============================================
    # Gauge
    # ============================================

    def set(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Устанавливает значение."""
        key = self._key(name, tags)
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(
                    name=name, tags=dict(tags or {})
                )
            self._gauges[key].set(value)

    def get_gauge(
        self, name: str, tags: Optional[Dict[str, str]] = None
    ) -> float:
        """Возвращает значение."""
        key = self._key(name, tags)
        with self._lock:
            gauge = self._gauges.get(key)
            return gauge.value if gauge else 0.0

    # ============================================
    # Histogram
    # ============================================

    def observe(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Добавляет значение в гистограмму."""
        key = self._key(name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram(
                    name=name, tags=dict(tags or {})
                )
            self._histograms[key].observe(value)

    def get_histogram_stats(
        self, name: str, tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Возвращает статистику гистограммы."""
        key = self._key(name, tags)
        with self._lock:
            h = self._histograms.get(key)
            if h is None:
                return {}
            return {
                "count": float(h.count),
                "sum": h.sum,
                "mean": h.mean,
                "min": h.min,
                "max": h.max,
                "p50": h.percentile(50),
                "p90": h.percentile(90),
                "p99": h.percentile(99),
            }

    # ============================================
    # Экспорт
    # ============================================

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация всех метрик в dict."""
        with self._lock:
            return {
                "uptime_sec": (
                    (time.time_ns() - self._created_ts_ns) / 1e9
                ),
                "counters": {
                    k: {
                        "name": c.name,
                        "value": c.value,
                        "tags": c.tags,
                    }
                    for k, c in self._counters.items()
                },
                "gauges": {
                    k: {
                        "name": g.name,
                        "value": g.value,
                        "tags": g.tags,
                    }
                    for k, g in self._gauges.items()
                },
                "histograms": {
                    k: {
                        "name": h.name,
                        "count": h.count,
                        "sum": h.sum,
                        "mean": h.mean,
                        "min": h.min,
                        "max": h.max,
                        "p50": h.percentile(50),
                        "p90": h.percentile(90),
                        "p99": h.percentile(99),
                        "tags": h.tags,
                    }
                    for k, h in self._histograms.items()
                },
            }

    def dump(self, path: str) -> None:
        """Сохраняет метрики в JSON-файл."""
        data = self.to_dict()
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def summary(self) -> str:
        """Текстовая сводка для консоли."""
        data = self.to_dict()
        lines = [
            f"=== Telemetry (uptime: {data['uptime_sec']:.0f}s) ===",
        ]

        if data["counters"]:
            lines.append("\nCounters:")
            for key, c in sorted(data["counters"].items()):
                lines.append(f"  {c['name']}: {c['value']:.0f}")

        if data["gauges"]:
            lines.append("\nGauges:")
            for key, g in sorted(data["gauges"].items()):
                lines.append(f"  {g['name']}: {g['value']:.2f}")

        if data["histograms"]:
            lines.append("\nHistograms:")
            for key, h in sorted(data["histograms"].items()):
                lines.append(
                    f"  {h['name']}: n={h['count']:.0f}, "
                    f"mean={h['mean']:.2f}, "
                    f"p50={h['p50']:.2f}, "
                    f"p99={h['p99']:.2f}"
                )

        return "\n".join(lines)

    # ============================================
    # Сброс
    # ============================================

    def reset(self) -> None:
        """Полный сброс всех метрик."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._created_ts_ns = time.time_ns()

    def reset_histograms(self) -> None:
        """Сброс только гистограмм (для периодических окон)."""
        with self._lock:
            for h in self._histograms.values():
                h.reset()

    # ============================================
    # Внутреннее
    # ============================================

    def _key(
        self, name: str, tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Генерирует уникальный ключ для метрики с тегами."""
        if not tags:
            return name
        tag_str = ",".join(
            f"{k}={v}" for k, v in sorted(tags.items())
        )
        return f"{name}[{tag_str}]"


# ============================================================
# Глобальный экземпляр
# ============================================================

telemetry = Telemetry()