"""
Загрузка и валидация конфигурации приложения.

Читает YAML-конфиг и предоставляет типизированные структуры
для каждого модуля системы. Все секции имеют дефолтные значения,
поэтому конфиг может содержать только переопределения.

Структура конфига:
    symbols: [BTCUSDT, ETHUSDT]
    collector: {...}
    strategy: {...}
    risk: {...}
    execution: {...}
    journal: {...}
    backtest: {...}

Использование:
    from proscalper.core.config import load_app_config
    
    config = load_app_config("configs/default.yaml")
    
    # Доступ к секциям
    for symbol in config.symbols:
        tick_size = ...
    
    if config.risk.max_open_positions >= 1:
        ...

Принципы:
- все секции типизированы (dataclass)
- дефолтные значения для каждого поля
- валидация при загрузке (быстрые ошибки)
- секреты НЕ хранятся в конфиге (только через core/secrets.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ============================================================
# Исключения
# ============================================================

class ConfigError(Exception):
    """Ошибка загрузки или валидации конфигурации."""
    pass


# ============================================================
# Секция: коллектор
# ============================================================

@dataclass(frozen=True)
class CollectorSection:
    """Конфигурация сборщика рыночных данных."""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    data_dir: str = "./data/binance_futures"
    depth_update_speed: str = "100ms"
    snapshot_limit: int = 1000
    batch_size: int = 1000
    flush_interval_ms: int = 200
    queue_size: int = 50_000
    history_interval: str = "5m"
    history_hours: float = 24.0


# ============================================================
# Секция: стратегия
# ============================================================

@dataclass(frozen=True)
class StrategySection:
    """Конфигурация торговой стратегии."""
    # Уровни
    level_timeframe: str = "5m"
    level_min_touches: int = 2
    level_max_age_hours: float = 48.0
    
    # Пробой
    breakout_min_approach_ms: int = 300
    breakout_max_approach_ms: int = 30_000
    breakout_confirmation_window_ms: int = 200
    breakout_min_compression_bars: int = 5
    
    # Импульс
    impulse_min_volume_zscore: float = 2.0
    impulse_min_delta_ratio: float = 0.6
    
    # Ложный пробой
    false_breakout_reentry_ms: int = 500
    false_breakout_max_depth_ticks: int = 10
    
    # Ретест
    retest_max_wait_ms: int = 60_000
    retest_min_hold_ms: int = 300
    
    # Фильтры
    max_spoof_score: float = 0.5
    min_cross_confirmation: float = 0.3
    min_imbalance: float = 0.25


# ============================================================
# Секция: риски
# ============================================================

@dataclass(frozen=True)
class RiskSection:
    """Конфигурация риск-менеджмента."""
    # Лимиты
    max_open_positions: int = 1
    max_daily_trades: int = 20
    max_daily_loss_usdt: float = 100.0
    max_notional_per_trade: float = 1000.0
    min_notional_per_trade: float = 5.0
    
    # Размер позиции
    risk_per_trade_pct: float = 0.5       # % от капитала на сделку
    initial_capital: float = 10_000.0
    
    # Стопы
    stop_buffer_ticks: int = 2
    stop_buffer_pct: float = 0.0002
    break_even_at_r: float = 1.0
    trailing_activation_r: float = 1.5
    trailing_distance_ticks: int = 5
    
    # Аварийные лимиты
    max_unprotected_ms: int = 120
    emergency_flatten_on_failure: bool = True


# ============================================================
# Секция: исполнение
# ============================================================

@dataclass(frozen=True)
class ExecutionSection:
    """Конфигурация исполнения ордеров."""
    # Режим: "paper" | "live"
    mode: str = "paper"
    
    # Биржа
    exchange: str = "binance_futures"
    
    # Вход
    entry_order_type: str = "MARKETABLE_LIMIT_IOC"
    max_slippage_ticks: int = 4
    
    # Стопы
    stop_working_type: str = "CONTRACT_PRICE"
    
    # Таймауты
    max_entry_wait_ms: int = 5000
    stop_retry_interval_ms: int = 20
    max_stop_retries: int = 5
    
    # Бэктест-режим
    latency_region: str = "TOKYO"
    taker_fee_pct: float = 0.04
    maker_fee_pct: float = 0.02


# ============================================================
# Секция: журналирование
# ============================================================

@dataclass(frozen=True)
class JournalSection:
    """Конфигурация журналов."""
    decision_journal_dir: str = "./logs/journal"
    incident_journal_dir: str = "./logs/incidents"
    trade_journal_dir: str = "./logs/trades"
    
    # Размеры очередей
    decision_queue_size: int = 100_000
    incident_queue_size: int = 10_000
    
    # Интервалы записи
    decision_flush_ms: int = 200
    incident_flush_ms: int = 100
    
    # Логирование в консоль
    log_level: str = "INFO"
    log_format: str = "text"   # "text" | "json"


# ============================================================
# Секция: бэктест
# ============================================================

@dataclass(frozen=True)
class BacktestSection:
    """Конфигурация бэктестинга."""
    source_dir: str = "./data/binance_futures"
    use_parquet: bool = False
    tick_size: float = 0.1
    depth_for_matching: int = 50
    initial_balance: float = 10_000.0
    latency_region: str = "TOKYO"


# ============================================================
# Объединяющий конфиг
# ============================================================

@dataclass(frozen=True)
class AppConfig:
    """
    Полная конфигурация приложения.
    
    Все секции имеют дефолтные значения, поэтому можно
    создать конфиг только с нужными переопределениями.
    """
    config_path: str = ""
    
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    
    collector: CollectorSection = field(default_factory=CollectorSection)
    strategy: StrategySection = field(default_factory=StrategySection)
    risk: RiskSection = field(default_factory=RiskSection)
    execution: ExecutionSection = field(default_factory=ExecutionSection)
    journal: JournalSection = field(default_factory=JournalSection)
    backtest: BacktestSection = field(default_factory=BacktestSection)
    
    def validate(self) -> None:
        """Проверяет консистентность конфигурации."""
        errors: List[str] = []
        
        if not self.symbols:
            errors.append("symbols не может быть пустым")
        
        if self.risk.max_open_positions < 1:
            errors.append("risk.max_open_positions должен быть >= 1")
        
        if self.risk.risk_per_trade_pct <= 0 or self.risk.risk_per_trade_pct > 10:
            errors.append("risk.risk_per_trade_pct должен быть в диапазоне (0, 10]")
        
        if self.execution.mode not in ("paper", "live"):
            errors.append("execution.mode должен быть 'paper' или 'live'")
        
        if self.execution.entry_order_type not in (
            "MARKETABLE_LIMIT_IOC",
            "MARKET",
            "LIMIT",
        ):
            errors.append(
                "execution.entry_order_type должен быть "
                "MARKETABLE_LIMIT_IOC, MARKET или LIMIT"
            )
        
        if self.risk.max_unprotected_ms <= 0:
            errors.append("risk.max_unprotected_ms должен быть > 0")
        
        if self.journal.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            errors.append("journal.log_level должен быть DEBUG/INFO/WARNING/ERROR")
        
        if errors:
            raise ConfigError(
                "Ошибки конфигурации:\n  - " + "\n  - ".join(errors)
            )


# ============================================================
# Загрузка
# ============================================================

def _build_section(section_cls, data: Optional[Dict[str, Any]]):
    """
    Строит секцию из dict, игнорируя неизвестные ключи.
    Неизвестные ключи логируются как предупреждение (не ошибка),
    чтобы конфиги разных версий были совместимы.
    """
    if data is None:
        return section_cls()
    
    if not isinstance(data, dict):
        raise ConfigError(f"Секция должна быть объектом, получено: {type(data)}")
    
    # Берём только известные поля
    known_fields = {f.name for f in section_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    
    try:
        return section_cls(**filtered)
    except TypeError as exc:
        raise ConfigError(f"Ошибка построения секции {section_cls.__name__}: {exc}")


def load_app_config(config_path: str) -> AppConfig:
    """
    Загружает конфигурацию из YAML-файла.
    
    Если файл не найден — возвращает конфиг с дефолтами.
    Это позволяет запускать систему без конфига для разработки.
    
    Args:
        config_path: путь к YAML-файлу
        
    Returns:
        AppConfig с заполненными секциями
        
    Raises:
        ConfigError: при ошибках валидации
    """
    path = Path(config_path)
    
    if not path.exists():
        # Возвращаем дефолтный конфиг с пометкой пути
        return AppConfig(config_path=str(config_path))
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Ошибка парсинга YAML {config_path}: {exc}")
    
    if not isinstance(raw, dict):
        raise ConfigError(f"Корень конфига должен быть объектом: {config_path}")
    
    # Собираем конфиг
    config = AppConfig(
        config_path=str(config_path),
        symbols=list(raw.get("symbols", ["BTCUSDT", "ETHUSDT"])),
        collector=_build_section(CollectorSection, raw.get("collector")),
        strategy=_build_section(StrategySection, raw.get("strategy")),
        risk=_build_section(RiskSection, raw.get("risk")),
        execution=_build_section(ExecutionSection, raw.get("execution")),
        journal=_build_section(JournalSection, raw.get("journal")),
        backtest=_build_section(BacktestSection, raw.get("backtest")),
    )
    
    # Синхронизируем символы между корневым уровнем и коллектором
    # (если в коллекторе не указаны явно, берём из корня)
    if config.collector.symbols == ["BTCUSDT", "ETHUSDT"] and config.symbols:
        # Пересоздаём секцию коллектора с символами из корня
        collector_dict = {
            "symbols": config.symbols,
            "data_dir": config.collector.data_dir,
            "depth_update_speed": config.collector.depth_update_speed,
            "snapshot_limit": config.collector.snapshot_limit,
            "batch_size": config.collector.batch_size,
            "flush_interval_ms": config.collector.flush_interval_ms,
            "queue_size": config.collector.queue_size,
            "history_interval": config.collector.history_interval,
            "history_hours": config.collector.history_hours,
        }
        object.__setattr__(config, "collector", CollectorSection(**collector_dict))
    
    config.validate()
    return config


def load_collector_config(config_path: str) -> CollectorSection:
    """
    Загружает только секцию коллектора.
    Удобно для app/collector.py, который не нуждается в полном конфиге.
    """
    path = Path(config_path)
    
    if not path.exists():
        return CollectorSection()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Ошибка парсинга YAML {config_path}: {exc}")
    
    # Поддерживаем два формата:
    # 1. {collector: {...}} — полный конфиг
    # 2. {...} — только секция коллектора
    if "collector" in raw:
        section_data = raw.get("collector")
    else:
        section_data = raw
    
    return _build_section(CollectorSection, section_data)