"""
Replay Runner: прогон бэктеста на записанных рыночных данных.

Читает записанные коллекционером данные (через delta_writer + keyframe_writer),
восстанавливает стаканы через ReplayBuilder и прогоняет через
событийный движок бэктеста.

Отличие от бэктеста на синтетических данных:
- Используем РЕАЛЬНЫЕ рыночные события
- Учитываем реальные гэпы и рассинхроны
- Получаем точную картину проскальзывания

Использование:
    # Прогон за конкретную дату
    python -m proscalper.app.replay_runner configs/default.yaml \
        --symbol BTCUSDT --date 2026-09-17

    # Прогон за все доступные даты
    python -m proscalper.app.replay_runner configs/default.yaml \
        --symbol BTCUSDT --all-dates

    # С указанием режима (по умолчанию 'check' — без стратегии)
    python -m proscalper.app.replay_runner configs/default.yaml \
        --symbol BTCUSDT --date 2026-09-17 --mode check

Режимы:
- check: прогон инфраструктуры без стратегии (проверяем данные)
- strategy: прогон с простой стратегией пробоя уровня

Используется в связке с:
- core/config.py (конфигурация)
- storage/replay_builder.py (восстановление стакана)
- backtest/engine.py (событийный движок)
- backtest/report.py (отчёты)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from proscalper.backtest.engine import (
    BacktestContext,
    BacktestEngine,
    BacktestEngineConfig,
)
from proscalper.backtest.report import BacktestReport, ReportConfig
from proscalper.core.config import load_app_config
from proscalper.core.types import OrderSide
from proscalper.backtest.latency import Region


# ============================================================
# Простая стратегия для проверки (пробой уровня)
# ============================================================

class SimpleBreakoutStrategy:
    """
    Простая стратегия пробоя для проверки инфраструктуры.

    Логика:
    - Отслеживает максимум/минимум за последние 60 секунд
    - Если текущая цена пробивает максимум → покупка
    - Если текущая цена пробивает минимум → продажа
    - Закрытие позиции через 30 секунд или при обратном пробое

    Это НЕ боевая стратегия, только для проверки бэктестера.
    """

    def __init__(self, lookback_sec: float = 60.0, hold_sec: float = 30.0):
        self.lookback_ns = int(lookback_sec * 1_000_000_000)
        self.hold_ns = int(hold_sec * 1_000_000_000)
        self._prices: List[tuple] = []  # (ts_ns, price)
        self._last_entry_ts_ns: int = 0

    def on_market_update(
        self,
        ctx: BacktestContext,
        engine: BacktestEngine,
    ) -> None:
        # Добавляем текущую цену в историю
        mid = ctx.mid_price
        if mid is None or mid <= 0:
            return

        self._prices.append((ctx.ts_ns, mid))

        # Чистим старые записи
        cutoff = ctx.ts_ns - self.lookback_ns
        while self._prices and self._prices[0][0] < cutoff:
            self._prices.pop(0)

        # Нужно минимум 10 точек для анализа
        if len(self._prices) < 10:
            return

        # Если позиция открыта — проверяем выход
        if ctx.position.is_open:
            # Выход по времени
            if (ctx.ts_ns - self._last_entry_ts_ns) >= self.hold_ns:
                engine.close_position(reason="HOLD_EXPIRED")
                return

            # Выход по обратному пробою
            recent_prices = [p for _, p in self._prices[-10:]]
            if ctx.position.side.value == "LONG" and mid <= min(recent_prices):
                engine.close_position(reason="REVERSE_BREAK")
                return
            if ctx.position.side.value == "SHORT" and mid >= max(recent_prices):
                engine.close_position(reason="REVERSE_BREAK")
                return

            return  # Не открываем вторую позицию

        # Проверяем пробой
        historical_prices = [p for _, p in self._prices[:-1]]  # кроме текущей
        if not historical_prices:
            return

        max_price = max(historical_prices)
        min_price = min(historical_prices)

        # Минимальный диапазон для пробоя (защита от шума)
        range_size = max_price - min_price
        if range_size <= 0:
            return

        # Пробой максимума → покупка
        if mid > max_price:
            engine.submit_market_order(
                symbol=ctx.symbol,
                side=OrderSide.BUY,
                quantity=0.001,  # фиксированный размер для проверки
                reason="BREAKOUT_UP",
            )
            self._last_entry_ts_ns = ctx.ts_ns
            return

        # Пробой минимума → продажа
        if mid < min_price:
            engine.submit_market_order(
                symbol=ctx.symbol,
                side=OrderSide.SELL,
                quantity=0.001,
                reason="BREAKOUT_DOWN",
            )
            self._last_entry_ts_ns = ctx.ts_ns
            return


# ============================================================
# Вспомогательные функции
# ============================================================

def find_available_dates(data_dir: str, symbol: str) -> List[str]:
    """Ищет доступные даты записи для символа."""
    base = Path(data_dir)
    if not base.exists():
        return []

    dates = set()

    # Проверяем оба типа данных: дельты и снапшоты
    for event_type in ("book_delta", "book_snapshot"):
        type_dir = base / event_type
        if not type_dir.exists():
            continue

        for date_dir in type_dir.iterdir():
            if not date_dir.is_dir():
                continue

            symbol_file = date_dir / f"{symbol.upper()}.jsonl"
            if symbol_file.exists():
                dates.add(date_dir.name)

    return sorted(dates)


def check_data_integrity(
    data_dir: str,
    symbol: str,
    date_str: str,
) -> Dict[str, any]:
    """Проверяет целостность данных для указанной даты."""
    base = Path(data_dir)
    result = {
        "has_deltas": False,
        "has_snapshots": False,
        "deltas_count": 0,
        "snapshots_count": 0,
        "deltas_size_mb": 0.0,
        "snapshots_size_mb": 0.0,
    }

    # Дельты
    deltas_file = base / "book_delta" / date_str / f"{symbol.upper()}.jsonl"
    if deltas_file.exists():
        result["has_deltas"] = True
        result["deltas_size_mb"] = deltas_file.stat().st_size / (1024 * 1024)
        # Считаем строки (быстро — только для небольших файлов)
        if result["deltas_size_mb"] < 100:
            with open(deltas_file, "r") as f:
                result["deltas_count"] = sum(1 for _ in f)

    # Снапшоты
    snapshots_file = base / "book_snapshot" / date_str / f"{symbol.upper()}.jsonl"
    if snapshots_file.exists():
        result["has_snapshots"] = True
        result["snapshots_size_mb"] = snapshots_file.stat().st_size / (1024 * 1024)
        if result["snapshots_size_mb"] < 100:
            with open(snapshots_file, "r") as f:
                result["snapshots_count"] = sum(1 for _ in f)

    return result


# ============================================================
# Основная функция запуска
# ============================================================

def run_replay(
    config_path: str,
    symbol: str,
    date_str: str,
    mode: str = "check",
    output_dir: Optional[str] = None,
) -> int:
    """
    Запускает прогон бэктеста.

    Возвращает код возврата (0 = успех, 1 = ошибка).
    """
    # Загружаем конфиг
    config = load_app_config(config_path)
    symbol = symbol.upper()

    print("=" * 60)
    print("REPLAY RUNNER")
    print("=" * 60)
    print(f"Символ: {symbol}")
    print(f"Дата: {date_str}")
    print(f"Режим: {mode}")
    print(f"Источник данных: {config.backtest.source_dir}")

    # Проверяем наличие данных
    integrity = check_data_integrity(config.backtest.source_dir, symbol, date_str)

    print(f"\nПроверка данных:")
    print(f"  Дельты: {'✅' if integrity['has_deltas'] else '❌'} "
          f"({integrity['deltas_count']} событий, "
          f"{integrity['deltas_size_mb']:.1f} MB)")
    print(f"  Снапшоты: {'✅' if integrity['has_snapshots'] else '❌'} "
          f"({integrity['snapshots_count']} снимков, "
          f"{integrity['snapshots_size_mb']:.1f} MB)")

    if not integrity["has_deltas"]:
        print(f"\n❌ Нет записанных дельт для {symbol} на {date_str}")
        print("Сначала запустите коллекционер для записи данных:")
        print(f"   python -m proscalper.app.collector {config_path}")
        return 1

    if not integrity["has_snapshots"]:
        print(f"\n⚠️ Нет снапшотов — бэктест может быть невалидным")
        print("Рекомендуется добавить keyframe_writer в коллекционер")

    # Создаём движок бэктеста
    engine_config = BacktestEngineConfig(
        source_dir=config.backtest.source_dir,
        use_parquet=config.backtest.use_parquet,
        tick_size=config.backtest.tick_size,
        depth_for_matching=config.backtest.depth_for_matching,
        latency_region=Region(config.backtest.latency_region)
            if hasattr(config.backtest, 'latency_region')
            else Region.TOKYO,
        initial_balance=config.backtest.initial_balance,
    )

    engine = BacktestEngine(config=engine_config)

    # Регистрируем стратегию в зависимости от режима
    if mode == "strategy":
        strategy = SimpleBreakoutStrategy(
            lookback_sec=60.0,
            hold_sec=30.0,
        )
        engine.register_strategy(strategy)
        print(f"\nСтратегия: SimpleBreakoutStrategy (lookback=60s, hold=30s)")
    else:
        print(f"\nРежим 'check': прогон без стратегии (проверка данных)")

    # Запускаем прогон
    print(f"\nЗапуск прогона...")
    start_time = time.time()

    result = engine.run(symbol=symbol, date_str=date_str)

    elapsed = time.time() - start_time
    print(f"Прогон завершён за {elapsed:.1f} сек")

    # Генерируем отчёт
    report = BacktestReport(
        result=result,
        config=ReportConfig(),
    )

    # Выводим сводку
    print()
    print(report.summary())

    # Сохраняем отчёты
    if output_dir is None:
        output_dir = f"./reports/{symbol}_{date_str}_{mode}"

    report.save_all(output_dir)
    print(f"\nОтчёты сохранены в: {output_dir}")
    print(f"  - summary.txt")
    print(f"  - detailed.txt")
    print(f"  - metrics.json")
    print(f"  - trades.jsonl")
    print(f"  - trades.csv")
    print(f"  - equity_curve.csv")

    return 0


# ============================================================
# Точка входа
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Runner: прогон бэктеста на записанных данных",
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="configs/default.yaml",
        help="Путь к конфигу (по умолчанию: configs/default.yaml)",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Символ инструмента (например, BTCUSDT)",
    )
    parser.add_argument(
        "--date",
        help="Дата записи в формате YYYY-MM-DD",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Прогнать все доступные даты",
    )
    parser.add_argument(
        "--mode",
        choices=["check", "strategy"],
        default="check",
        help="Режим: check (без стратегии) или strategy (с пробоем)",
    )
    parser.add_argument(
        "--output",
        help="Директория для отчётов",
    )

    args = parser.parse_args()

    # Определяем список дат
    if args.all_dates:
        config = load_app_config(args.config_path)
        dates = find_available_dates(config.backtest.source_dir, args.symbol)

        if not dates:
            print(f"❌ Нет записанных данных для {args.symbol}")
            sys.exit(1)

        print(f"Найдено дат для {args.symbol}: {len(dates)}")
        for date_str in dates:
            print(f"\n{'=' * 60}")
            print(f"Дата: {date_str}")
            print(f"{'=' * 60}")
            exit_code = run_replay(
                config_path=args.config_path,
                symbol=args.symbol,
                date_str=date_str,
                mode=args.mode,
                output_dir=args.output,
            )
            if exit_code != 0:
                sys.exit(exit_code)
    elif args.date:
        exit_code = run_replay(
            config_path=args.config_path,
            symbol=args.symbol,
            date_str=args.date,
            mode=args.mode,
            output_dir=args.output,
        )
        sys.exit(exit_code)
    else:
        parser.error("Укажите --date или --all-dates")


if __name__ == "__main__":
    main()