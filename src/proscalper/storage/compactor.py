"""
Компактор JSONL → Parquet.

После записи рыночных данных в JSONL (delta_writer.py) файлы
быстро растут и занимают много места. Компактор:
- читает JSONL по частям
- сохраняет в Parquet с ZSTD-сжатием
- даёт до 10x экономии на диске
- ускоряет аналитику (Polars/DuckDB/Arrow)

Компактор работает offline:
- по расписанию (cron / APScheduler)
- после завершения торговой сессии
- вручную из CLI

Не требует pandas: использует pyarrow напрямую для контроля памяти.
На больших файлах обрабатывает батчами (streaming).

Формат вывода:
    data/<event_type>/<YYYY-MM-DD>/<SYMBOL>.jsonl
    → data_parquet/<event_type>/<YYYY-MM-DD>/<SYMBOL>.parquet

Опции:
- keep_jsonl: сохранять ли исходный JSONL (по умолчанию False)
- partition_by_day: разделять ли по дням (по умолчанию True)
- compression: "zstd" | "snappy" | "gzip" | "none"
- batch_size: строк за один batch при чтении JSONL
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Импорт pyarrow должен быть явным, но опциональным,
# чтобы модуль не падал при импорте, если библиотека не установлена.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Для compactor нужен pyarrow. Установи: pip install pyarrow"
    ) from exc


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class CompactorConfig:
    """Конфигурация компактора."""
    source_dir: str = "./data/binance_futures"
    target_dir: str = "./data_parquet/binance_futures"
    batch_size: int = 50_000
    compression: str = "zstd"
    keep_jsonl: bool = False
    overwrite: bool = False


# ============================================================
# Результат
# ============================================================

@dataclass
class CompactionResult:
    """Результат компакции одного файла."""
    source_path: Path
    target_path: Optional[Path]
    rows: int = 0
    source_bytes: int = 0
    target_bytes: int = 0
    success: bool = True
    error: str = ""
    skipped: bool = False

    @property
    def compression_ratio(self) -> float:
        if self.target_bytes <= 0:
            return 0.0
        return self.source_bytes / self.target_bytes


# ============================================================
# Основной класс
# ============================================================

class Compactor:
    """
    Компактор JSONL → Parquet.

    Использование:
        config = CompactorConfig(source_dir="./data/binance_futures")
        compactor = Compactor(config)

        # Компакция всех файлов
        results = compactor.compact_all()

        # Компакция конкретного файла
        result = compactor.compact_file(path)

        # Или из CLI:
        python -m proscalper.storage.compactor --source ./data --target ./data_parquet
    """

    def __init__(self, config: Optional[CompactorConfig] = None) -> None:
        self.config = config or CompactorConfig()
        self._source_root = Path(self.config.source_dir)
        self._target_root = Path(self.config.target_dir)

    # ============================================
    # Public API
    # ============================================

    def compact_all(self) -> List[CompactionResult]:
        """
        Компактирует все .jsonl-файлы в source_dir (рекурсивно).
        """
        if not self._source_root.exists():
            return []

        results: List[CompactionResult] = []

        for jsonl_path in sorted(self._source_root.rglob("*.jsonl")):
            result = self.compact_file(jsonl_path)
            results.append(result)

        return results

    def compact_file(self, jsonl_path: Path) -> CompactionResult:
        """
        Компактирует один JSONL файл в Parquet.

        Структура source:  <source>/<event_type>/<date>/<SYMBOL>.jsonl
        Структура target:  <target>/<event_type>/<date>/<SYMBOL>.parquet
        """
        jsonl_path = Path(jsonl_path)
        result = CompactionResult(source_path=jsonl_path)

        if not jsonl_path.exists():
            result.success = False
            result.error = "source not found"
            return result

        try:
            relative = jsonl_path.relative_to(self._source_root)
        except ValueError:
            # Файл не под source_dir — используем только имя
            relative = Path(jsonl_path.name)

        target_path = self._target_root / relative.with_suffix(".parquet")
        result.target_path = target_path

        # Skip, если Parquet уже свежее
        if (
            target_path.exists()
            and not self.config.overwrite
            and target_path.stat().st_mtime >= jsonl_path.stat().st_mtime
        ):
            result.skipped = True
            return result

        try:
            result.source_bytes = jsonl_path.stat().st_size

            target_path.parent.mkdir(parents=True, exist_ok=True)

            rows_written = self._stream_jsonl_to_parquet(
                jsonl_path=jsonl_path,
                target_path=target_path,
            )

            result.rows = rows_written
            result.target_bytes = target_path.stat().st_size if target_path.exists() else 0

            if not self.config.keep_jsonl:
                jsonl_path.unlink()

        except Exception as exc:
            result.success = False
            result.error = str(exc)

        return result

    # ============================================
    # Streaming-обработка
    # ============================================

    def _stream_jsonl_to_parquet(
        self,
        jsonl_path: Path,
        target_path: Path,
    ) -> int:
        """
        Читает JSONL по батчам, конвертирует в Parquet потоково.

        Определяет схему по первой строке, дальше пишет через
        ParquetWriter (append), чтобы не держать всё в памяти.
        """
        writer: Optional[pq.ParquetWriter] = None
        schema: Optional[pa.Schema] = None
        total_rows = 0

        try:
            for batch_dicts in self._iter_jsonl_batches(jsonl_path):
                if not batch_dicts:
                    continue

                # Определяем схему по первой строке (лениво)
                if schema is None:
                    schema = self._infer_schema(batch_dicts[0])

                table = self._dicts_to_table(batch_dicts, schema)

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(target_path),
                        schema,
                        compression=self.config.compression,
                    )

                writer.write_table(table)
                total_rows += len(batch_dicts)

        finally:
            if writer is not None:
                writer.close()

        return total_rows

    def _iter_jsonl_batches(
        self,
        path: Path,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Итерирует JSONL батчами заданного размера."""
        batch: List[Dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        batch.append(obj)
                except json.JSONDecodeError:
                    # Битые строки пропускаем, но не падаем
                    continue

                if len(batch) >= self.config.batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

    # ============================================
    # Схема и конвертация
    # ============================================

    def _infer_schema(self, sample: Dict[str, Any]) -> pa.Schema:
        """
        Выводит схему Arrow по образцу.

        - int → int64
        - float → float64
        - bool → bool
        - str → string (large_string для больших текстов)
        - list → list<string> / list<double> (упрощённо)
        - dict → struct с вложенными полями
        - None → null (позже может быть переопределена при появлении данных)
        """
        fields: List[pa.Field] = []
        for key, value in sample.items():
            fields.append(pa.field(key, self._arrow_type(value)))
        return pa.schema(fields)

    def _arrow_type(self, value: Any) -> pa.DataType:
        """Определяет Arrow-тип по значению (упрощённо)."""
        if value is None:
            return pa.null()
        if isinstance(value, bool):
            return pa.bool_()
        if isinstance(value, int):
            return pa.int64()
        if isinstance(value, float):
            return pa.float64()
        if isinstance(value, str):
            return pa.large_string()
        if isinstance(value, list):
            # Пытаемся вывести тип элемента по первому
            if not value:
                return pa.list_(pa.null())
            return pa.list_(self._arrow_type(value[0]))
        if isinstance(value, dict):
            return pa.struct(
                [pa.field(k, self._arrow_type(v)) for k, v in value.items()]
            )
        # Fallback
        return pa.large_string()

    def _dicts_to_table(
        self,
        rows: List[Dict[str, Any]],
        schema: pa.Schema,
    ) -> pa.Table:
        """
        Конвертирует список dict в Arrow Table строго по схеме.

        Пропускает поля, которых нет в схеме; недостающие поля
        заполняет None.
        """
        columns: Dict[str, List[Any]] = {f.name: [] for f in schema}

        for row in rows:
            for f in schema:
                columns[f.name].append(row.get(f.name))

        return pa.table(columns, schema=schema)


# ============================================================
# CLI
# ============================================================

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Компактор JSONL → Parquet",
    )
    parser.add_argument(
        "--source",
        default="./data/binance_futures",
        help="Директория с исходными JSONL",
    )
    parser.add_argument(
        "--target",
        default="./data_parquet/binance_futures",
        help="Директория для Parquet",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "none"],
    )
    parser.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Не удалять исходные JSONL после компакции",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать существующие Parquet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )

    args = parser.parse_args()

    config = CompactorConfig(
        source_dir=args.source,
        target_dir=args.target,
        compression=args.compression,
        keep_jsonl=args.keep_jsonl,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )

    compactor = Compactor(config)

    print(f"📦 Компакция: {config.source_dir} → {config.target_dir}")
    start = time.time()
    results = compactor.compact_all()
    elapsed = time.time() - start

    ok = [r for r in results if r.success and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if not r.success]

    total_src = sum(r.source_bytes for r in ok)
    total_dst = sum(r.target_bytes for r in ok)
    total_rows = sum(r.rows for r in ok)

    print()
    print(f"✅ Скомпактировано: {len(ok)} файлов, {total_rows} строк")
    print(f"⏭️  Пропущено (свежие Parquet): {len(skipped)}")
    print(f"❌ Ошибок: {len(failed)}")
    print(f"📊 Сжатие: {total_src / 1e6:.1f} MB → {total_dst / 1e6:.1f} MB "
          f"(x{total_src / max(total_dst, 1):.2f})")
    print(f"⏱️  Время: {elapsed:.1f} сек")

    if failed:
        print()
        print("Ошибки:")
        for r in failed[:10]:
            print(f"  {r.source_path}: {r.error}")


if __name__ == "__main__":
    _main()