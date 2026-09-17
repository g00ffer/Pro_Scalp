"""
Структурированное логирование системы.

Заменяет разрозненные print() вызовы на единый логгер с:
- уровнями (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- контекстом (символ, модуль, идентификаторы)
- единым форматом для консоли и файлов
- цветным выводом для консоли
- совместимостью со стандартным модулем logging

Использование:
    from proscalper.core.logging import get_logger, LogContext
    
    logger = get_logger(__name__)
    logger.info("WS connected", symbol="BTCUSDT", details={"streams": 2})
    
    # Или с контекстом
    with LogContext(logger, symbol="BTCUSDT", module="orderbook_fast"):
        logger.warning("Sequence gap detected", gap=5)
    
    # Настройка при старте приложения
    from proscalper.core.logging import setup_logging
    setup_logging(level="INFO", format="text", log_file="logs/app.log")

Принципы:
- логгер не блокирует (пишет через стандартный logging)
- формат единый для всех модулей
- можно переключить на JSON для интеграции с ELK/Loki
- совместим с journal/decision_journal.py (разные вещи:
  логгер = для людей/отладки, журнал = для аудита/аналитики)
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# Уровни логирования
# ============================================================

class LogLevel(str, Enum):
    """Уровень логирования."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


# ============================================================
# Конфигурация
# ============================================================

@dataclass(frozen=True)
class LoggingConfig:
    """Конфигурация логирования."""
    # Уровень по умолчанию
    level: str = "INFO"
    
    # Формат: "text" (человекочитаемый) или "json" (для ELK/Loki)
    format: str = "text"
    
    # Файл для записи (None = только консоль)
    log_file: Optional[str] = None
    
    # Цветной вывод для консоли
    use_colors: bool = True
    
    # Выводить модуль в лог
    show_module: bool = True
    
    # Выводить время в лог
    show_timestamp: bool = True
    
    # Максимальный размер файла логов (для ротации)
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


# ============================================================
# Цвета для консоли
# ============================================================

class _Colors:
    """ANSI коды цветов для консоли."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Цвета по уровням
    DEBUG = "\033[36m"     # cyan
    INFO = "\033[32m"      # green
    WARNING = "\033[33m"   # yellow
    ERROR = "\033[31m"     # red
    CRITICAL = "\033[35m"  # magenta
    
    # Цвета для компонентов
    TIMESTAMP = "\033[90m"   # gray
    MODULE = "\033[94m"      # blue
    SYMBOL = "\033[36m"      # cyan


_LEVEL_COLORS = {
    "DEBUG": _Colors.DEBUG,
    "INFO": _Colors.INFO,
    "WARNING": _Colors.WARNING,
    "ERROR": _Colors.ERROR,
    "CRITICAL": _Colors.CRITICAL,
}


# ============================================================
# Форматтеры
# ============================================================

class TextFormatter(logging.Formatter):
    """
    Человекочитаемый форматтер.
    
    Формат:
        [12:34:56.789] [INFO] [module] message (key=value, key2=value2)
    """
    
    def __init__(
        self,
        use_colors: bool = True,
        show_module: bool = True,
        show_timestamp: bool = True,
    ) -> None:
        super().__init__()
        self.use_colors = use_colors
        self.show_module = show_module
        self.show_timestamp = show_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        # Время
        ts = ""
        if self.show_timestamp:
            local_time = time.localtime(record.created)
            ms = int(record.msecs)
            ts = f"[{time.strftime('%H:%M:%S', local_time)}.{ms:03d}]"
            if self.use_colors:
                ts = f"{_Colors.TIMESTAMP}{ts}{_Colors.RESET}"
        
        # Уровень
        level = record.levelname
        if self.use_colors:
            color = _LEVEL_COLORS.get(level, "")
            level = f"{color}{level}{_Colors.RESET}"
        level = f"[{level:>8}]"
        
        # Модуль
        module = ""
        if self.show_module:
            mod_name = record.name.split(".")[-1] if "." in record.name else record.name
            if self.use_colors:
                module = f"{_Colors.MODULE}[{mod_name}]{_Colors.RESET}"
            else:
                module = f"[{mod_name}]"
        
        # Сообщение
        message = record.getMessage()
        
        # Дополнительные поля из контекста
        extra = ""
        if hasattr(record, "_context") and record._context:
            ctx_str = ", ".join(
                f"{k}={v}" for k, v in record._context.items()
            )
            extra = f" ({ctx_str})"
        
        # Собираем строку
        parts = [p for p in [ts, level, module] if p]
        prefix = " ".join(parts)
        
        return f"{prefix} {message}{extra}"


class JsonFormatter(logging.Formatter):
    """
    JSON форматтер для интеграции с ELK/Loki.
    
    Каждая запись — одна строка JSON:
        {"ts": "...", "level": "INFO", "module": "...", "message": "...", ...}
    """
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        entry = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.gmtime(record.created),
            ) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        
        # Добавляем контекст
        if hasattr(record, "_context") and record._context:
            entry.update(record._context)
        
        # Добавляем информацию об ошибке
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(entry, ensure_ascii=False)


# ============================================================
# Контекст логирования
# ============================================================

class LogContext:
    """
    Контекстный менеджер для добавления структурированных полей.
    
    Использование:
        with LogContext(logger, symbol="BTCUSDT", module="orderbook"):
            logger.info("Book updated", depth=50)
    """
    
    def __init__(
        self,
        logger: StructuredLogger,
        **context: Any,
    ) -> None:
        self._logger = logger
        self._context = context
        self._old_context: Dict[str, Any] = {}
    
    def __enter__(self) -> "LogContext":
        self._old_context = self._logger._default_context.copy()
        self._logger._default_context.update(self._context)
        return self
    
    def __exit__(self, *args) -> None:
        self._logger._default_context = self._old_context


# ============================================================
# Структурированный логгер
# ============================================================

class StructuredLogger:
    """
    Обёртка над стандартным логгером, поддерживающая
    структурированный контекст через **kwargs.
    
    Использование:
        logger = get_logger(__name__)
        logger.info("Message", symbol="BTCUSDT", value=42)
        # В логе: ... Message (symbol=BTCUSDT, value=42)
    """
    
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        self._default_context: Dict[str, Any] = {}
    
    def _log(
        self,
        level: int,
        msg: str,
        args: tuple,
        context: Dict[str, Any],
        exc_info=None,
    ) -> None:
        # Объединяем контекст по умолчанию с переданным
        full_context = {**self._default_context, **context}
        
        # Передаём контекст через extra для форматтера
        extra = {"_context": full_context}
        
        self._logger.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            extra=extra,
        )
    
    def debug(self, msg: str, *args, **context) -> None:
        self._log(logging.DEBUG, msg, args, context)
    
    def info(self, msg: str, *args, **context) -> None:
        self._log(logging.INFO, msg, args, context)
    
    def warning(self, msg: str, *args, **context) -> None:
        self._log(logging.WARNING, msg, args, context)
    
    def error(self, msg: str, *args, **context) -> None:
        self._log(logging.ERROR, msg, args, context)
    
    def critical(self, msg: str, *args, **context) -> None:
        self._log(logging.CRITICAL, msg, args, context)
    
    def exception(self, msg: str, *args, **context) -> None:
        context["exc_info"] = True
        self._log(logging.ERROR, msg, args, context, exc_info=True)

# ============================================================
# Фабрика логгеров
# ============================================================

_initialized = False


def get_logger(name: str) -> StructuredLogger:
    """
    Возвращает структурированный логгер для модуля.
    
    Использование:
        logger = get_logger(__name__)
        logger.info("Message", symbol="BTCUSDT")
    """
    return StructuredLogger(name)


def _setup_handler(
    logger: logging.Logger,
    config: LoggingConfig,
    handler: logging.Handler,
) -> None:
    """Настраивает обработчик с форматтером."""
    if config.format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter(
            use_colors=config.use_colors,
            show_module=config.show_module,
            show_timestamp=config.show_timestamp,
        ))
    
    handler.setLevel(_LEVEL_MAP.get(LogLevel(config.level), logging.INFO))
    logger.addHandler(handler)


# ============================================================
# Инициализация
# ============================================================

def setup_logging(
    level: str = "INFO",
    format: str = "text",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    show_module: bool = True,
    show_timestamp: bool = True,
) -> None:
    """
    Инициализирует логирование для всего приложения.
    
    Вызывается один раз при старте (в main).
    
    Использование:
        setup_logging(level="INFO", format="text")
        # или
        setup_logging(level="DEBUG", format="json", log_file="logs/app.log")
    """
    global _initialized
    if _initialized:
        return
    _initialized = True
    
    config = LoggingConfig(
        level=level,
        format=format,
        log_file=log_file,
        use_colors=use_colors,
        show_module=show_module,
        show_timestamp=show_timestamp,
    )
    
    # Корневой логгер
    root = logging.getLogger()
    root.setLevel(_LEVEL_MAP.get(LogLevel(level), logging.INFO))
    
    # Очищаем существующие обработчики
    root.handlers.clear()
    
    # Консольный обработчик
    console = logging.StreamHandler(sys.stdout)
    _setup_handler(root, config, console)
    
    # Файловый обработчик (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        _setup_handler(root, config, file_handler)
    
    # Подавляем слишком болтливые библиотеки
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)