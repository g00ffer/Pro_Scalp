"""
Загрузчик секретов из переменных окружения или .env файла.

Использование:
    from proscalper.core.secrets import get_secret
    
    api_key = get_secret("BINANCE_FUTURES_API_KEY")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Попытка импортировать python-dotenv для загрузки .env
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


_loaded = False


def _ensure_dotenv_loaded() -> None:
    """Загружает .env файл, если ещё не загружен."""
    global _loaded
    if _loaded:
        return
    
    if not DOTENV_AVAILABLE:
        _loaded = True
        return
    
    # Ищем .env в корне проекта
    project_root = Path(__file__).parent.parent.parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
    
    _loaded = True


def get_secret(name: str, required: bool = True) -> Optional[str]:
    """
    Получает секрет из переменных окружения.
    
    Args:
        name: Имя переменной окружения
        required: Если True и переменная не найдена - бросает исключение
    
    Returns:
        Значение секрета или None
    
    Raises:
        ValueError: Если required=True и переменная не найдена
    """
    _ensure_dotenv_loaded()
    
    value = os.environ.get(name)
    
    if value is None and required:
        raise ValueError(
            f"Секрет '{name}' не найден. "
            f"Добавьте его в .env файл или переменные окружения."
        )
    
    return value


def get_binance_credentials() -> tuple[str, str]:
    """Возвращает API ключ и секрет для Binance Futures."""
    api_key = get_secret("BINANCE_FUTURES_API_KEY")
    api_secret = get_secret("BINANCE_FUTURES_API_SECRET")
    return api_key, api_secret


def get_bybit_credentials() -> tuple[str, str]:
    """Возвращает API ключ и секрет для Bybit."""
    api_key = get_secret("BYBIT_API_KEY")
    api_secret = get_secret("BYBIT_API_SECRET")
    return api_key, api_secret


def is_development() -> bool:
    """Проверяет, находимся ли мы в режиме разработки."""
    return get_secret("ENVIRONMENT", required=False) == "development"


def is_paper_trading() -> bool:
    """Проверяет, находимся ли мы в режиме paper trading."""
    return get_secret("ENVIRONMENT", required=False) == "paper"


def is_live_trading() -> bool:
    """Проверяет, находимся ли мы в live режиме."""
    return get_secret("ENVIRONMENT", required=False) == "live"
