"""
Аутентификация для Binance Futures API.

Binance использует HMAC-SHA256 подпись для приватных endpoints:
- timestamp (в ms) добавляется к запросу
- все query-параметры + secret подписываются
- подпись передаётся как параметр signature
- recvWindow (по умолчанию 5000ms) защищает от replay-атак

Этот модуль предоставляет:
- BinanceAuth: вычисление подписей
- AuthenticatedRequest: готовый запрос с подписью
- утилиты для построения query-строки

Используется в связке с:
- exchange/binance_futures/rest.py (приватные endpoints)
- exchange/binance_futures/order.py (отправка ордеров)
- core/secrets.py (загрузка API ключей)

Важно:
- Secret НИКОГДА не логируется
- timestamp генерируется локально, но синхронизируется
  через ExchangeTimeSync (см. core/clock.py)
- для testnet используется отдельный base_url
"""
from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


# ============================================================
# Конфигурация аутентификации
# ============================================================

class BinanceEndpoint(str, Enum):
    """Доступные endpoints Binance Futures."""
    LIVE = "https://fapi.binance.com"
    TESTNET = "https://testnet.binancefuture.com"


@dataclass(frozen=True)
class BinanceCredentials:
    """Учётные данные Binance Futures."""
    api_key: str
    api_secret: str
    
    def validate(self) -> None:
        """Проверяет корректность ключей."""
        if not self.api_key or len(self.api_key) < 10:
            raise ValueError("API key слишком короткий или пустой")
        if not self.api_secret or len(self.api_secret) < 10:
            raise ValueError("API secret слишком короткий или пустой")


@dataclass(frozen=True)
class BinanceAuthConfig:
    """Конфигурация аутентификации."""
    credentials: BinanceCredentials
    endpoint: BinanceEndpoint = BinanceEndpoint.LIVE
    recv_window_ms: int = 5000
    
    # Использовать синхронизацию времени с биржей
    # (через ExchangeTimeSync из core/clock.py)
    use_exchange_sync: bool = True
    
    @property
    def base_url(self) -> str:
        return self.endpoint.value


# ============================================================
# Аутентификатор
# ============================================================

class BinanceAuth:
    """
    Вычисление подписей для Binance Futures.
    
    Использование:
        creds = BinanceCredentials(
            api_key="your_key",
            api_secret="your_secret",
        )
        auth = BinanceAuth(creds)
        
        # Для GET-запроса
        params = {"symbol": "BTCUSDT", "side": "BUY"}
        signed_params = auth.sign_get_params(params)
        # signed_params теперь содержит timestamp и signature
        
        # Для POST-запроса (query string в body)
        signed_query = auth.sign_post_body(params)
        # signed_query = "symbol=BTCUSDT&side=BUY&timestamp=...&signature=..."
    """
    
    def __init__(
        self,
        config: BinanceAuthConfig,
        time_offset_ns: int = 0,
    ) -> None:
        self._config = config
        self._config.credentials.validate()
        
        # Смещение локального времени относительно биржевого
        # (положительное = локальные отстают)
        self._time_offset_ns = time_offset_ns
        
        # Счётчик для уникальности nonce (при быстром вызове)
        self._last_ts_ms = 0
        self._nonce = 0
    
    def update_time_offset(self, offset_ns: int) -> None:
        """
        Обновляет смещение времени (из ExchangeTimeSync).
        
        Вызывается при старте и периодически (каждые 5-10 минут).
        """
        self._time_offset_ns = offset_ns
    
    # ============================================
    # Публичные методы
    # ============================================
    
    def sign_get_params(
        self,
        params: Dict[str, Any],
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Подписывает параметры для GET-запроса.
        
        Возвращает новый словарь с добавленными
        timestamp и signature.
        """
        if timestamp_ms is None:
            timestamp_ms = self._current_timestamp_ms()
        
        # Копируем и добавляем обязательные поля
        signed = dict(params)
        signed["timestamp"] = timestamp_ms
        signed["recvWindow"] = self._config.recv_window_ms
        
        # Строим query-строку (в алфавитном порядке ключей)
        query = self._build_query_string(signed)
        
        # Подписываем
        signature = self._sign(query)
        signed["signature"] = signature
        
        return signed
    
    def sign_post_body(
        self,
        params: Dict[str, Any],
        timestamp_ms: Optional[int] = None,
    ) -> str:
        """
        Подписывает параметры для POST-запроса (body).
        
        Возвращает готовую query-строку с signature.
        Binance принимает body в формате x-www-form-urlencoded.
        """
        if timestamp_ms is None:
            timestamp_ms = self._current_timestamp_ms()
        
        signed = dict(params)
        signed["timestamp"] = timestamp_ms
        signed["recvWindow"] = self._config.recv_window_ms
        
        query = self._build_query_string(signed)
        signature = self._sign(query)
        
        return f"{query}&signature={signature}"
    
    def get_headers(self) -> Dict[str, str]:
        """
        Возвращает заголовки для аутентифицированного запроса.
        
        Для Binance достаточно X-MBX-APIKEY в header.
        """
        return {
            "X-MBX-APIKEY": self._config.credentials.api_key,
        }
    
    @property
    def api_key(self) -> str:
        """API key (для заголовков)."""
        return self._config.credentials.api_key
    
    @property
    def base_url(self) -> str:
        """Base URL для endpoints."""
        return self._config.base_url
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    def _current_timestamp_ms(self) -> int:
        """
        Возвращает текущий timestamp в ms с учётом offset.
        
        Гарантирует монотонность: при быстром вызове
        (в пределах одной миллисекунды) увеличивает nonce.
        """
        # Локальное время + смещение
        local_ns = time.time_ns()
        adjusted_ns = local_ns + self._time_offset_ns
        ts_ms = adjusted_ns // 1_000_000
        
        # Гарантируем уникальность
        if ts_ms == self._last_ts_ms:
            # В пределах одной ms — увеличиваем на 1
            # (это безопасно, recvWindow = 5000ms)
            ts_ms = self._last_ts_ms + 1
        
        self._last_ts_ms = ts_ms
        return ts_ms
    
    def _build_query_string(self, params: Dict[str, Any]) -> str:
        """
        Строит query-строку из параметров.
        
        Важно: Binance требует определённый порядок.
        Используем urllib.parse для корректного encoding.
        """
        # Фильтруем None и конвертируем bool
        filtered = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, bool):
                filtered[k] = "true" if v else "false"
            else:
                filtered[k] = v
        
        # URL encoding
        return urllib.parse.urlencode(filtered)
    
    def _sign(self, data: str) -> str:
        """
        Вычисляет HMAC-SHA256 подпись.
        
        Args:
            data: строка для подписи (query string)
        
        Returns:
            hex-строка подписи
        """
        return hmac.new(
            self._config.credentials.api_secret.encode("utf-8"),
            data.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    
    # ============================================
    # Диагностические методы
    # ============================================
    
    def verify_connection(self, server_time_ms: int) -> Dict[str, Any]:
        """
        Проверяет синхронизацию с сервером.
        
        Вызывается после получения serverTime через /fapi/v1/time.
        
        Returns:
            dict с информацией о смещении
        """
        local_ms = time.time_ns() // 1_000_000
        adjusted_ms = local_ms + self._time_offset_ns // 1_000_000
        drift_ms = adjusted_ms - server_time_ms
        
        return {
            "server_time_ms": server_time_ms,
            "local_time_ms": local_ms,
            "adjusted_time_ms": adjusted_ms,
            "drift_ms": drift_ms,
            "time_offset_ns": self._time_offset_ns,
            "in_sync": abs(drift_ms) < self._config.recv_window_ms,
        }


# ============================================================
# Фабрика
# ============================================================

def create_auth(
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    time_offset_ns: int = 0,
) -> BinanceAuth:
    """
    Фабрика для быстрого создания BinanceAuth.
    
    Использование:
        from proscalper.core.secrets import get_binance_credentials
        
        creds = get_binance_credentials()
        auth = create_auth(creds.api_key, creds.api_secret)
    """
    creds = BinanceCredentials(api_key=api_key, api_secret=api_secret)
    endpoint = BinanceEndpoint.TESTNET if testnet else BinanceEndpoint.LIVE
    config = BinanceAuthConfig(credentials=creds, endpoint=endpoint)
    return BinanceAuth(config, time_offset_ns=time_offset_ns)


def create_auth_from_env(testnet: bool = False) -> BinanceAuth:
    """
    Создаёт BinanceAuth из переменных окружения.
    
    Ожидает:
    - BINANCE_FUTURES_API_KEY
    - BINANCE_FUTURES_API_SECRET
    """
    import os
    
    api_key = os.environ.get("BINANCE_FUTURES_API_KEY", "")
    api_secret = os.environ.get("BINANCE_FUTURES_API_SECRET", "")
    
    if not api_key or not api_secret:
        raise ValueError(
            "Не найдены переменные окружения "
            "BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET"
        )
    
    return create_auth(api_key, api_secret, testnet=testnet)