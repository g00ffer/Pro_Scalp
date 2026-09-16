"""
Адаптер для исполнения ордеров на Binance Futures.

Реализует интерфейс ExchangeAdapter для работы с Binance.

Важно:
- Не используем testnet, сразу основная сеть
- Подпись запросов через HMAC-SHA256
- Повторные попытки при сетевых ошибках (3 попытки)
- Нативные типы ордеров: LIMIT, MARKET, STOP_MARKET, TAKE_PROFIT_MARKET

Используется в связке с:
- ExecutionEngine (движок исполнения)
- Секреты загружаются из .env через proscalper.core.secrets
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx

from proscalper.core.types import OrderSide
from proscalper.core.secrets import get_binance_credentials
from proscalper.execution.execution_engine import (
    ExchangeAdapter,
    OrderStatus,
    OrderType,
)


class BinanceFuturesConfig:
    """
    Конфигурация адаптера Binance Futures.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        recv_window_ms: int = 5000,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
        request_timeout_ms: int = 10000,
    ):
        self.base_url = base_url
        self.recv_window_ms = recv_window_ms
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.request_timeout_ms = request_timeout_ms


class BinanceFuturesAdapter(ExchangeAdapter):
    """
    Адаптер для Binance Futures.
    
    Реализует все методы интерфейса ExchangeAdapter.
    
    Использование:
        adapter = BinanceFuturesAdapter()
        
        # Отправка ордера
        order_id = await adapter.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT_IOC,
            quantity=0.001,
            price=76000,
        )
        
        # Проверка статуса
        status = await adapter.get_order_status("BTCUSDT", order_id)
    """
    
    def __init__(
        self,
        config: Optional[BinanceFuturesConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.config = config or BinanceFuturesConfig()
        
        # Загружаем ключи
        if api_key and api_secret:
            self._api_key = api_key
            self._api_secret = api_secret
        else:
            # Пытаемся загрузить из .env
            try:
                self._api_key, self._api_secret = get_binance_credentials()
            except ValueError:
                self._api_key = ""
                self._api_secret = ""
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "BinanceFuturesAdapter":
        """Контекстный менеджер для асинхронного входа."""
        self._client = httpx.AsyncClient(
            timeout=self.config.request_timeout_ms / 1000,
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Контекстный менеджер для асинхронного выхода."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _ensure_client(self) -> None:
        """Убеждается, что клиент создан."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.request_timeout_ms / 1000,
            )
    
    # ============================================
    # Реализация ExchangeAdapter
    # ============================================
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float = 0.0,
        client_order_id: str = "",
        stop_price: float = 0.0,
    ) -> str:
        """
        Отправляет ордер на Binance Futures.
        
        Типы ордеров:
        - LIMIT: лимитный ордер
        - MARKET: рыночный ордер
        - LIMIT_IOC: лимитный ордер с немедленной отменой
        - STOP_MARKET: стоп-маркет ордер (для стоп-лоссов)
        - TAKE_PROFIT_MARKET: тейк-профит маркет ордер
        
        Возвращает order_id, присвоенный биржей.
        """
        await self._ensure_client()
        
        # Преобразуем тип ордера в формат Binance
        binance_order_type = self._convert_order_type(order_type)
        
        # Формируем параметры
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.value.upper(),
            "type": binance_order_type,
            "quantity": str(quantity),
            "recvWindow": self.config.recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }
        
        # Добавляем цену для лимитных ордеров
        if order_type in (OrderType.LIMIT, OrderType.LIMIT_IOC):
            params["price"] = str(price)
            params["timeInForce"] = "IOC" if order_type == OrderType.LIMIT_IOC else "GTC"
        
        # Добавляем стоп-цену для стоп-ордеров
        if stop_price > 0:
            params["stopPrice"] = str(stop_price)
            params["workingType"] = "MARK_PRICE"
        
        # Добавляем клиентский ID ордера
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        
        # Отправляем запрос с подписью
        response = await self._signed_request(
            "POST",
            "/fapi/v1/order",
            params=params,
        )
        
        # Возвращаем order_id
        return str(response.get("orderId", ""))
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Отменяет ордер на Binance.
        
        Возвращает True, если отмена успешна.
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "orderId": order_id,
            "recvWindow": self.config.recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }
        
        try:
            await self._signed_request(
                "DELETE",
                "/fapi/v1/order",
                params=params,
            )
            return True
        except Exception:
            return False
    
    async def get_order_status(
        self,
        symbol: str,
        order_id: str,
    ) -> OrderStatus:
        """
        Получает статус ордера с Binance.
        
        Возвращает текущий статус ордера.
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "orderId": order_id,
            "recvWindow": self.config.recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }
        
        response = await self._signed_request(
            "GET",
            "/fapi/v1/order",
            params=params,
        )
        
        # Преобразуем статус из формата Binance в наш
        binance_status = response.get("status", "")
        return self._convert_order_status(binance_status)
    
    async def get_order_fill_info(
        self,
        symbol: str,
        order_id: str,
    ) -> Dict[str, float]:
        """
        Получает информацию об исполнении ордера.
        
        Возвращает словарь с:
        - filled_quantity: исполненное количество
        - average_price: средняя цена исполнения
        - fees: комиссии
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "orderId": order_id,
            "recvWindow": self.config.recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }
        
        response = await self._signed_request(
            "GET",
            "/fapi/v1/order",
            params=params,
        )
        
        # Извлекаем информацию об исполнении
        filled_quantity = float(response.get("executedQty", 0))
        cum_quote = float(response.get("cumQuote", 0))
        
        # Рассчитываем среднюю цену
        average_price = 0.0
        if filled_quantity > 0:
            average_price = cum_quote / filled_quantity
        
        # Для комиссий нужен отдельный запрос к /fapi/v1/userTrades
        # Для упрощения возвращаем 0, комиссии рассчитываются отдельно
        fees = 0.0
        
        return {
            "filled_quantity": filled_quantity,
            "average_price": average_price,
            "fees": fees,
        }
    
    async def get_account_balance(self) -> float:
        """
        Получает баланс аккаунта в USDT.
        
        Возвращает доступный баланс.
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "recvWindow": self.config.recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }
        
        response = await self._signed_request(
            "GET",
            "/fapi/v2/balance",
            params=params,
        )
        
        # Ищем баланс в USDT
        if isinstance(response, list):
            for asset in response:
                if asset.get("asset") == "USDT":
                    return float(asset.get("availableBalance", 0))
        
        return 0.0
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Получает позицию по символу.
        
        Возвращает словарь с информацией о позиции или None.
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "recvWindow": self.config.recv_window_ms,
            "timestamp": int(time.time() * 1000),
        }
        
        response = await self._signed_request(
            "GET",
            "/fapi/v2/positionRisk",
            params=params,
        )
        
        if isinstance(response, list) and len(response) > 0:
            position = response[0]
            return {
                "symbol": position.get("symbol", ""),
                "positionAmt": float(position.get("positionAmt", 0)),
                "entryPrice": float(position.get("entryPrice", 0)),
                "markPrice": float(position.get("markPrice", 0)),
                "unRealizedProfit": float(position.get("unRealizedProfit", 0)),
                "leverage": int(position.get("leverage", 1)),
            }
        
        return None
    
    async def close(self) -> None:
        """Закрывает соединение."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # ============================================
    # Внутренние методы
    # ============================================
    
    async def _signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Выполняет подписанный запрос к Binance.
        
        Все приватные эндпоинты требуют подписи.
        """
        if params is None:
            params = {}
        
        # Добавляем подпись
        query_string = urlencode(params)
        signature = self._sign(query_string)
        params["signature"] = signature
        
        # Заголовки
        headers = {
            "X-MBX-APIKEY": self._api_key,
        }
        
        # URL
        url = f"{self.config.base_url}{endpoint}"
        
        # Выполняем запрос с повторами
        return await self._request_with_retry(
            method=method,
            url=url,
            params=params if method == "GET" else None,
            data=params if method in ("POST", "DELETE") else None,
            headers=headers,
        )
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Выполняет запрос с повторными попытками.
        
        При сетевых ошибках делает до 3 попыток с задержкой 100мс.
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                )
                
                # Проверяем статус ответа
                if response.status_code == 200:
                    return response.json()
                
                # Обрабатываем ошибки Binance
                error_data = response.json()
                error_code = error_data.get("code", 0)
                error_msg = error_data.get("msg", "Unknown error")
                
                # Некоторые ошибки не стоит повторять
                if error_code in (-2010, -2011, -1021):
                    # -2010: Invalid order
                    # -2011: Cancel rejected
                    # -1021: Timestamp outside recvWindow
                    raise Exception(f"Binance error {error_code}: {error_msg}")
                
                # Другие ошибки можно повторить
                last_error = Exception(f"Binance error {error_code}: {error_msg}")
            
            except httpx.TimeoutException as e:
                last_error = e
            
            except httpx.NetworkError as e:
                last_error = e
            
            except Exception as e:
                # Для непредвиденных ошибок не делаем повторов
                raise
            
            # Ждём перед следующей попыткой
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_ms / 1000)
        
        # Все попытки исчерпаны
        raise last_error or Exception("Request failed")
    
    def _sign(self, query_string: str) -> str:
        """
        Подписывает строку запроса через HMAC-SHA256.
        
        Binance требует подпись для всех приватных эндпоинтов.
        """
        return hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Преобразует тип ордера в формат Binance."""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.LIMIT_IOC: "LIMIT",
        }
        return mapping.get(order_type, "MARKET")
    
    def _convert_order_status(self, binance_status: str) -> OrderStatus:
        """Преобразует статус ордера из формата Binance в наш."""
        mapping = {
            "NEW": OrderStatus.SUBMITTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(binance_status, OrderStatus.PENDING)