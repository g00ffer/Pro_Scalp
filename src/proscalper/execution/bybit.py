"""
Адаптер для исполнения ордеров на Bybit (V5 API).

Реализует интерфейс ExchangeAdapter для работы с Bybit.

Важно:
- Не используем testnet, сразу основная сеть
- Подпись запросов через HMAC-SHA256 (формат Bybit)
- Повторные попытки при сетевых ошибках (3 попытки)
- Категория: linear (линейные фьючерсы)

Отличия от Binance:
- Другой формат подписи: timestamp + api_key + recv_window + query
- Другой формат заголовков: X-BAPI-*
- Другой формат тела запроса (JSON)

Используется в связке с:
- ExecutionEngine (движок исполнения)
- Секреты загружаются из .env через proscalper.core.secrets
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx

from proscalper.core.types import OrderSide
from proscalper.core.secrets import get_bybit_credentials
from proscalper.execution.execution_engine import (
    ExchangeAdapter,
    OrderStatus,
    OrderType,
)


class BybitConfig:
    """
    Конфигурация адаптера Bybit.
    
    Все параметры подобраны как разумные значения по умолчанию.
    """
    
    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        category: str = "linear",
        recv_window_ms: int = 5000,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
        request_timeout_ms: int = 10000,
    ):
        self.base_url = base_url
        self.category = category
        self.recv_window_ms = recv_window_ms
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.request_timeout_ms = request_timeout_ms


class BybitAdapter(ExchangeAdapter):
    """
    Адаптер для Bybit (V5 API).
    
    Реализует все методы интерфейса ExchangeAdapter.
    
    Использование:
        adapter = BybitAdapter()
        
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
        config: Optional[BybitConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.config = config or BybitConfig()
        
        # Загружаем ключи
        if api_key and api_secret:
            self._api_key = api_key
            self._api_secret = api_secret
        else:
            # Пытаемся загрузить из .env
            try:
                self._api_key, self._api_secret = get_bybit_credentials()
            except ValueError:
                self._api_key = ""
                self._api_secret = ""
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "BybitAdapter":
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
        Отправляет ордер на Bybit.
        
        Типы ордеров:
        - Market: рыночный ордер
        - Limit: лимитный ордер
        
        Для стоп-ордеров используется параметр triggerPrice.
        
        Возвращает order_id, присвоенный биржей.
        """
        await self._ensure_client()
        
        # Формируем тело запроса
        body: Dict[str, Any] = {
            "category": self.config.category,
            "symbol": symbol.upper(),
            "side": self._convert_order_side(side),
            "orderType": self._convert_order_type(order_type),
            "qty": str(quantity),
            "timeInForce": "IOC" if order_type == OrderType.LIMIT_IOC else "GoodTillCancelled",
        }
        
        # Добавляем цену для лимитных ордеров
        if order_type in (OrderType.LIMIT, OrderType.LIMIT_IOC):
            body["price"] = str(price)
        
        # Добавляем стоп-цену для стоп-ордеров
        if stop_price > 0:
            body["triggerPrice"] = str(stop_price)
            body["triggerDirection"] = 2 if side == OrderSide.SELL else 1
        
        # Добавляем клиентский ID ордера
        if client_order_id:
            body["orderLinkId"] = client_order_id
        
        # Отправляем запрос с подписью
        response = await self._signed_request(
            "POST",
            "/v5/order/create",
            body=body,
        )
        
        # Возвращаем order_id
        result = response.get("result", {})
        return str(result.get("orderId", ""))
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Отменяет ордер на Bybit.
        
        Возвращает True, если отмена успешна.
        """
        await self._ensure_client()
        
        body: Dict[str, Any] = {
            "category": self.config.category,
            "symbol": symbol.upper(),
            "orderId": order_id,
        }
        
        try:
            await self._signed_request(
                "POST",
                "/v5/order/cancel",
                body=body,
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
        Получает статус ордера с Bybit.
        
        Возвращает текущий статус ордера.
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "category": self.config.category,
            "symbol": symbol.upper(),
            "orderId": order_id,
        }
        
        response = await self._signed_request(
            "GET",
            "/v5/order/realtime",
            params=params,
        )
        
        # Извлекаем статус
        result = response.get("result", {})
        order_list = result.get("list", [])
        
        if order_list:
            order = order_list[0]
            bybit_status = order.get("orderStatus", "")
            return self._convert_order_status(bybit_status)
        
        return OrderStatus.PENDING
    
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
            "category": self.config.category,
            "symbol": symbol.upper(),
            "orderId": order_id,
        }
        
        response = await self._signed_request(
            "GET",
            "/v5/order/realtime",
            params=params,
        )
        
        # Извлекаем информацию об исполнении
        result = response.get("result", {})
        order_list = result.get("list", [])
        
        if not order_list:
            return {
                "filled_quantity": 0.0,
                "average_price": 0.0,
                "fees": 0.0,
            }
        
        order = order_list[0]
        
        filled_quantity = float(order.get("cumExecQty", 0))
        cum_value = float(order.get("cumExecValue", 0))
        cum_fee = float(order.get("cumExecFee", 0))
        
        # Рассчитываем среднюю цену
        average_price = 0.0
        if filled_quantity > 0:
            average_price = cum_value / filled_quantity
        
        return {
            "filled_quantity": filled_quantity,
            "average_price": average_price,
            "fees": cum_fee,
        }
    
    async def get_account_balance(self) -> float:
        """
        Получает баланс аккаунта в USDT.
        
        Возвращает доступный баланс.
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "accountType": "UNIFIED",
        }
        
        response = await self._signed_request(
            "GET",
            "/v5/account/wallet-balance",
            params=params,
        )
        
        # Ищем баланс в USDT
        result = response.get("result", {})
        account_list = result.get("list", [])
        
        for account in account_list:
            coin_list = account.get("coin", [])
            for coin in coin_list:
                if coin.get("coin") == "USDT":
                    return float(coin.get("availableToWithdraw", 0))
        
        return 0.0
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Получает позицию по символу.
        
        Возвращает словарь с информацией о позиции или None.
        """
        await self._ensure_client()
        
        params: Dict[str, Any] = {
            "category": self.config.category,
            "symbol": symbol.upper(),
        }
        
        response = await self._signed_request(
            "GET",
            "/v5/position/list",
            params=params,
        )
        
        result = response.get("result", {})
        position_list = result.get("list", [])
        
        if position_list:
            position = position_list[0]
            return {
                "symbol": position.get("symbol", ""),
                "positionAmt": float(position.get("size", 0)),
                "entryPrice": float(position.get("entryPrice", 0)),
                "markPrice": float(position.get("markPrice", 0)),
                "unRealizedProfit": float(position.get("unrealisedPnl", 0)),
                "leverage": float(position.get("leverage", 1)),
                "side": position.get("side", ""),
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
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Выполняет подписанный запрос к Bybit.
        
        Все приватные эндпоинты требуют подписи.
        
        Формат подписи Bybit V5:
            HMAC-SHA256(timestamp + api_key + recv_window + query_string)
        
        Для POST запросов query_string = JSON body.
        """
        timestamp = str(int(time.time() * 1000))
        recv_window = str(self.config.recv_window_ms)
        
        # Формируем строку для подписи
        query_string = ""
        
        if method == "GET" and params:
            query_string = urlencode(params)
        elif method == "POST" and body:
            query_string = json.dumps(body, separators=(",", ":"))
        
        # Строка для подписи: timestamp + api_key + recv_window + query
        sign_string = timestamp + self._api_key + recv_window + query_string
        
        # Подписываем
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            sign_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        
        # Заголовки Bybit V5
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        
        # URL
        url = f"{self.config.base_url}{endpoint}"
        
        # Выполняем запрос с повторами
        return await self._request_with_retry(
            method=method,
            url=url,
            params=params if method == "GET" else None,
            json_body=body if method == "POST" else None,
            headers=headers,
        )
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
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
                    json=json_body,
                    headers=headers,
                )
                
                # Проверяем статус ответа
                if response.status_code == 200:
                    data = response.json()
                    
                    # Проверяем код ответа Bybit
                    ret_code = data.get("retCode", 0)
                    if ret_code == 0:
                        return data
                    
                    # Обрабатываем ошибки Bybit
                    ret_msg = data.get("retMsg", "Unknown error")
                    
                    # Некоторые ошибки не стоит повторять
                    if ret_code in (10001, 10002, 10003):
                        # 10001: Invalid API key
                        # 10002: Invalid signature
                        # 10003: Invalid timestamp
                        raise Exception(f"Bybit error {ret_code}: {ret_msg}")
                    
                    # Другие ошибки можно повторить
                    last_error = Exception(f"Bybit error {ret_code}: {ret_msg}")
                    continue
                
                # HTTP ошибки
                last_error = Exception(f"HTTP {response.status_code}")
            
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
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Преобразует тип ордера в формат Bybit."""
        mapping = {
            OrderType.MARKET: "Market",
            OrderType.LIMIT: "Limit",
            OrderType.LIMIT_IOC: "Limit",
        }
        return mapping.get(order_type, "Market")
    
    def _convert_order_side(self, side: OrderSide) -> str:
        """Преобразует сторону ордера в формат Bybit."""
        return "Buy" if side == OrderSide.BUY else "Sell"
    
    def _convert_order_status(self, bybit_status: str) -> OrderStatus:
        """Преобразует статус ордера из формата Bybit в наш."""
        mapping = {
            "Created": OrderStatus.SUBMITTED,
            "New": OrderStatus.SUBMITTED,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
            "Deactivated": OrderStatus.EXPIRED,
        }
        return mapping.get(bybit_status, OrderStatus.PENDING)