"""
Отправка и управление ордерами на Binance Futures.

Предоставляет:
- OrderParams: параметры ордера
- OrderResponse: ответ от биржи
- BinanceOrderClient: отправка/отмена/проверка статуса

Поддерживаемые типы ордеров:
- MARKET — рыночный (немедленное исполнение)
- LIMIT — лимитный (ожидание по цене)
- STOP_MARKET — стоп-маркет (срабатывает при достижении stopPrice)
- TAKE_PROFIT_MARKET — тейк-профит маркет
- MARKETABLE_LIMIT_IOC — лимитный с немедленным частичным исполнением

Используется в связке с:
- exchange/binance_futures/auth.py (подпись запросов)
- exchange/binance_futures/rest.py (HTTP клиент)
- execution/order_builder.py (построение ордеров)
- execution/bracket.py (bracket orders)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from proscalper.exchange.binance_futures.auth import BinanceAuth
from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient


# ============================================================
# Типы ордеров
# ============================================================

class OrderType(str, Enum):
    """Тип ордера."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    STOP = "STOP"                      # STOP_LIMIT
    TAKE_PROFIT = "TAKE_PROFIT"        # TAKE_PROFIT_LIMIT


class OrderSide(str, Enum):
    """Сторона ордера."""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(str, Enum):
    """Время жизни ордера."""
    GTC = "GTC"          # Good Till Cancel
    IOC = "IOC"          # Immediate Or Cancel
    FOK = "FOK"          # Fill Or Kill
    GTX = "GTX"          # Post Only (maker only)


class WorkingType(str, Enum):
    """Тип цены для триггера стопов."""
    MARK_PRICE = "MARK_PRICE"
    CONTRACT_PRICE = "CONTRACT_PRICE"


class OrderStatus(str, Enum):
    """Статус ордера."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


# ============================================================
# Параметры ордера
# ============================================================

@dataclass
class OrderParams:
    """
    Параметры ордера для отправки на биржу.
    
    Обязательные поля зависят от типа ордера:
    - MARKET: symbol, side, quantity
    - LIMIT: symbol, side, quantity, price, timeInForce
    - STOP_MARKET: symbol, side, quantity, stopPrice
    - MARKETABLE_LIMIT_IOC: symbol, side, quantity, price (timeInForce=IOC)
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    
    # Для LIMIT и STOP_LIMIT
    price: Optional[float] = None
    time_in_force: Optional[TimeInForce] = None
    
    # Для стоп-ордеров
    stop_price: Optional[float] = None
    working_type: WorkingType = WorkingType.CONTRACT_PRICE
    
    # Дополнительные параметры
    reduce_only: bool = False
    close_position: bool = False
    client_order_id: Optional[str] = None
    
    # Callback rate limit (защита от слишком частых запросов)
    activation_price: Optional[float] = None
    callback_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в dict для отправки на биржу."""
        params = {
            "symbol": self.symbol.upper(),
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": str(self.quantity),
        }
        
        if self.price is not None:
            params["price"] = str(self.price)
        
        if self.time_in_force is not None:
            params["timeInForce"] = self.time_in_force.value
        
        if self.stop_price is not None:
            params["stopPrice"] = str(self.stop_price)
            params["workingType"] = self.working_type.value
        
        if self.reduce_only:
            params["reduceOnly"] = "true"
        
        if self.close_position:
            params["closePosition"] = "true"
        
        if self.client_order_id is not None:
            params["newClientOrderId"] = self.client_order_id
        
        if self.callback_rate is not None:
            params["callbackRate"] = str(self.callback_rate)
        
        if self.activation_price is not None:
            params["activationPrice"] = str(self.activation_price)
        
        return params


# ============================================================
# Ответ от биржи
# ============================================================

@dataclass
class OrderResponse:
    """Ответ от биржи на создание/проверку ордера."""
    # Идентификаторы
    order_id: int
    client_order_id: str
    symbol: str
    
    # Параметры ордера
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    stop_price: float = 0.0
    
    # Статус
    status: OrderStatus = OrderStatus.NEW
    
    # Исполнение
    executed_qty: float = 0.0
    cum_quote: float = 0.0          # сумма исполненного в quote currency
    avg_price: float = 0.0
    
    # Время
    update_time: int = 0            # ms
    working_type: str = "CONTRACT_PRICE"
    
    # Raw response от биржи
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
    
    @property
    def is_terminal(self) -> bool:
        """Финальный статус (не изменится)."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )
    
    @classmethod
    def from_exchange(cls, data: Dict[str, Any]) -> OrderResponse:
        """Создание из ответа биржи."""
        return cls(
            order_id=int(data.get("orderId", 0)),
            client_order_id=str(data.get("clientOrderId", "")),
            symbol=str(data.get("symbol", "")),
            side=OrderSide(data.get("side", "BUY")),
            order_type=OrderType(data.get("type", "MARKET")),
            price=float(data.get("price", 0.0)),
            quantity=float(data.get("origQty", 0.0)),
            stop_price=float(data.get("stopPrice", 0.0)),
            status=OrderStatus(data.get("status", "NEW")),
            executed_qty=float(data.get("executedQty", 0.0)),
            cum_quote=float(data.get("cumQuote", 0.0)),
            avg_price=float(data.get("avgPrice", 0.0)),
            update_time=int(data.get("updateTime", 0)),
            working_type=str(data.get("workingType", "CONTRACT_PRICE")),
            raw=data,
        )


# ============================================================
# Ошибки
# ============================================================

class OrderError(Exception):
    """Ошибка при работе с ордерами."""
    def __init__(self, message: str, code: int = 0, raw: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.raw = raw or {}


# ============================================================
# Клиент ордеров
# ============================================================

class BinanceOrderClient:
    """
    Клиент для отправки и управления ордерами.
    
    Использование:
        async with BinanceFuturesRestClient() as rest:
            auth = create_auth(api_key, api_secret)
            order_client = BinanceOrderClient(rest, auth)
            
            # Отправка MARKET ордера
            params = OrderParams(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
            )
            response = await order_client.create_order(params)
            
            # Проверка статуса
            status = await order_client.query_order(
                symbol="BTCUSDT",
                order_id=response.order_id,
            )
            
            # Отмена
            await order_client.cancel_order(
                symbol="BTCUSDT",
                order_id=response.order_id,
            )
    """
    
    def __init__(
        self,
        rest_client: BinanceFuturesRestClient,
        auth: BinanceAuth,
    ) -> None:
        self._rest = rest_client
        self._auth = auth
    
    # ============================================
    # Создание ордеров
    # ============================================
    
    async def create_order(self, params: OrderParams) -> OrderResponse:
        """
        Отправляет ордер на биржу.
        
        Args:
            params: параметры ордера
            
        Returns:
            OrderResponse с подтверждением от биржи
            
        Raises:
            OrderError: если биржа отклонила ордер
        """
        endpoint = f"{self._auth.base_url}/fapi/v1/order"
        
        # Подписываем параметры
        signed_params = self._auth.sign_post_body(params.to_dict())
        
        # Отправляем POST
        headers = self._auth.get_headers()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        response = await self._rest._client.post(
            endpoint,
            content=signed_params.encode("utf-8"),
            headers=headers,
        )
        
        # Проверяем статус
        if response.status_code != 200:
            error_data = response.json()
            raise OrderError(
                message=error_data.get("msg", "Unknown error"),
                code=error_data.get("code", 0),
                raw=error_data,
            )
        
        data = response.json()
        return OrderResponse.from_exchange(data)
    
    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> OrderResponse:
        """Упрощённый метод для MARKET ордера."""
        params = OrderParams(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            client_order_id=client_order_id,
            reduce_only=reduce_only,
        )
        return await self.create_order(params)
    
    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Упрощённый метод для LIMIT ордера."""
        params = OrderParams(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        return await self.create_order(params)
    
    async def create_stop_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        working_type: WorkingType = WorkingType.CONTRACT_PRICE,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> OrderResponse:
        """Упрощённый метод для STOP_MARKET ордера."""
        params = OrderParams(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_MARKET,
            quantity=quantity,
            stop_price=stop_price,
            working_type=working_type,
            client_order_id=client_order_id,
            reduce_only=reduce_only,
        )
        return await self.create_order(params)
    
    async def create_marketable_limit_ioc(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """
        MARKETABLE_LIMIT_IOC — лимитный ордер с немедленным исполнением.
        
        Используется для входа с защитой от проскальзывания:
        - цена ограничена (не хуже price)
        - timeInForce=IOC (только немедленное исполнение)
        - что не исполнилось — отменяется
        """
        params = OrderParams(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.IOC,
            client_order_id=client_order_id,
        )
        return await self.create_order(params)
    
    # ============================================
    # Batch orders (пакетная отправка)
    # ============================================
    
    async def create_batch_orders(
        self,
        orders: List[OrderParams],
    ) -> List[OrderResponse]:
        """
        Отправляет до 5 ордеров одним запросом.
        
        Binance позволяет отправлять до 5 ордеров в одном batch.
        Все ордера должны быть для одного аккаунта.
        
        Args:
            orders: список параметров ордеров (максимум 5)
            
        Returns:
            Список OrderResponse (может содержать ошибки для отдельных ордеров)
        """
        if len(orders) > 5:
            raise ValueError("Batch orders: максимум 5 ордеров за раз")
        
        if len(orders) == 0:
            return []
        
        endpoint = f"{self._auth.base_url}/fapi/v1/batchOrders"
        
        # Формируем batch data
        batch_data = []
        for params in orders:
            signed = self._auth.sign_get_params(params.to_dict())
            batch_data.append(signed)
        
        # Отправляем как batchList
        headers = self._auth.get_headers()
        
        import json
        payload = {"batchOrders": json.dumps(batch_data)}
        
        response = await self._rest._client.post(
            endpoint,
            json=payload,
            headers=headers,
        )
        
        if response.status_code != 200:
            error_data = response.json()
            raise OrderError(
                message=error_data.get("msg", "Batch order failed"),
                code=error_data.get("code", 0),
                raw=error_data,
            )
        
        # Ответ — список результатов (может содержать ошибки для отдельных ордеров)
        results = response.json()
        responses = []
        
        for item in results:
            if "code" in item and item["code"] != 200:
                # Этот ордер отклонён
                responses.append(OrderResponse(
                    order_id=0,
                    client_order_id="",
                    symbol="",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=0.0,
                    status=OrderStatus.REJECTED,
                    raw=item,
                ))
            else:
                responses.append(OrderResponse.from_exchange(item))
        
        return responses
    
    # ============================================
    # Проверка статуса
    # ============================================
    
    async def query_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """
        Запрашивает статус ордера.
        
        Args:
            symbol: символ
            order_id: ID ордера (или client_order_id)
            client_order_id: client order ID (альтернатива order_id)
            
        Returns:
            OrderResponse с текущим статусом
        """
        params = {"symbol": symbol.upper()}
        
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Нужен либо order_id, либо client_order_id")
        
        signed_params = self._auth.sign_get_params(params)
        endpoint = f"{self._auth.base_url}/fapi/v1/order"
        
        response = await self._rest._client.get(
            endpoint,
            params=signed_params,
            headers=self._auth.get_headers(),
        )
        
        if response.status_code != 200:
            error_data = response.json()
            raise OrderError(
                message=error_data.get("msg", "Query failed"),
                code=error_data.get("code", 0),
                raw=error_data,
            )
        
        return OrderResponse.from_exchange(response.json())
    
    async def query_all_orders(
        self,
        symbol: str,
        limit: int = 50,
    ) -> List[OrderResponse]:
        """Запрашивает все ордера для символа."""
        params = {
            "symbol": symbol.upper(),
            "limit": limit,
        }
        
        signed_params = self._auth.sign_get_params(params)
        endpoint = f"{self._auth.base_url}/fapi/v1/allOrders"
        
        response = await self._rest._client.get(
            endpoint,
            params=signed_params,
            headers=self._auth.get_headers(),
        )
        
        if response.status_code != 200:
            error_data = response.json()
            raise OrderError(
                message=error_data.get("msg", "Query all failed"),
                code=error_data.get("code", 0),
                raw=error_data,
            )
        
        return [OrderResponse.from_exchange(item) for item in response.json()]
    
    async def query_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[OrderResponse]:
        """Запрашивает все открытые ордера."""
        params = {}
        if symbol is not None:
            params["symbol"] = symbol.upper()
        
        signed_params = self._auth.sign_get_params(params)
        endpoint = f"{self._auth.base_url}/fapi/v1/openOrders"
        
        response = await self._rest._client.get(
            endpoint,
            params=signed_params,
            headers=self._auth.get_headers(),
        )
        
        if response.status_code != 200:
            error_data = response.json()
            raise OrderError(
                message=error_data.get("msg", "Query open failed"),
                code=error_data.get("code", 0),
                raw=error_data,
            )
        
        return [OrderResponse.from_exchange(item) for item in response.json()]
    
    # ============================================
    # Отмена ордеров
    # ============================================
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """
        Отменяет ордер.
        
        Args:
            symbol: символ
            order_id: ID ордера (или client_order_id)
            client_order_id: client order ID
            
        Returns:
            OrderResponse с подтверждением отмены
        """
        params = {"symbol": symbol.upper()}
        
        if order_id is not None:
            params["orderId"] = order_id
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Нужен либо order_id, либо client_order_id")
        
        signed_params = self._auth.sign_get_params(params)
        endpoint = f"{self._auth.base_url}/fapi/v1/order"
        
        response = await self._rest._client.delete(
            endpoint,
            params=signed_params,
            headers=self._auth.get_headers(),
        )
        
        if response.status_code != 200:
            error_data = response.json()
            raise OrderError(
                message=error_data.get("msg", "Cancel failed"),
                code=error_data.get("code", 0),
                raw=error_data,
            )
        
        return OrderResponse.from_exchange(response.json())
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Отменяет все открытые ордера для символа."""
        params = {"symbol": symbol.upper()}
        signed_params = self._auth.sign_get_params(params)
        endpoint = f"{self._auth.base_url}/fapi/v1/allOpenOrders"
        
        response = await self._rest._client.delete(
            endpoint,
            params=signed_params,
            headers=self._auth.get_headers(),
        )
        
        if response.status_code != 200:
            error_data = response.json()
            raise OrderError(
                message=error_data.get("msg", "Cancel all failed"),
                code=error_data.get("code", 0),
                raw=error_data,
            )
        
        return response.json()
    
    # ============================================
    # Ожидание исполнения
    # ============================================
    
    async def wait_for_fill(
        self,
        symbol: str,
        order_id: int,
        timeout_ms: int = 5000,
        poll_interval_ms: int = 100,
    ) -> OrderResponse:
        """
        Ожидает исполнения ордера с polling.
        
        Args:
            symbol: символ
            order_id: ID ордера
            timeout_ms: максимальное время ожидания
            poll_interval_ms: интервал между запросами
            
        Returns:
            OrderResponse с финальным статусом
            
        Raises:
            TimeoutError: если ордер не исполнился за timeout
        """
        start_ms = int(time.time() * 1000)
        
        while True:
            response = await self.query_order(symbol, order_id=order_id)
            
            if response.is_terminal:
                return response
            
            elapsed_ms = int(time.time() * 1000) - start_ms
            if elapsed_ms >= timeout_ms:
                raise TimeoutError(
                    f"Order {order_id} not filled within {timeout_ms}ms"
                )
            
            await asyncio.sleep(poll_interval_ms / 1000.0)