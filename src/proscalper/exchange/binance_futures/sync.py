"""
Синхронизация состояния аккаунта с биржей.

При старте в режиме `live` нужно знать текущее состояние:
- баланс аккаунта (доступная маржа, эквити)
- открытые позиции (символ, сторона, объём, вход)
- открытые ордера (стопы, лимитки)

Модуль делает это через приватные REST-эндпоинты:
- GET /fapi/v2/account     — баланс + позиции
- GET /fapi/v2/balance     — баланс по активам
- GET /fapi/v2/positionRisk — детально по позициям
- GET /fapi/v1/openOrders  — открытые ордера

Используется в связке с:
- app/live_runner.py (синхронизация при старте)
- execution/reconciliation.py (периодическая сверка)
- risk/position_manager.py (инициализация позиций)
- execution/order_manager.py (восстановление стопов)

Принципы:
- синхронизация идемпотентна (можно вызывать многократно)
- не изменяет состояние на бирже (только чтение)
- при ошибке сети — повтор с экспоненциальным бэкоффом
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from proscalper.exchange.binance_futures.auth import BinanceAuth
from proscalper.exchange.binance_futures.rest import BinanceFuturesRestClient
from proscalper.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# Снимки данных
# ============================================================

@dataclass(frozen=True)
class BalanceSnapshot:
    """Баланс по одному активу."""
    asset: str
    balance: float
    available_balance: float
    cross_wallet_balance: float
    cross_unpnl: float
    margin_balance: float

    @classmethod
    def from_exchange(cls, data: Dict[str, Any]) -> BalanceSnapshot:
        return cls(
            asset=str(data.get("asset", "")),
            balance=float(data.get("balance", 0.0)),
            available_balance=float(data.get("availableBalance", 0.0)),
            cross_wallet_balance=float(data.get("crossWalletBalance", 0.0)),
            cross_unpnl=float(data.get("crossUnPnl", 0.0)),
            margin_balance=float(data.get("marginBalance", 0.0)),
        )


@dataclass(frozen=True)
class PositionSnapshot:
    """Открытая позиция на бирже."""
    symbol: str
    position_side: str        # LONG / SHORT / BOTH
    position_amt: float       # объём (с учётом направления)
    entry_price: float
    unrealized_pnl: float
    leverage: float
    liquidation_price: float
    margin: float
    isolated_margin: float
    notional: float
    mark_price: float

    @classmethod
    def from_exchange(cls, data: Dict[str, Any]) -> PositionSnapshot:
        return cls(
            symbol=str(data.get("symbol", "")),
            position_side=str(data.get("positionSide", "BOTH")),
            position_amt=float(data.get("positionAmt", 0.0)),
            entry_price=float(data.get("entryPrice", 0.0)),
            unrealized_pnl=float(data.get("unRealizedProfit", 0.0)),
            leverage=float(data.get("leverage", 1.0)),
            liquidation_price=float(data.get("liquidationPrice", 0.0)),
            margin=float(data.get("margin", 0.0)),
            isolated_margin=float(data.get("isolatedMargin", 0.0)),
            notional=float(data.get("notional", 0.0)),
            mark_price=float(data.get("markPrice", 0.0)),
        )

    @property
    def is_open(self) -> bool:
        return abs(self.position_amt) > 1e-12

    @property
    def is_long(self) -> bool:
        return self.position_amt > 0

    @property
    def is_short(self) -> bool:
        return self.position_amt < 0


@dataclass(frozen=True)
class OpenOrderSnapshot:
    """Открытый ордер на бирже."""
    order_id: int
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    price: float
    stop_price: float
    orig_qty: float
    executed_qty: float
    status: str
    time_in_force: str
    working_type: str

    @classmethod
    def from_exchange(cls, data: Dict[str, Any]) -> OpenOrderSnapshot:
        return cls(
            order_id=int(data.get("orderId", 0)),
            client_order_id=str(data.get("clientOrderId", "")),
            symbol=str(data.get("symbol", "")),
            side=str(data.get("side", "")),
            order_type=str(data.get("type", "")),
            price=float(data.get("price", 0.0)),
            stop_price=float(data.get("stopPrice", 0.0)),
            orig_qty=float(data.get("origQty", 0.0)),
            executed_qty=float(data.get("executedQty", 0.0)),
            status=str(data.get("status", "")),
            time_in_force=str(data.get("timeInForce", "")),
            working_type=str(data.get("workingType", "")),
        )


@dataclass
class SyncResult:
    """Полный результат синхронизации."""
    ts_ns: int = 0
    success: bool = False
    error: str = ""

    # Данные
    balances: List[BalanceSnapshot] = field(default_factory=list)
    positions: List[PositionSnapshot] = field(default_factory=list)
    open_orders: List[OpenOrderSnapshot] = field(default_factory=list)

    # Агрегаты
    total_wallet_balance: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_available_balance: float = 0.0
    open_positions_count: int = 0
    open_orders_count: int = 0

    def get_balance(self, asset: str = "USDT") -> Optional[BalanceSnapshot]:
        for b in self.balances:
            if b.asset == asset:
                return b
        return None

    def get_position(self, symbol: str) -> Optional[PositionSnapshot]:
        symbol = symbol.upper()
        for p in self.positions:
            if p.symbol == symbol and p.is_open:
                return p
        return None

    def get_open_orders(self, symbol: str) -> List[OpenOrderSnapshot]:
        symbol = symbol.upper()
        return [o for o in self.open_orders if o.symbol == symbol]


# ============================================================
# Синхронизатор
# ============================================================

class BinanceSyncer:
    """
    Синхронизация состояния аккаунта с биржей.

    Использование:
        syncer = BinanceSyncer(rest_client, auth)

        # Полная синхронизация
        result = await syncer.full_sync()
        if result.success:
            for pos in result.positions:
                if pos.is_open:
                    print(f"{pos.symbol}: {pos.position_amt}")

        # Только баланс
        balances = await syncer.sync_balances()

        # Только позиции
        positions = await syncer.sync_positions()

        # Только открытые ордера
        orders = await syncer.sync_open_orders("BTCUSDT")
    """

    def __init__(
        self,
        rest_client: BinanceFuturesRestClient,
        auth: BinanceAuth,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ) -> None:
        self._rest = rest_client
        self._auth = auth
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec

    # ============================================
    # Полная синхронизация
    # ============================================

    async def full_sync(
        self,
        symbols: Optional[List[str]] = None,
    ) -> SyncResult:
        """
        Полная синхронизация: баланс + позиции + ордера.

        Вызывается при старте в режиме `live`.
        """
        result = SyncResult(ts_ns=time.time_ns())

        try:
            # Баланс и позиции из одного эндпоинта
            account_data = await self._fetch_account()

            if account_data is not None:
                # Балансы
                raw_balances = account_data.get("assets", [])
                result.balances = [
                    BalanceSnapshot.from_exchange(b) for b in raw_balances
                ]

                # Позиции
                raw_positions = account_data.get("positions", [])
                result.positions = [
                    PositionSnapshot.from_exchange(p) for p in raw_positions
                ]

                # Агрегаты
                result.total_wallet_balance = sum(
                    b.cross_wallet_balance for b in result.balances
                )
                result.total_unrealized_pnl = sum(
                    b.cross_unpnl for b in result.balances
                )
                result.total_available_balance = sum(
                    b.available_balance for b in result.balances
                )
                result.open_positions_count = sum(
                    1 for p in result.positions if p.is_open
                )

            # Открытые ордера
            orders = await self._fetch_open_orders(symbols)
            result.open_orders = orders
            result.open_orders_count = len(orders)

            result.success = True

            logger.info(
                "Account sync complete",
                balance=f"{result.total_wallet_balance:.2f}",
                positions=result.open_positions_count,
                orders=result.open_orders_count,
            )

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            logger.error(f"Account sync failed: {exc}")

        return result

    # ============================================
    # Отдельные синхронизации
    # ============================================

    async def sync_balances(self) -> List[BalanceSnapshot]:
        """Синхронизация балансов."""
        data = await self._fetch_balances()
        if data is None:
            return []
        return [BalanceSnapshot.from_exchange(b) for b in data]

    async def sync_positions(self) -> List[PositionSnapshot]:
        """Синхронизация позиций."""
        data = await self._fetch_position_risk()
        if data is None:
            return []
        return [PositionSnapshot.from_exchange(p) for p in data]

    async def sync_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[OpenOrderSnapshot]:
        """Синхронизация открытых ордеров."""
        return await self._fetch_open_orders(
            [symbol] if symbol else None
        )

    # ============================================
    # Внутренние запросы
    # ============================================

    async def _fetch_account(self) -> Optional[Dict[str, Any]]:
        """GET /fapi/v2/account — баланс + позиции."""
        return await self._request_with_retry(
            "GET",
            "/fapi/v2/account",
        )

    async def _fetch_balances(self) -> Optional[List[Dict[str, Any]]]:
        """GET /fapi/v2/balance — баланс по активам."""
        return await self._request_with_retry(
            "GET",
            "/fapi/v2/balance",
        )

    async def _fetch_position_risk(self) -> Optional[List[Dict[str, Any]]]:
        """GET /fapi/v2/positionRisk — детально по позициям."""
        return await self._request_with_retry(
            "GET",
            "/fapi/v2/positionRisk",
        )

    async def _fetch_open_orders(
        self,
        symbols: Optional[List[str]] = None,
    ) -> List[OpenOrderSnapshot]:
        """GET /fapi/v1/openOrders — открытые ордера."""
        if symbols is None:
            data = await self._request_with_retry(
                "GET",
                "/fapi/v1/openOrders",
            )
        else:
            # Binance требует symbol для /fapi/v1/openOrders
            # Запрашиваем по каждому символу
            all_orders = []
            for symbol in symbols:
                data = await self._request_with_retry(
                    "GET",
                    "/fapi/v1/openOrders",
                    params={"symbol": symbol.upper()},
                )
                if data is not None:
                    all_orders.extend(data)
            return [
                OpenOrderSnapshot.from_exchange(o) for o in all_orders
            ]

        if data is None:
            return []
        return [OpenOrderSnapshot.from_exchange(o) for o in data]

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Выполняет подписанный запрос с ретраями."""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                # Подписываем параметры
                signed_params = self._auth.sign_get_params(params or {})
                endpoint = f"{self._auth.base_url}{path}"

                response = await self._rest._client.request(
                    method,
                    endpoint,
                    params=signed_params,
                    headers=self._auth.get_headers(),
                )

                if response.status_code == 200:
                    return response.json()

                error_data = response.json()
                error_code = error_data.get("code", 0)

                # Rate limit — ждём дольше
                if error_code == -1003:
                    wait = self._retry_delay_sec * (attempt + 1) * 5
                    logger.warning(
                        f"Rate limit hit, waiting {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    continue

                # Прочие ошибки
                logger.error(
                    f"API error: {error_code} - {error_data.get('msg')}"
                )
                return None

            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    wait = self._retry_delay_sec * (attempt + 1)
                    logger.warning(
                        f"Request failed, retry in {wait:.1f}s: {exc}"
                    )
                    await asyncio.sleep(wait)

        if last_error:
            logger.error(f"All retries exhausted: {last_error}")
        return None