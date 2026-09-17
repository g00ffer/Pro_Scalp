"""
REST-клиент для Binance USDT-M Futures.

Публичные эндпоинты (без подписи):
- /fapi/v1/exchangeInfo
- /fapi/v1/time
- /fapi/v1/depth

Используется для:
- получения информации об инструментах
- синхронизации времени
- снапшотов стакана при ресинхронизации
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import time
import httpx
import msgspec

from proscalper.core.types import InstrumentInfo, Symbol


class BinanceFuturesRestClient:
    """
    Асинхронный REST-клиент для Binance Futures.

    Использовать как контекстный менеджер:
        async with BinanceFuturesRestClient() as client:
            info = await client.get_exchange_info()
    """

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "BinanceFuturesRestClient":
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_server_time(self) -> int:
        """
        Возвращает время сервера в миллисекундах.
        Используется для синхронизации часов.
        """
        if self._client is None:
            raise RuntimeError("Клиент не инициализирован. Используй 'async with'.")

        response = await self._client.get(f"{self._base_url}/fapi/v1/time")
        response.raise_for_status()

        data = msgspec.json.decode(response.content)
        return int(data["serverTime"])

    async def get_exchange_info(self) -> List[InstrumentInfo]:
        """
        Возвращает информацию обо всех инструментах.

        Парсит filters для извлечения:
        - PRICE_FILTER -> tickSize
        - LOT_SIZE -> stepSize
        - MIN_NOTIONAL -> notional
        """
        if self._client is None:
            raise RuntimeError("Клиент не инициализирован. Используй 'async with'.")

        response = await self._client.get(f"{self._base_url}/fapi/v1/exchangeInfo")
        response.raise_for_status()

        data = msgspec.json.decode(response.content)

        instruments: List[InstrumentInfo] = []

        for symbol_info in data.get("symbols", []):
            instrument = self._parse_symbol_info(symbol_info)
            if instrument is not None:
                instruments.append(instrument)

        return instruments

    async def get_depth_snapshot(
        self,
        symbol: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Возвращает сырой снапшот стакана.

        Используется при ресинхронизации локального стакана.
        """
        if self._client is None:
            raise RuntimeError("Клиент не инициализирован. Используй 'async with'.")

        response = await self._client.get(
            f"{self._base_url}/fapi/v1/depth",
            params={"symbol": symbol.upper(), "limit": limit},
        )
        response.raise_for_status()

        return msgspec.json.decode(response.content)

    def _parse_symbol_info(
        self,
        symbol_info: Dict[str, Any],
    ) -> Optional[InstrumentInfo]:
        """
        Парсит информацию об одном символе из exchangeInfo.
        """
        try:
            symbol = symbol_info.get("symbol", "")
            base_asset = symbol_info.get("baseAsset", "")
            quote_asset = symbol_info.get("quoteAsset", "")
            status = symbol_info.get("status", "")
            price_precision = symbol_info.get("pricePrecision", 0)
            quantity_precision = symbol_info.get("quantityPrecision", 0)

            # Извлекаем фильтры
            tick_size = 0.0
            step_size = 0.0
            min_notional = 0.0

            for filter_info in symbol_info.get("filters", []):
                filter_type = filter_info.get("filterType", "")

                if filter_type == "PRICE_FILTER":
                    tick_size = float(filter_info.get("tickSize", 0))

                elif filter_type == "LOT_SIZE":
                    step_size = float(filter_info.get("stepSize", 0))

                elif filter_type == "MIN_NOTIONAL":
                    min_notional = float(filter_info.get("notional", 0))

            # Пропускаем инструменты с некорректными параметрами
            if tick_size <= 0 or step_size <= 0:
                return None

            return InstrumentInfo(
                symbol=Symbol(symbol),
                base_asset=base_asset,
                quote_asset=quote_asset,
                tick_size=tick_size,
                step_size=step_size,
                min_notional=min_notional,
                price_precision=price_precision,
                quantity_precision=quantity_precision,
                is_trading=(status == "TRADING"),
            )

        except Exception:
            return None

    async def get_klines(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 288,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List]:
        """
        Загрузка исторических свечей (klines).
        
        Используется для:
        - Построения карты уровней при старте
        - Бэктестинга
        
        Args:
            symbol: Символ инструмента (например, "BTCUSDT")
            interval: Таймфрейм свечи. Возможные значения:
                      1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
            limit: Количество свечей (максимум 1500)
            start_time: Время начала в миллисекундах (опционально)
            end_time: Время конца в миллисекундах (опционально)
        
        Returns:
            Список свечей в формате Binance:
            [
                [
                    1499040000000,      # Open time (ms)
                    "0.01634790",       # Open
                    "0.80000000",       # High
                    "0.01575800",       # Low
                    "0.01577100",       # Close
                    "148976.11427815",  # Volume
                    1499644799999,      # Close time (ms)
                    "2434.19055334",    # Quote asset volume
                    308,                # Number of trades
                    "1756.87402397",    # Taker buy base asset volume
                    "28.46694368",      # Taker buy quote asset volume
                    "17928899.62484339" # Ignore
                ],
                ...
            ]
        
        Пример:
            # 24 часа 5-минутных свечей
            klines = await client.get_klines("BTCUSDT", interval="5m", limit=288)
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1500),
        }
        
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        
        response = await self._request_with_retry(
            "GET",
            f"{self._base_url}/fapi/v1/klines",
            params=params,
        )
        
        return msgspec.json.decode(response.content)
    
    async def get_klines_last_hours(
        self,
        symbol: str,
        interval: str = "5m",
        hours: float = 24.0,
    ) -> List[List]:
        """
        Загрузка свечей за последние N часов.
        
        Удобная обёртка над get_klines() для загрузки истории
        при старте системы.
        
        Пример:
            # 24 часа 5-минутных свечей
            klines = await client.get_klines_last_hours("BTCUSDT", interval="5m", hours=24)
        """
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(hours * 3600 * 1000)
        
        # Рассчитываем нужное количество свечей
        interval_map = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "8h": 480, "12h": 720, "1d": 1440,
        }
        interval_minutes = interval_map.get(interval, 5)
        
        # Количество свечей за указанный период
        total_minutes = hours * 60
        limit = int(total_minutes / interval_minutes) + 1
        limit = min(limit, 1500)  # API максимум
        
        return await self.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_ms,
        )            