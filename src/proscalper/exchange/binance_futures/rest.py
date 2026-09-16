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