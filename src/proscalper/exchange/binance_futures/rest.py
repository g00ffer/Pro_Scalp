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

    # ============================================
    # Внутренний HTTP-метод с retry
    # ============================================

    async def _request_with_retry(
        self,
        method: str,
        url_or_path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 5,
    ) -> Any:
        """
        Универсальный HTTP-запрос с повторами.

        Возвращает httpx.Response объект (а не dict!),
        чтобы вызывающий код мог работать с .content / .json().

        Args:
            method: HTTP метод ("GET", "POST", и т.д.)
            url_or_path: либо полный URL (https://fapi.binance.com/...),
                         либо относительный путь (/fapi/v1/...)
            params: query-параметры
            json_body: JSON body для POST/PUT
            headers: HTTP заголовки
            max_retries: максимальное число попыток
        """
        import asyncio

        # Определяем финальный URL
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            url = url_or_path
        else:
            url = f"{self._base_url}{url_or_path}"

        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                request_kwargs: Dict[str, Any] = {
                    "timeout": 30.0,
                }
                if params is not None:
                    request_kwargs["params"] = params
                if json_body is not None:
                    request_kwargs["json"] = json_body
                if headers is not None:
                    request_kwargs["headers"] = headers

                response = await self._client.request(
                    method.upper(),
                    url,
                    **request_kwargs,
                )

                # Успех — возвращаем response целиком
                if response.status_code == 200:
                    return response

                # Rate limit — ждём и повторяем
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", 2 ** (attempt + 1))
                    )
                    wait_sec = min(retry_after, 30)
                    print(
                        f"[REST] Rate limit (429), "
                        f"waiting {wait_sec}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_sec)
                    continue

                # Серверные ошибки — повторяем
                if response.status_code >= 500:
                    wait_sec = min(2 ** (attempt + 1), 30)
                    print(
                        f"[REST] Server error {response.status_code}, "
                        f"retry in {wait_sec}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_sec)
                    continue

                # Клиентские ошибки — сразу падаем
                error_text = response.text[:500]
                raise RuntimeError(
                    f"HTTP {response.status_code} "
                    f"for {method.upper()} {url}: {error_text}"
                )

            except RuntimeError:
                # Клиентская ошибка — не повторяем
                raise

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait_sec = min(2 ** (attempt + 1), 15)
                    print(
                        f"[REST] Network error: {exc}, "
                        f"retry in {wait_sec}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_sec)
                else:
                    break

        raise RuntimeError(
            f"Request failed after {max_retries} attempts: "
            f"{method.upper()} {url}: {last_exc}"
        )

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