"""
Реестр инструментов.

Хранит информацию о торговых инструментах и предоставляет
доступ к конвертерам цен и объёмов для каждого символа.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from proscalper.core.types import InstrumentInfo, Symbol
from proscalper.core.price_math import PriceConverter, QuantityConverter


class InstrumentRegistry:
    """
    Реестр инструментов с кэшем конвертеров.

    Использование:
        registry = InstrumentRegistry()
        registry.load_from_exchange_info(instruments)

        tick_size = registry.get_tick_size("BTCUSDT")
        converter = registry.get_price_converter("BTCUSDT")
    """

    def __init__(self) -> None:
        self._instruments: Dict[str, InstrumentInfo] = {}
        self._price_converters: Dict[str, PriceConverter] = {}
        self._quantity_converters: Dict[str, QuantityConverter] = {}

    def register(self, instrument: InstrumentInfo) -> None:
        """
        Зарегистрировать инструмент.
        """
        symbol = instrument.symbol.upper()

        self._instruments[symbol] = instrument

        self._price_converters[symbol] = PriceConverter(
            tick_size=instrument.tick_size,
            precision=instrument.price_precision,
        )

        self._quantity_converters[symbol] = QuantityConverter(
            step_size=instrument.step_size,
            precision=instrument.quantity_precision,
        )

    def load_from_exchange_info(self, instruments: List[InstrumentInfo]) -> None:
        """
        Загрузить список инструментов из exchangeInfo.
        """
        for instrument in instruments:
            self.register(instrument)

    def get_instrument(self, symbol: str) -> Optional[InstrumentInfo]:
        """
        Получить информацию об инструменте.
        """
        return self._instruments.get(symbol.upper())

    def get_price_converter(self, symbol: str) -> Optional[PriceConverter]:
        """
        Получить конвертер цен для символа.
        """
        return self._price_converters.get(symbol.upper())

    def get_quantity_converter(self, symbol: str) -> Optional[QuantityConverter]:
        """
        Получить конвертер объёмов для символа.
        """
        return self._quantity_converters.get(symbol.upper())

    def get_tick_size(self, symbol: str) -> Optional[float]:
        """
        Получить tick_size для символа.
        """
        instrument = self.get_instrument(symbol)
        return instrument.tick_size if instrument else None

    def get_step_size(self, symbol: str) -> Optional[float]:
        """
        Получить step_size для символа.
        """
        instrument = self.get_instrument(symbol)
        return instrument.step_size if instrument else None

    def all_symbols(self) -> List[str]:
        """
        Все зарегистрированные символы.
        """
        return list(self._instruments.keys())

    def trading_symbols(self) -> List[str]:
        """
        Только символы в статусе TRADING.
        """
        return [
            symbol
            for symbol, instrument in self._instruments.items()
            if instrument.is_trading
        ]