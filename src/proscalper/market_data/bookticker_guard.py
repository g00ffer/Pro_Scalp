"""
FastGuard для сигналов.

Использует самый свежий bookTicker как источник последней правды
перед отправкой ордера или во время confirmation window.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from proscalper.core.events import BookTickerEvent, Signal
from proscalper.core.types import OrderSide


@dataclass(frozen=True)
class FastGuardConfig:
    """
    Параметры быстрого защитного слоя.

    max_bookticker_age_ms:
        Максимальный возраст последнего bookTicker.
        Если данные старше, сигнал запрещается.

    max_spread_ticks:
        Максимально допустимый спред в тиках.

    max_chase_ticks:
        Насколько разрешено "догонять" цену за уровнем.

    abort_buffer_ticks:
        Насколько цена может вернуться против пробоя,
        чтобы быстро отменить сетап.
    """
    max_bookticker_age_ms: int = 80
    max_spread_ticks: int = 3
    max_chase_ticks: int = 5
    abort_buffer_ticks: int = 2
    require_bookticker: bool = True


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str

    @staticmethod
    def success() -> "GuardResult":
        return GuardResult(ok=True, reason="OK")

    @staticmethod
    def reject(reason: str) -> "GuardResult":
        return GuardResult(ok=False, reason=reason)


@dataclass
class FastBookTickerState:
    """
    Последний быстрый срез лучших цен.
    """
    symbol: str
    ts_exchange_ns: int
    ts_local_ns: int
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    def age_ms(self, now_ms: int) -> int:
        local_ms = self.ts_local_ns // 1_000_000
        return max(0, now_ms - local_ms)


class BookTickerGuard:
    """
    Быстрый контроллер допустимости сигналов.

    Используется:
    1. Перед отправкой ордера.
    2. Во время динамического confirmation window.
    3. Для экстренной отмены сетапа при резком развороте.
    """

    def __init__(
        self,
        config: FastGuardConfig,
        tick_sizes: Dict[str, float],
        now_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self._config = config
        self._tick_sizes = {k.upper(): v for k, v in tick_sizes.items()}
        self._states: Dict[str, FastBookTickerState] = {}
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)

    def update(self, event: BookTickerEvent) -> None:
        """
        Обновить состояние fast book ticker.
        """
        state = FastBookTickerState(
            symbol=event.symbol.upper(),
            ts_exchange_ns=event.ts_exchange_ns,
            ts_local_ns=event.ts_local_ns,
            bid_price=event.bid_price,
            bid_qty=event.bid_qty,
            ask_price=event.ask_price,
            ask_qty=event.ask_qty,
        )
        self._states[state.symbol] = state

    def set_tick_size(self, symbol: str, tick_size: float) -> None:
        self._tick_sizes[symbol.upper()] = tick_size

    def latest(self, symbol: str) -> Optional[FastBookTickerState]:
        return self._states.get(symbol.upper())

    def can_send_order(self, signal: Signal) -> GuardResult:
        """
        Проверка перед отправкой ордера.
        """
        return self._validate(
            symbol=signal.symbol,
            side=signal.side,
            level_price=signal.level_price,
            check_spread=True,
        )

    def can_continue_confirmation(
        self,
        symbol: str,
        side: OrderSide,
        level_price: float,
    ) -> GuardResult:
        """
        Проверка во время confirmation window.
        """
        return self._validate(
            symbol=symbol,
            side=side,
            level_price=level_price,
            check_spread=True,
        )

    def _validate(
        self,
        symbol: str,
        side: OrderSide,
        level_price: float,
        check_spread: bool,
    ) -> GuardResult:
        symbol = symbol.upper()

        state = self._states.get(symbol)
        if state is None:
            if self._config.require_bookticker:
                return GuardResult.reject("BOOKTICKER_MISSING")
            return GuardResult.success()

        now_ms = self._now_ms()
        age_ms = state.age_ms(now_ms)

        if age_ms > self._config.max_bookticker_age_ms:
            return GuardResult.reject("BOOKTICKER_STALE")

        tick_size = self._tick_sizes.get(symbol)
        if tick_size is None or tick_size <= 0:
            return GuardResult.reject("TICK_SIZE_UNKNOWN")

        if check_spread:
            spread_ticks = int(round(state.spread / tick_size))
            if spread_ticks > self._config.max_spread_ticks:
                return GuardResult.reject("SPREAD_TOO_WIDE")

        if side == OrderSide.BUY:
            max_acceptable_ask = (
                level_price
                + self._config.max_chase_ticks * tick_size
            )

            if state.ask_price > max_acceptable_ask:
                return GuardResult.reject("PRICE_ALREADY_TOO_FAR")

            abort_bid = (
                level_price
                - self._config.abort_buffer_ticks * tick_size
            )

            if state.bid_price < abort_bid:
                return GuardResult.reject("FAST_REVERSAL")

        elif side == OrderSide.SELL:
            min_acceptable_bid = (
                level_price
                - self._config.max_chase_ticks * tick_size
            )

            if state.bid_price < min_acceptable_bid:
                return GuardResult.reject("PRICE_ALREADY_TOO_FAR")

            abort_ask = (
                level_price
                + self._config.abort_buffer_ticks * tick_size
            )

            if state.ask_price > abort_ask:
                return GuardResult.reject("FAST_REVERSAL")

        else:
            return GuardResult.reject("UNKNOWN_ORDER_SIDE")

        return GuardResult.success()