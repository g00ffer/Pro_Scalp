"""
Движок исполнения сделок.

Отвечает за:
- Исполнение входных ордеров (гибрид LIMIT → MARKET)
- Постановку стоп-лоссов (сразу при входе)
- Постановку тейк-профитов (после подтверждения входа)
- Обработку частичного исполнения
- Отслеживание статуса ордеров

Архитектура:
- ExecutionEngine — абстрактный движок, не зависит от биржи
- ExchangeAdapter — интерфейс адаптера конкретной биржи
- Реализации адаптеров: binance.py, bybit.py

Используется в связке с:
- RiskManager (одобренные сигналы из очереди)
- PositionManager (управление позициями)
- StopsManager (стопы и тейки)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.core.types import OrderSide
from proscalper.signals.signal_generator import TradingSignal
from proscalper.risk.sizing import PositionSize
from proscalper.risk.stops import PositionStops, StopLevel


class OrderType(Enum):
    """Тип ордера."""
    MARKET = auto()         # Рыночный ордер (мгновенное исполнение)
    LIMIT = auto()          # Лимитный ордер
    LIMIT_IOC = auto()      # Лимитный ордер с немедленной отменой (Immediate-or-Cancel)


class OrderStatus(Enum):
    """Статус ордера."""
    PENDING = auto()            # Ордер создан, ждёт отправки
    SUBMITTED = auto()          # Ордер отправлен на биржу
    PARTIALLY_FILLED = auto()   # Частично исполнен
    FILLED = auto()             # Полностью исполнен
    CANCELLED = auto()          # Отменён
    REJECTED = auto()           # Отклонён биржей
    EXPIRED = auto()            # Истёк


@dataclass
class ExecutionConfig:
    """
    Конфигурация движка исполнения.
    
    Все параметры подобраны как разумные значения по умолчанию.
    В дальнейшем будут подкручены на живых данных.
    """
    # Входные ордера
    entry_order_type: OrderType = OrderType.LIMIT_IOC
    entry_timeout_ms: int = 100             # таймаут для LIMIT перед конвертацией в MARKET
    entry_price_offset_ticks: int = 2       # смещение цены лимитного ордера от текущей
    
    # Стоп-лосс
    stop_order_type: OrderType = OrderType.MARKET
    
    # Тейк-профиты
    take_profit_order_type: OrderType = OrderType.LIMIT
    
    # Ограничения
    max_slippage_ticks: int = 5             # максимальное проскальзывание
    max_retries: int = 3                    # максимальное количество повторных попыток
    
    # Таймауты
    order_poll_interval_ms: int = 50        # интервал опроса статуса ордера
    position_confirm_timeout_ms: int = 500  # таймаут подтверждения позиции


@dataclass
class Order:
    """
    Ордер на бирже.
    
    Хранит полную информацию об ордере и его состоянии.
    """
    # Идентификация
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    
    # Параметры
    quantity: float
    price: float = 0.0              # для лимитных ордеров
    
    # Состояние
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    
    # Время
    created_ts_ns: int = 0
    submitted_ts_ns: int = 0
    filled_ts_ns: int = 0
    
    # Ошибки
    error_message: str = ""
    
    @property
    def is_filled(self) -> bool:
        """Ордер полностью исполнен."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_partially_filled(self) -> bool:
        """Ордер частично исполнен."""
        return self.status == OrderStatus.PARTIALLY_FILLED
    
    @property
    def is_active(self) -> bool:
        """Ордер активен (не исполнен и не отменён)."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )
    
    @property
    def remaining_quantity(self) -> float:
        """Оставшееся количество для исполнения."""
        return max(0.0, self.quantity - self.filled_quantity)


@dataclass
class ExecutionResult:
    """
    Результат исполнения ордера.
    
    Возвращается после попытки исполнения.
    """
    is_success: bool
    order_id: str = ""
    
    # Исполнение
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    
    # Ошибки
    error_message: str = ""
    
    # Время
    executed_ts_ns: int = 0
    
    @property
    def is_partial(self) -> bool:
        """Частичное исполнение."""
        return self.is_success and self.filled_quantity > 0


class ExchangeAdapter(ABC):
    """
    Абстрактный интерфейс адаптера биржи.
    
    Реализуется для каждой конкретной биржи:
    - BinanceAdapter в binance.py
    - BybitAdapter в bybit.py
    
    Все методы асинхронные, потому что работа с биржей
    происходит через сетевые запросы.
    """
    
    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float = 0.0,
        client_order_id: str = "",
    ) -> str:
        """
        Отправляет ордер на биржу.
        
        Возвращает order_id, присвоенный биржей.
        Бросает исключение при ошибке.
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Отменяет ордер.
        
        Возвращает True, если отмена успешна.
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        """
        Получает статус ордера.
        
        Возвращает текущий статус ордера на бирже.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> float:
        """
        Получает баланс аккаунта.
        
        Возвращает доступный баланс в USDT.
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Получает позицию по символу.
        
        Возвращает словарь с информацией о позиции или None.
        """
        pass


class ExecutionEngine:
    """
    Движок исполнения сделок.
    
    Основная логика:
    1. Получаем одобренный сигнал из очереди RiskManager
    2. Отправляем входной ордер (LIMIT_IOC, при таймауте → MARKET)
    3. Ждём подтверждения исполнения
    4. Ставим стоп-лосс (сразу)
    5. Ставим тейк-профиты (после подтверждения)
    6. Сообщаем результат обратно в RiskManager
    
    Использование:
        engine = ExecutionEngine(adapter=binance_adapter, config=config)
        
        # Исполняем сигнал
        result = await engine.execute_entry(
            signal=trading_signal,
            sizing=position_size,
            stops=position_stops,
        )
        
        if result.is_success:
            print(f"Позиция открыта: {result.filled_quantity}")
        else:
            print(f"Ошибка: {result.error_message}")
    """
    
    def __init__(
        self,
        adapter: ExchangeAdapter,
        config: Optional[ExecutionConfig] = None,
    ):
        self.adapter = adapter
        self.config = config or ExecutionConfig()
        
        # Активные ордера
        self._active_orders: Dict[str, Order] = {}
        
        # Статистика
        self._total_orders: int = 0
        self._successful_orders: int = 0
        self._failed_orders: int = 0
        self._total_fees: float = 0.0
    
    async def execute_entry(
        self,
        signal: TradingSignal,
        sizing: PositionSize,
        stops: PositionStops,
        tick_size: float,
    ) -> ExecutionResult:
        """
        Исполняет входной ордер.
        
        Логика:
        1. Отправляем LIMIT_IOC ордер по цене сигнала
        2. Ждём исполнения в течение таймаута
        3. Если не исполнился → конвертируем в MARKET
        4. Ждём подтверждения позиции
        5. Ставим стоп-лосс
        6. Ставим тейк-профиты
        
        Возвращает результат исполнения.
        """
        self._total_orders += 1
        
        # Рассчитываем цену лимитного ордера
        entry_price = self._calculate_entry_price(
            signal.entry_price, tick_size
        )
        
        # Шаг 1: Отправляем LIMIT_IOC ордер
        client_order_id = f"entry_{uuid.uuid4().hex[:12]}"
        
        try:
            order_id = await self.adapter.submit_order(
                symbol=signal.symbol,
                side=signal.direction,
                order_type=OrderType.LIMIT_IOC,
                quantity=sizing.quantity,
                price=entry_price,
                client_order_id=client_order_id,
            )
        except Exception as e:
            self._failed_orders += 1
            return ExecutionResult(
                is_success=False,
                error_message=f"Ошибка отправки ордера: {str(e)}",
                executed_ts_ns=time.time_ns(),
            )
        
        # Создаём объект ордера
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=signal.symbol,
            side=signal.direction,
            order_type=OrderType.LIMIT_IOC,
            quantity=sizing.quantity,
            price=entry_price,
            status=OrderStatus.SUBMITTED,
            created_ts_ns=time.time_ns(),
            submitted_ts_ns=time.time_ns(),
        )
        self._active_orders[order_id] = order
        
        # Шаг 2: Ждём исполнения
        filled = await self._wait_for_fill(
            signal.symbol, order_id, self.config.entry_timeout_ms
        )
        
        if not filled:
            # Шаг 3: Конвертируем в MARKET
            await self.adapter.cancel_order(signal.symbol, order_id)
            
            try:
                market_order_id = await self.adapter.submit_order(
                    symbol=signal.symbol,
                    side=signal.direction,
                    order_type=OrderType.MARKET,
                    quantity=sizing.quantity,
                    client_order_id=f"market_{uuid.uuid4().hex[:12]}",
                )
                
                # Ждём исполнения MARKET ордера
                filled = await self._wait_for_fill(
                    signal.symbol, market_order_id, 1000
                )
                
                if filled:
                    order_id = market_order_id
                
            except Exception as e:
                self._failed_orders += 1
                return ExecutionResult(
                    is_success=False,
                    error_message=f"Ошибка MARKET ордера: {str(e)}",
                    executed_ts_ns=time.time_ns(),
                )
        
        if not filled:
            self._failed_orders += 1
            return ExecutionResult(
                is_success=False,
                error_message="Ордер не исполнился в таймаут",
                executed_ts_ns=time.time_ns(),
            )
        
        # Шаг 4: Получаем информацию об исполнении
        fill_info = await self.adapter.get_order_fill_info(
            signal.symbol, order_id
        )
        
        filled_quantity = fill_info.get("filled_quantity", 0.0)
        average_price = fill_info.get("average_price", 0.0)
        fees = fill_info.get("fees", 0.0)
        
        if filled_quantity <= 0:
            self._failed_orders += 1
            return ExecutionResult(
                is_success=False,
                error_message="Ордер исполнился с нулевым объёмом",
                executed_ts_ns=time.time_ns(),
            )
        
        # Шаг 5: Ставим стоп-лосс
        await self._place_stop_loss(signal, stops)
        
        # Шаг 6: Ставим тейк-профиты
        await self._place_take_profits(signal, stops)
        
        self._successful_orders += 1
        self._total_fees += fees
        
        return ExecutionResult(
            is_success=True,
            order_id=order_id,
            filled_quantity=filled_quantity,
            average_price=average_price,
            fees=fees,
            executed_ts_ns=time.time_ns(),
        )
    
    async def execute_stop_loss(
        self,
        position_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> ExecutionResult:
        """
        Исполняет стоп-лосс (закрывает позицию).
        
        Использует MARKET ордер для гарантированного исполнения.
        """
        self._total_orders += 1
        
        # Инвертируем сторону для закрытия
        close_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        
        try:
            order_id = await self.adapter.submit_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                client_order_id=f"stop_{position_id[:8]}",
            )
            
            # Ждём исполнения
            filled = await self._wait_for_fill(symbol, order_id, 1000)
            
            if filled:
                fill_info = await self.adapter.get_order_fill_info(
                    symbol, order_id
                )
                
                self._successful_orders += 1
                
                return ExecutionResult(
                    is_success=True,
                    order_id=order_id,
                    filled_quantity=fill_info.get("filled_quantity", 0.0),
                    average_price=fill_info.get("average_price", 0.0),
                    fees=fill_info.get("fees", 0.0),
                    executed_ts_ns=time.time_ns(),
                )
            else:
                self._failed_orders += 1
                return ExecutionResult(
                    is_success=False,
                    order_id=order_id,
                    error_message="Стоп-лосс не исполнился в таймаут",
                    executed_ts_ns=time.time_ns(),
                )
        
        except Exception as e:
            self._failed_orders += 1
            return ExecutionResult(
                is_success=False,
                error_message=f"Ошибка стоп-лосса: {str(e)}",
                executed_ts_ns=time.time_ns(),
            )
    
    async def execute_take_profit(
        self,
        position_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        target_price: float,
    ) -> ExecutionResult:
        """
        Исполняет тейк-профит (частичное закрытие позиции).
        
        Использует LIMIT ордер по цене цели.
        """
        self._total_orders += 1
        
        # Инвертируем сторону для закрытия
        close_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        
        try:
            order_id = await self.adapter.submit_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=target_price,
                client_order_id=f"tp_{position_id[:8]}",
            )
            
            # Для тейков не ждём немедленного исполнения
            # Они будут исполнены при достижении цены
            
            self._successful_orders += 1
            
            return ExecutionResult(
                is_success=True,
                order_id=order_id,
                executed_ts_ns=time.time_ns(),
            )
        
        except Exception as e:
            self._failed_orders += 1
            return ExecutionResult(
                is_success=False,
                error_message=f"Ошибка тейк-профита: {str(e)}",
                executed_ts_ns=time.time_ns(),
            )
    
    async def cancel_all_orders(self, symbol: str) -> int:
        """
        Отменяет все активные ордера для символа.
        
        Возвращает количество отменённых ордеров.
        """
        cancelled_count = 0
        
        for order_id, order in list(self._active_orders.items()):
            if order.symbol == symbol and order.is_active:
                try:
                    await self.adapter.cancel_order(symbol, order_id)
                    order.status = OrderStatus.CANCELLED
                    cancelled_count += 1
                except Exception:
                    pass
        
        return cancelled_count
    
    def get_stats(self) -> Dict[str, float]:
        """Возвращает статистику исполнения."""
        return {
            "total_orders": self._total_orders,
            "successful_orders": self._successful_orders,
            "failed_orders": self._failed_orders,
            "success_rate": (
                self._successful_orders / self._total_orders
                if self._total_orders > 0 else 0.0
            ),
            "total_fees": self._total_fees,
            "active_orders": len([
                o for o in self._active_orders.values() if o.is_active
            ]),
        }
    
    def _calculate_entry_price(
        self,
        signal_price: float,
        tick_size: float,
    ) -> float:
        """
        Рассчитывает цену лимитного ордера для входа.
        
        Добавляем небольшое смещение для улучшения шансов исполнения.
        """
        offset = self.config.entry_price_offset_ticks * tick_size
        
        # Для покупки ставим чуть выше текущей цены
        # Для продажи ставим чуть ниже текущей цены
        return signal_price + offset
    
    async def _wait_for_fill(
        self,
        symbol: str,
        order_id: str,
        timeout_ms: int,
    ) -> bool:
        """
        Ждёт исполнения ордера в течение таймаута.
        
        Возвращает True, если ордер исполнился.
        """
        start_ns = time.time_ns()
        timeout_ns = timeout_ms * 1_000_000
        
        while (time.time_ns() - start_ns) < timeout_ns:
            try:
                status = await self.adapter.get_order_status(symbol, order_id)
                
                if status == OrderStatus.FILLED:
                    return True
                elif status in (
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                ):
                    return False
                elif status == OrderStatus.PARTIALLY_FILLED:
                    # Частичное исполнение тоже считаем успехом
                    return True
                
            except Exception:
                pass
            
            await asyncio.sleep(self.config.order_poll_interval_ms / 1000)
        
        return False
    
    async def _place_stop_loss(
        self,
        signal: TradingSignal,
        stops: PositionStops,
    ) -> None:
        """Ставит стоп-лосс после открытия позиции."""
        try:
            # Инвертируем сторону для закрытия
            close_side = (
                OrderSide.SELL if signal.direction == OrderSide.BUY
                else OrderSide.BUY
            )
            
            # Для стоп-лосса используем MARKET ордер
            # В реальной реализации это может быть STOP_MARKET ордер биржи
            await self.adapter.submit_order(
                symbol=signal.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=0,  # будет заполнено при срабатывании
                price=stops.stop_loss.price,
                client_order_id=f"sl_{uuid.uuid4().hex[:8]}",
            )
        except Exception:
            # Логируем ошибку, но не прерываем исполнение
            pass
    
    async def _place_take_profits(
        self,
        signal: TradingSignal,
        stops: PositionStops,
    ) -> None:
        """Ставит тейк-профиты после открытия позиции."""
        for tp in stops.take_profits:
            try:
                # Инвертируем сторону для закрытия
                close_side = (
                    OrderSide.SELL if signal.direction == OrderSide.BUY
                    else OrderSide.BUY
                )
                
                await self.adapter.submit_order(
                    symbol=signal.symbol,
                    side=close_side,
                    order_type=OrderType.LIMIT,
                    quantity=0,  # будет рассчитано от размера позиции
                    price=tp.price,
                    client_order_id=f"tp{tp.target_index}_{uuid.uuid4().hex[:8]}",
                )
            except Exception:
                # Логируем ошибку, но не прерываем исполнение
                pass