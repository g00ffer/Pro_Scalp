"""
Быстрая работа с ценами через integer ticks.
В hot-path избегаем Decimal, используем int для сравнений и bucket-структур.
"""
from typing import Optional
import math

from proscalper.core.types import PriceTick, InstrumentInfo


class PriceConverter:
    """Конвертер между float ценами и integer тиками."""
    
    def __init__(self, tick_size: float, precision: int = 8):
        self.tick_size = tick_size
        self.precision = precision
        self._inv_tick_size = 1.0 / tick_size
    
    def price_to_ticks(self, price: float) -> PriceTick:
        """Конвертирует цену в количество тиков (integer)."""
        return PriceTick(round(price * self._inv_tick_size))
    
    def ticks_to_price(self, ticks: PriceTick) -> float:
        """Конвертирует тики обратно в цену."""
        return round(ticks * self.tick_size, self.precision)
    
    def normalize_price(self, price: float) -> float:
        """Нормализует цену до валидного тика."""
        ticks = self.price_to_ticks(price)
        return self.ticks_to_price(ticks)
    
    def ticks_between(self, price1: float, price2: float) -> int:
        """Количество тиков между двумя ценами."""
        return abs(self.price_to_ticks(price1) - self.price_to_ticks(price2))
    
    def add_ticks(self, price: float, ticks: int) -> float:
        """Добавляет тики к цене."""
        base_ticks = self.price_to_ticks(price)
        return self.ticks_to_price(PriceTick(base_ticks + ticks))
    
    def subtract_ticks(self, price: float, ticks: int) -> float:
        """Вычитает тики из цены."""
        base_ticks = self.price_to_ticks(price)
        return self.ticks_to_price(PriceTick(base_ticks - ticks))


class QuantityConverter:
    """Конвертер объёмов."""
    
    def __init__(self, step_size: float, precision: int = 8):
        self.step_size = step_size
        self.precision = precision
        self._inv_step_size = 1.0 / step_size
    
    def normalize_quantity(self, qty: float) -> float:
        """Нормализует объём до валидного шага."""
        return round(math.floor(qty * self._inv_step_size) * self.step_size, self.precision)
    
    def calculate_quantity_from_notional(
        self,
        notional: float,
        price: float,
        min_notional: float
    ) -> float:
        """Рассчитывает объём из номинала."""
        if price <= 0:
            return 0.0
        raw_qty = notional / price
        return self.normalize_quantity(raw_qty)


class SlippageCalculator:
    """Калькулятор проскальзывания."""
    
    def __init__(self, tick_size: float):
        self.tick_size = tick_size
    
    def expected_slippage_ticks(
        self,
        order_qty: float,
        book_depth: list[tuple[float, float]],  # [(price, qty), ...]
        is_buy: bool
    ) -> int:
        """Оценивает проскальзывание в тиках на основе глубины стакана."""
        remaining = order_qty
        worst_price = None
        best_price = book_depth[0][0] if book_depth else 0.0
        
        for price, qty in book_depth:
            fill_qty = min(remaining, qty)
            remaining -= fill_qty
            worst_price = price
            
            if remaining <= 0:
                break
        
        if worst_price is None or best_price == 0:
            return 0
        
        slippage = abs(worst_price - best_price)
        return max(0, int(slippage / self.tick_size))
    
    def max_acceptable_price(
        self,
        best_price: float,
        max_slippage_ticks: int,
        is_buy: bool
    ) -> float:
        """Максимальная приемлемая цена с учётом проскальзывания."""
        slippage_amount = max_slippage_ticks * self.tick_size
        
        if is_buy:
            return best_price + slippage_amount
        else:
            return best_price - slippage_amount


# === Утилиты ===

def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Безопасное деление с обработкой нуля."""
    if denominator == 0 or math.isnan(denominator) or math.isinf(denominator):
        return default
    result = numerator / denominator
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Ограничивает значение диапазоном."""
    return max(min_val, min(max_val, value))


def zscore(value: float, mean: float, std: float) -> float:
    """Z-score нормализация."""
    if std == 0 or math.isnan(std):
        return 0.0
    return (value - mean) / std


def is_near_round_number(price: float, round_levels: list[float]) -> bool:
    """Проверяет, близка ли цена к круглому числу."""
    tolerance = price * 0.001  # 0.1%
    for level in round_levels:
        if abs(price - level) <= tolerance:
            return True
    return False


def generate_round_levels(min_price: float, max_price: float) -> list[float]:
    """Генерирует список круглых уровней в диапазоне."""
    levels = []
    
    # Определяем шаг в зависимости от порядка цены
    if max_price < 1:
        steps = [0.01, 0.05, 0.1]
    elif max_price < 10:
        steps = [0.1, 0.5, 1.0]
    elif max_price < 100:
        steps = [1, 5, 10]
    elif max_price < 1000:
        steps = [10, 50, 100]
    else:
        steps = [100, 500, 1000]
    
    for step in steps:
        start = math.ceil(min_price / step) * step
        level = start
        while level <= max_price:
            levels.append(level)
            level += step
    
    return sorted(set(levels))