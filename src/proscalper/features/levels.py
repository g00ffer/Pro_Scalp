"""
Детектор уровней поддержки/сопротивления.

Находит значимые уровни на основе:
- Локальных high/low (фракталов)
- Касаний с реакцией цены
- Кластеризации близких уровней
- Объёма в зоне уровня

Уровни используются для:
- Сигналов пробоя (breakout)
- Сигналов ретеста (retest)
- Определения стоп-лоссов
- Фильтрации ложных пробоев
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

from proscalper.market_data.bar_aggregator import Bar


class LevelSide(Enum):
    """Тип уровня."""
    SUPPORT = auto()
    RESISTANCE = auto()


class LevelState(Enum):
    """Состояние уровня."""
    FORMING = auto()       # Уровень формируется (не подтверждён)
    ACTIVE = auto()        # Уровень активен и торгуется
    TESTED = auto()        # Уровень только что протестирован
    BROKEN = auto()        # Уровень пробит
    FAILED_BREAK = auto()  # Ложный пробой (закол)
    RETESTABLE = auto()    # Уровень пробит и ждёт ретеста
    INVALID = auto()       # Уровень устарел или разрушен


@dataclass
class Level:
    """
    Значимый уровень поддержки/сопротивления.
    
    Уровень - это не одна цена, а зона (center ± zone_width/2).
    """
    id: str
    symbol: str
    side: LevelSide
    center: float          # центральная цена зоны
    zone_width: float      # ширина зоны в цене
    
    # Метрики силы
    touches: int = 0
    volume_at_touches: float = 0.0
    created_ts_ns: int = 0
    last_touch_ts_ns: int = 0
    
    # Состояние
    state: LevelState = LevelState.FORMING
    strength: float = 0.0
    
    # Бонусы/штрафы
    round_bonus: float = 0.0
    fakeout_count: int = 0
    
    # История пробоев
    breakout_attempts: int = 0
    successful_breakouts: int = 0
    
    @property
    def zone_lower(self) -> float:
        return self.center - self.zone_width / 2
    
    @property
    def zone_upper(self) -> float:
        return self.center + self.zone_width / 2
    
    def contains_price(self, price: float) -> bool:
        """Проверяет, находится ли цена в зоне уровня."""
        return self.zone_lower <= price <= self.zone_upper
    
    def distance_to(self, price: float) -> float:
        """Расстояние от цены до зоны (0 если внутри)."""
        if self.contains_price(price):
            return 0.0
        return min(
            abs(price - self.zone_lower),
            abs(price - self.zone_upper)
        )
    
    def age_sec(self, now_ns: int) -> float:
        """Возраст уровня в секундах."""
        return (now_ns - self.created_ts_ns) / 1_000_000_000


@dataclass
class LevelDetectorConfig:
    """
    Конфигурация детектора уровней.
    
    Обновление: добавлен параметр timeframe для работы
    с 5-минутными барами вместо 1-секундных.
    """
    # Таймфрейм баров для детекции уровней
    # "1s" — старый режим (шум), "5m" — новый режим (значимые уровни)
    timeframe: str = "5m"
    
    # Параметры фракталов
    left_bars: int = 5              # баров слева от пика
    right_bars: int = 5             # баров справа от пика
    min_prominence: float = 0.001   # минимальная "выпуклость" (0.1%)
    
    # Кластеризация
    cluster_distance_pct: float = 0.002   # 0.2% для кластеризации (5м бары)
    min_touches_for_active: int = 2       # минимум касаний для активации
    
    # Сила уровня
    touch_weight: float = 1.0
    volume_weight: float = 0.5
    age_decay_hours: float = 48.0   # затухание силы за 48 часов
    
    # Ограничения
    max_levels: int = 20            # максимум уровней на символ
    max_touches: int = 50           # максимум касаний (защита от шума)
    
    # Окно для детекции касаний (в барах)
    touch_lookback_bars: int = 10
    
    # Минимальный возраст уровня для активации (в барах)
    min_age_bars_for_active: int = 3


class FractalDetector:
    """
    Детектор фракталов (локальных high/low).
    
    Фрактал high:
    - high[i] >= max(high[i-left : i])
    - high[i] >= max(high[i+1 : i+right+1])
    
    Подтверждается только после появления right баров справа (без lookahead).
    """
    
    def __init__(self, config: LevelDetectorConfig):
        self.config = config
        self._bars: List[Bar] = []
    
    def on_bar(self, bar: Bar) -> tuple[Optional[Bar], Optional[Bar]]:
        """
        Обработка нового бара.
        
        Возвращает (fractal_high, fractal_low) если обнаружены, иначе (None, None).
        """
        self._bars.append(bar)
        
        # Индекс бара, который мы проверяем (right_bars назад)
        pivot_idx = len(self._bars) - self.config.fractal_right_bars - 1
        
        if pivot_idx < self.config.fractal_left_bars:
            return None, None
        
        pivot_bar = self._bars[pivot_idx]
        
        fractal_high = None
        fractal_low = None
        
        # Проверяем fractal high
        if self._is_fractal_high(pivot_idx):
            fractal_high = pivot_bar
        
        # Проверяем fractal low
        if self._is_fractal_low(pivot_idx):
            fractal_low = pivot_bar
        
        return fractal_high, fractal_low
    
    def _is_fractal_high(self, idx: int) -> bool:
        """Проверяет, является ли бар фрактальным high."""
        pivot_high = self._bars[idx].high
        
        # Проверяем left bars
        for i in range(idx - self.config.fractal_left_bars, idx):
            if self._bars[i].high > pivot_high:
                return False
        
        # Проверяем right bars
        for i in range(idx + 1, idx + self.config.fractal_right_bars + 1):
            if i >= len(self._bars):
                return False
            if self._bars[i].high > pivot_high:
                return False
        
        return True
    
    def _is_fractal_low(self, idx: int) -> bool:
        """Проверяет, является ли бар фрактальным low."""
        pivot_low = self._bars[idx].low
        
        # Проверяем left bars
        for i in range(idx - self.config.fractal_left_bars, idx):
            if self._bars[i].low < pivot_low:
                return False
        
        # Проверяем right bars
        for i in range(idx + 1, idx + self.config.fractal_right_bars + 1):
            if i >= len(self._bars):
                return False
            if self._bars[i].low < pivot_low:
                return False
        
        return True
    
    def reset(self) -> None:
        """Сбрасывает историю баров."""
        self._bars.clear()


class LevelDetector:
    """
    Детектор значимых уровней поддержки/сопротивления.
    
    Работает на барах (от BarAggregator) и создаёт/обновляет уровни.
    """
    
    def __init__(
        self,
        symbol: str,
        tick_size: float,
        config: Optional[LevelDetectorConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.tick_size = tick_size
        self.config = config or LevelDetectorConfig()
        
        self._fractal_detector = FractalDetector(self.config)
        self._levels: Dict[str, Level] = {}
        self._level_counter = 0

    def load_from_klines(self, klines: List[List]) -> int:
        """
        Инициализация детектора из исторических свечей.
        
        Вызывается при старте системы для загрузки карты уровней
        из 24-часовой истории.
        
        Args:
            klines: список свечей в формате Binance (из get_klines())
        
        Returns:
            Количество обнаруженных уровней
        """
        # Конвертируем klines в Bar объекты
        bars = []
        for kline in klines:
            bar = Bar(
                ts_ns=kline[0] * 1_000_000,  # ms -> ns
                symbol=self.symbol,
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5]),
                notional=float(kline[7]),  # quote volume
                trade_count=int(kline[8]),
                buy_volume=float(kline[9]) if len(kline) > 9 else 0.0,
                sell_volume=float(kline[5]) - float(kline[9]) if len(kline) > 9 else 0.0,
            )
            bars.append(bar)
        
        # Прогоняем все бары через детектор
        for bar in bars:
            self.on_bar(bar)
        
        # Возвращаем количество найденных уровней
        return len(self.levels) 

    def on_bar(self, bar: Bar) -> List[Level]:
        """
        Обработка нового бара.
        
        Возвращает список новых или обновлённых уровней.
        """
        # ВАЖНО: игнорируем невалидные бары (с нулевой ценой)
        if bar.high <= 0 or bar.low <= 0 or bar.open <= 0 or bar.close <= 0:
            return []
        
        # Проверяем фракталы
        fractal_high, fractal_low = self._fractal_detector.on_bar(bar)
        
        new_levels = []
        
        if fractal_high is not None:
            level = self._create_or_update_level(
                side=LevelSide.RESISTANCE,
                price=fractal_high.high,
                ts_ns=fractal_high.ts_ns,
                volume=fractal_high.notional,
            )
            if level is not None:
                new_levels.append(level)
        
        if fractal_low is not None:
            level = self._create_or_update_level(
                side=LevelSide.SUPPORT,
                price=fractal_low.low,
                ts_ns=fractal_low.ts_ns,
                volume=fractal_low.notional,
            )
            if level is not None:
                new_levels.append(level)
        
        # Проверяем касания для всех активных уровней
        self._update_touches(bar)
        
        # Обновляем силу и состояния уровней
        self._update_level_states(bar.ts_ns)
        
        return new_levels
    
    def _create_or_update_level(
        self,
        side: LevelSide,
        price: float,
        ts_ns: int,
        volume: float,
    ) -> Optional[Level]:
        """
        Создаёт новый уровень или кластеризует с существующим.
        """
        # ВАЖНО: игнорируем нулевые и отрицательные цены
        if price <= 0:
            return None
        
        # Проверяем, есть ли близкий уровень для кластеризации
        existing = self._find_nearby_level(price, side)
        
        if existing is not None:
            # Обновляем существующий уровень (усредняем цену)
            existing.center = (existing.center + price) / 2
            existing.volume_at_touches += volume
            existing.touches += 1
            existing.last_touch_ts_ns = ts_ns
            self._recalculate_strength(existing, ts_ns)
            return existing
        
        # Создаём новый уровень
        zone_width = self._calculate_zone_width(price)
        
        self._level_counter += 1
        level_id = f"{self.symbol}_{side.name}_{self._level_counter}"
        
        level = Level(
            id=level_id,
            symbol=self.symbol,
            side=side,
            center=price,
            zone_width=zone_width,
            touches=1,
            volume_at_touches=volume,
            created_ts_ns=ts_ns,
            last_touch_ts_ns=ts_ns,
            state=LevelState.FORMING,
        )
        
        # Проверяем бонус за круглое число
        if self._is_near_round_number(price):
            level.round_bonus = self.config.round_bonus
        
        self._recalculate_strength(level, ts_ns)
        self._levels[level_id] = level
        
        return level
    
    def _find_nearby_level(
        self,
        price: float,
        side: LevelSide,
    ) -> Optional[Level]:
        """
        Ищет существующий уровень в зоне кластеризации.
        
        Использует увеличенное расстояние для лучшего объединения близких уровней.
        """
        # Увеличиваем расстояние кластеризации для лучшего объединения
        cluster_distance = self.config.cluster_distance_ticks * self.tick_size * 2
        
        for level in self._levels.values():
            if level.side != side:
                continue
            
            # Ищем только среди активных и формирующихся уровней
            if level.state not in (LevelState.FORMING, LevelState.ACTIVE):
                continue
            
            # Проверяем близость цены
            if abs(level.center - price) <= cluster_distance:
                return level
        
        return None
    
    def _calculate_zone_width(self, price: float) -> float:
        """Рассчитывает ширину зоны уровня."""
        width_ticks = self.config.zone_width_ticks * self.tick_size
        width_pct = price * self.config.zone_width_pct
        return max(width_ticks, width_pct)
    
    def _is_near_round_number(self, price: float) -> bool:
        """Проверяет, близка ли цена к круглому числу."""
        # Определяем шаг в зависимости от порядка цены
        if price < 1:
            round_levels = [0.01, 0.05, 0.1]
        elif price < 10:
            round_levels = [0.1, 0.5, 1.0]
        elif price < 100:
            round_levels = [1, 5, 10]
        elif price < 1000:
            round_levels = [10, 50, 100]
        else:
            round_levels = [100, 500, 1000, 5000, 10000]
        
        tolerance = price * 0.001  # 0.1%
        for level in round_levels:
            if abs(price - level) <= tolerance:
                return True
        
        return False
    
    def _update_touches(self, bar: Bar) -> None:
        """Обновляет касания для всех активных уровней."""
        for level in self._levels.values():
            if level.state not in (LevelState.FORMING, LevelState.ACTIVE):
                continue
            
            # Проверяем, касается ли бар зоны уровня
            if not self._bar_touches_zone(bar, level):
                continue
            
            # Пропускаем, если этот бар уже создал/обновил уровень
            # (чтобы не засчитывать касание для только что созданного уровня)
            if level.last_touch_ts_ns == bar.ts_ns:
                continue
            
            # Проверяем, была ли реакция (rejection)
            if self._is_rejection_confirmed(bar, level):
                # Проверяем минимальный интервал между касаниями
                time_since_last_touch_ms = (
                    (bar.ts_ns - level.last_touch_ts_ns) / 1_000_000
                )
                
                if time_since_last_touch_ms >= self.config.min_touch_interval_ms:
                    level.touches += 1
                    level.last_touch_ts_ns = bar.ts_ns
                    level.volume_at_touches += bar.notional
                    
                    if level.touches >= self.config.min_touches_for_active:
                        level.state = LevelState.ACTIVE
                    
                    self._recalculate_strength(level, bar.ts_ns)
    
    def _bar_touches_zone(self, bar: Bar, level: Level) -> bool:
        """Проверяет, касается ли бар зоны уровня."""
        return (
            bar.low <= level.zone_upper and
            bar.high >= level.zone_lower
        )
    
    def _is_rejection_confirmed(self, bar: Bar, level: Level) -> bool:
        """
        Проверяет, была ли реакция цены от уровня.
        
        Для resistance: цена должна уйти вниз минимум на min_reversal_ticks
        Для support: цена должна уйти вверх минимум на min_reversal_ticks
        """
        min_reversal = self.config.min_reversal_ticks * self.tick_size
        
        if level.side == LevelSide.RESISTANCE:
            # Цена должна уйти вниз от уровня
            return bar.low < level.center - min_reversal
        else:
            # Цена должна уйти вверх от уровня
            return bar.high > level.center + min_reversal
    
    def _update_level_states(self, now_ns: int) -> None:
        """Обновляет состояния уровней (устаревание, пробой и т.д.)."""
        for level in self._levels.values():
            age_sec = level.age_sec(now_ns)
            
            # Устаревшие уровни помечаем как INVALID
            if age_sec > self.config.max_age_sec:
                level.state = LevelState.INVALID
                continue
            
            # Обновляем силу (с учётом затухания)
            self._recalculate_strength(level, now_ns)
            
            # Если сила упала ниже порога - INVALID
            if level.state == LevelState.ACTIVE:
                if level.strength < self.config.min_strength_for_active:
                    level.state = LevelState.INVALID
    
    def _recalculate_strength(self, level: Level, now_ns: int) -> None:
        """
        Пересчитывает силу уровня.
        
        Формула:
        strength = w_touches * touches
                 + w_volume * (volume / median)
                 + w_age * age_decay
                 + w_round * round_bonus
                 - w_fakeout * fakeout_count
        """
        age_sec = level.age_sec(now_ns)
        age_half_life_sec = self.config.max_age_sec / 2
        
        # Экспоненциальное затухание с возрастом
        age_decay = 2 ** (-age_sec / age_half_life_sec) if age_half_life_sec > 0 else 0
        
        # Нормализация объёма (примерная медиана)
        median_volume = 50000  # можно заменить на реальную медиану из TapeAnalyzer
        volume_score = (level.volume_at_touches / median_volume) if median_volume > 0 else 0
        
        # Веса (можно вынести в config)
        w_touches = 1.0
        w_volume = 0.8
        w_age = 0.3
        w_round = 1.0
        w_fakeout = 0.5
        
        level.strength = (
            w_touches * level.touches
            + w_volume * volume_score
            + w_age * age_decay
            + w_round * level.round_bonus
            - w_fakeout * level.fakeout_count
        )
    
    def get_active_levels(self) -> List[Level]:
        """Возвращает список активных уровней."""
        return [
            level for level in self._levels.values()
            if level.state == LevelState.ACTIVE
        ]
    
    def get_all_levels(self) -> List[Level]:
        """Возвращает все уровни (включая FORMING)."""
        return list(self._levels.values())
    
    def get_level_by_id(self, level_id: str) -> Optional[Level]:
        """Возвращает уровень по ID."""
        return self._levels.get(level_id)
    
    def mark_level_broken(self, level_id: str) -> None:
        """Помечает уровень как пробитый."""
        level = self._levels.get(level_id)
        if level is not None:
            level.state = LevelState.BROKEN
            level.breakout_attempts += 1
    
    def mark_failed_breakout(self, level_id: str) -> None:
        """Помечает ложный пробой уровня."""
        level = self._levels.get(level_id)
        if level is not None:
            level.state = LevelState.FAILED_BREAK
            level.fakeout_count += 1
            self._recalculate_strength(level, time.time_ns())
    
    def reset(self) -> None:
        """Сбрасывает все уровни."""
        self._levels.clear()
        self._level_counter = 0
        self._fractal_detector.reset()


class LevelManager:
    """
    Менеджер LevelDetector'ов для множества символов.
    """
    
    def __init__(self, tick_sizes: Dict[str, float]):
        self._detectors: Dict[str, LevelDetector] = {}
        self._tick_sizes = tick_sizes
    
    async def initialize_from_history(
        self,
        rest_client,
        symbols: List[str],
        interval: str = "5m",
        hours: float = 24.0,
    ) -> Dict[str, int]:
        """
        Инициализация детекторов уровней из истории через REST.
        
        Вызывается при старте коллекционера для загрузки
        карты уровней из 24-часовой истории.
        
        Args:
            rest_client: BinanceFuturesRestClient
            symbols: список символов для инициализации
            interval: таймфрейм баров (по умолчанию "5m")
            hours: глубина истории в часах (по умолчанию 24)
        
        Returns:
            Словарь {символ: количество уровней}
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Загружаем историю
                klines = await rest_client.get_klines_last_hours(
                    symbol=symbol,
                    interval=interval,
                    hours=hours,
                )
                
                # Получаем или создаём детектор
                detector = self.get_or_create(symbol)
                
                # Загружаем уровни из истории
                level_count = detector.load_from_klines(klines)
                results[symbol] = level_count
                
            except Exception as exc:
                # Ошибка загрузки не должна ронять систему
                results[symbol] = 0
        
        return results

    def get_or_create(
        self,
        symbol: str,
        config: Optional[LevelDetectorConfig] = None,
    ) -> LevelDetector:
        """Возвращает детектор для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._detectors:
            tick_size = self._tick_sizes.get(symbol)
            if tick_size is None:
                raise ValueError(f"Не найден tick_size для {symbol}")
            
            self._detectors[symbol] = LevelDetector(
                symbol=symbol,
                tick_size=tick_size,
                config=config,
            )
        
        return self._detectors[symbol]
    
    def on_bar(self, bar: Bar) -> List[Level]:
        """Обработка бара для нужного символа."""
        detector = self._detectors.get(bar.symbol.upper())
        if detector is None:
            return []
        return detector.on_bar(bar)
    
    def get_active_levels(self, symbol: str) -> List[Level]:
        """Возвращает активные уровни для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return []
        return detector.get_active_levels()
    
    def get_all_levels(self, symbol: str) -> List[Level]:
        """Возвращает все уровни для символа."""
        detector = self._detectors.get(symbol.upper())
        if detector is None:
            return []
        return detector.get_all_levels()
    
    def reset_all(self) -> None:
        """Сбрасывает все детекторы."""
        for detector in self._detectors.values():
            detector.reset()
    
    def all_symbols(self) -> List[str]:
        return list(self._detectors.keys())