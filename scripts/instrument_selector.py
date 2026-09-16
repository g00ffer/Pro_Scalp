#!/usr/bin/env python3
"""
Скрипт отбора пула инструментов для торговли.

Запускается ежедневно через cron. Просматривает все инструменты биржи
и отбирает пул, удовлетворяющий условиям:
- Ликвидность (глубина стакана достаточна для позиции)
- Наличие подтверждённых уровней
- Корреляция с уже отобранными < 0.7

Результат сохраняется в configs/instruments.json.

Использование:
    python scripts/instrument_selector.py --output configs/instruments.json
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class SelectionConfig:
    """Конфигурация отбора инструментов."""
    
    # Ликвидность
    min_volume_24h: float = 10_000_000      # минимальный объём за 24ч ($10M)
    min_depth_multiplier: float = 10.0      # глубина стакана ≥ 10× позиции
    position_size: float = 100.0            # типичный размер позиции ($)
    depth_zone_ticks: int = 10              # зона проверки глубины (±10 тиков)
    
    # Уровни
    min_level_strength: float = 3.0         # минимальная сила уровня
    max_level_age_hours: float = 4.0        # максимальный возраст уровня (часы)
    
    # Корреляция
    max_correlation: float = 0.7            # максимальная корреляция между инструментами
    
    # Волатильность (адаптивная)
    volatility_window_days: int = 30        # окно для расчёта средней волатильности
    volatility_low_mult: float = 0.5        # нижняя граница = средняя × 0.5
    volatility_high_mult: float = 2.0       # верхняя граница = средняя × 2.0
    
    # Ограничения
    max_instruments_in_pool: int = 100      # максимум инструментов в пуле
    request_timeout_sec: float = 30.0       # таймаут запросов
    requests_per_second: int = 10           # лимит запросов в секунду


@dataclass
class InstrumentCandidate:
    """Кандидат на отбор."""
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: float
    step_size: float
    
    # Метрики
    volume_24h: float = 0.0
    depth_score: float = 0.0
    volatility: float = 0.0
    levels_score: float = 0.0
    correlation_score: float = 0.0
    
    # Итоговый скор
    total_score: float = 0.0
    
    # Причины отклонения
    rejection_reasons: List[str] = field(default_factory=list)
    
    @property
    def is_selected(self) -> bool:
        """Инструмент прошёл отбор."""
        return len(self.rejection_reasons) == 0


class InstrumentSelector:
    """
    Основной класс отбора инструментов.
    
    Логика работы:
    1. Загружаем все инструменты биржи
    2. Фильтруем по статусу (только TRADING)
    3. Фильтруем по объёму за 24ч
    4. Для оставшихся проверяем глубину стакана
    5. Для оставшихся детектируем уровни
    6. Применяем фильтр корреляции
    7. Сохраняем результат
    """
    
    def __init__(
        self,
        base_url: str = "https://fapi1.binance.com",
        config: Optional[SelectionConfig] = None,
    ):
        self.base_url = base_url
        self.config = config or SelectionConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._request_timestamps: List[float] = []
    
    async def run(self) -> List[InstrumentCandidate]:
        """
        Запускает отбор инструментов.
        
        Возвращает список отобранных кандидатов.
        """
        async with httpx.AsyncClient(timeout=self.config.request_timeout_sec) as client:
            self._client = client
            
            try:
                # Шаг 1: Загружаем все инструменты
                print("📦 Загрузка инструментов биржи...")
                all_instruments = await self._load_exchange_info()
                print(f"  Загружено инструментов: {len(all_instruments)}")
                
                # Шаг 2: Фильтруем по статусу
                trading_instruments = [
                    inst for inst in all_instruments
                    if inst.get("status") == "TRADING"
                ]
                print(f"  После фильтра по статусу: {len(trading_instruments)}")
                
                # Шаг 3: Загружаем статистику за 24ч
                print("📊 Загрузка статистики за 24ч...")
                ticker_data = await self._load_24h_tickers()
                
                # Шаг 4: Фильтруем по объёму
                candidates = []
                for inst in trading_instruments:
                    symbol = inst.get("symbol", "")
                    ticker = ticker_data.get(symbol, {})
                    volume_24h = float(ticker.get("quoteVolume", 0))
                    
                    if volume_24h >= self.config.min_volume_24h:
                        candidate = self._create_candidate(inst, volume_24h)
                        candidates.append(candidate)
                
                print(f"  После фильтра по объёму: {len(candidates)}")
                
                # Шаг 5: Проверяем глубину стакана
                print("🔍 Проверка глубины стакана...")
                candidates = await self._check_depth(candidates)
                print(f"  После проверки глубины: {len([c for c in candidates if c.is_selected])}")
                
                # Шаг 6: Детектируем уровни
                print("📈 Детекция уровней...")
                candidates = await self._check_levels(candidates)
                print(f"  После детекции уровней: {len([c for c in candidates if c.is_selected])}")
                
                # Шаг 7: Применяем фильтр корреляции
                print("🔗 Применение фильтра корреляции...")
                candidates = self._apply_correlation_filter(candidates)
                print(f"  После фильтра корреляции: {len([c for c in candidates if c.is_selected])}")
                
                # Шаг 8: Сортируем по скору и ограничиваем количество
                selected = [c for c in candidates if c.is_selected]
                selected.sort(key=lambda c: c.total_score, reverse=True)
                
                if len(selected) > self.config.max_instruments_in_pool:
                    selected = selected[:self.config.max_instruments_in_pool]
                
                print(f"\n✅ Отобрано инструментов: {len(selected)}")
                
                return selected
            
            finally:
                self._client = None
    
    def save_results(
        self,
        candidates: List[InstrumentCandidate],
        output_path: str,
    ) -> None:
        """Сохраняет результаты отбора в JSON файл."""
        selected = [c for c in candidates if c.is_selected]
        
        result = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_scanned": len(candidates),
            "total_selected": len(selected),
            "config": {
                "min_volume_24h": self.config.min_volume_24h,
                "min_depth_multiplier": self.config.min_depth_multiplier,
                "min_level_strength": self.config.min_level_strength,
                "max_correlation": self.config.max_correlation,
            },
            "instruments": [
                {
                    "symbol": c.symbol,
                    "base_asset": c.base_asset,
                    "quote_asset": c.quote_asset,
                    "tick_size": c.tick_size,
                    "step_size": c.step_size,
                    "volume_24h": c.volume_24h,
                    "depth_score": c.depth_score,
                    "volatility": c.volatility,
                    "levels_score": c.levels_score,
                    "total_score": c.total_score,
                }
                for c in selected
            ],
        }
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Результаты сохранены в: {output_path}")
    
    async def _load_exchange_info(self) -> List[Dict[str, Any]]:
        """Загружает информацию обо всех инструментах."""
        await self._rate_limit()
        
        response = await self._client.get(f"{self.base_url}/fapi/v1/exchangeInfo")
        response.raise_for_status()
        
        data = response.json()
        return data.get("symbols", [])
    
    async def _load_24h_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Загружает статистику за 24 часа для всех инструментов."""
        await self._rate_limit()
        
        response = await self._client.get(f"{self.base_url}/fapi/v1/ticker/24hr")
        response.raise_for_status()
        
        data = response.json()
        
        # Индексируем по символу
        ticker_map = {}
        for ticker in data:
            symbol = ticker.get("symbol", "")
            if symbol:
                ticker_map[symbol] = ticker
        
        return ticker_map
    
    async def _check_depth(
        self,
        candidates: List[InstrumentCandidate],
    ) -> List[InstrumentCandidate]:
        """Проверяет глубину стакана для кандидатов."""
        for candidate in candidates:
            try:
                depth_ok = await self._check_single_depth(candidate)
                
                if not depth_ok:
                    candidate.rejection_reasons.append(
                        f"Недостаточная глубина стакана"
                    )
                
            except Exception as e:
                candidate.rejection_reasons.append(
                    f"Ошибка проверки глубины: {str(e)}"
                )
        
        return candidates
    
    async def _check_single_depth(self, candidate: InstrumentCandidate) -> bool:
        """Проверяет глубину стакана для одного инструмента."""
        await self._rate_limit()
        
        try:
            response = await self._client.get(
                f"{self.base_url}/fapi/v1/depth",
                params={"symbol": candidate.symbol, "limit": 20},
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Рассчитываем глубину в зоне ±10 тиков от лучшей цены
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            if not bids or not asks:
                return False
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            
            zone_distance = self.config.depth_zone_ticks * candidate.tick_size
            
            # Считаем ликвидность в зоне
            bid_liquidity = 0.0
            for price_str, qty_str in bids:
                price = float(price_str)
                if mid_price - price <= zone_distance:
                    bid_liquidity += price * float(qty_str)
            
            ask_liquidity = 0.0
            for price_str, qty_str in asks:
                price = float(price_str)
                if price - mid_price <= zone_distance:
                    ask_liquidity += price * float(qty_str)
            
            total_liquidity = bid_liquidity + ask_liquidity
            
            # Проверяем, что ликвидность достаточна
            min_required = self.config.position_size * self.config.min_depth_multiplier
            
            # Рассчитываем скор глубины
            if min_required > 0:
                candidate.depth_score = min(1.0, total_liquidity / min_required)
            
            return total_liquidity >= min_required
        
        except Exception:
            return False
    
    async def _check_levels(
        self,
        candidates: List[InstrumentCandidate],
    ) -> List[InstrumentCandidate]:
        """
        Проверяет наличие уровней для кандидатов.
        
        Для упрощения первой версии используем упрощённую детекцию:
        - Загружаем историю за 24 часа (5-минутные бары)
        - Ищем локальные экстремумы
        - Оцениваем их силу
        """
        for candidate in candidates:
            try:
                levels_ok = await self._check_single_levels(candidate)
                
                if not levels_ok:
                    candidate.rejection_reasons.append(
                        f"Нет подтверждённых уровней"
                    )
                
            except Exception as e:
                candidate.rejection_reasons.append(
                    f"Ошибка детекции уровней: {str(e)}"
                )
        
        return candidates
    
    async def _check_single_levels(self, candidate: InstrumentCandidate) -> bool:
        """Проверяет наличие уровней для одного инструмента."""
        await self._rate_limit()
        
        try:
            # Загружаем 5-минутные бары за 24 часа
            response = await self._client.get(
                f"{self.base_url}/fapi/v1/klines",
                params={
                    "symbol": candidate.symbol,
                    "interval": "5m",
                    "limit": 288,
                },
            )
            response.raise_for_status()
            
            klines = response.json()
            
            if len(klines) < 10:
                return False
            
            # Упрощённая детекция уровней: ищем локальные экстремумы
            levels_found = self._detect_simple_levels(klines, candidate)
            
            # Рассчитываем скор уровней
            if levels_found > 0:
                candidate.levels_score = min(1.0, levels_found / 5.0)
            
            return levels_found > 0
        
        except Exception:
            return False
    
    def _detect_simple_levels(
        self,
        klines: List[List[Any]],
        candidate: InstrumentCandidate,
    ) -> int:
        """
        Упрощённая детекция уровней.
        
        Ищем локальные экстремумы с подтверждением.
        """
        if len(klines) < 5:
            return 0
        
        # Извлекаем high и low
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        
        levels_found = 0
        fractal_size = 2  # 2 бара слева и справа
        
        # Ищем фрактальные максимумы (сопротивления)
        for i in range(fractal_size, len(highs) - fractal_size):
            is_high = True
            
            # Проверяем левые бары
            for j in range(i - fractal_size, i):
                if highs[j] > highs[i]:
                    is_high = False
                    break
            
            if not is_high:
                continue
            
            # Проверяем правые бары
            for j in range(i + 1, i + fractal_size + 1):
                if highs[j] > highs[i]:
                    is_high = False
                    break
            
            if is_high:
                levels_found += 1
        
        # Ищем фрактальные минимумы (поддержки)
        for i in range(fractal_size, len(lows) - fractal_size):
            is_low = True
            
            # Проверяем левые бары
            for j in range(i - fractal_size, i):
                if lows[j] < lows[i]:
                    is_low = False
                    break
            
            if not is_low:
                continue
            
            # Проверяем правые бары
            for j in range(i + 1, i + fractal_size + 1):
                if lows[j] < lows[i]:
                    is_low = False
                    break
            
            if is_low:
                levels_found += 1
        
        return levels_found
    
    def _apply_correlation_filter(
        self,
        candidates: List[InstrumentCandidate],
    ) -> List[InstrumentCandidate]:
        """
        Применяет фильтр корреляции.
        
        Для упрощения первой версии используем жадный алгоритм:
        1. Сортируем по скору
        2. Берём лучший
        3. Пропускаем коррелированные с уже отобранными
        
        В первой версии корреляция не рассчитывается (требует истории цен).
        Позже добавим расчёт корреляции по историческим данным.
        """
        # В первой версии пропускаем фильтр корреляции
        # Он будет добавлен после реализации расчёта корреляции
        
        # Для сейчас просто возвращаем кандидатов без изменений
        return candidates
    
    def _create_candidate(
        self,
        instrument_info: Dict[str, Any],
        volume_24h: float,
    ) -> InstrumentCandidate:
        """Создаёт кандидата из информации об инструменте."""
        # Извлекаем параметры инструмента
        tick_size = 0.0
        step_size = 0.0
        
        for filter_info in instrument_info.get("filters", []):
            filter_type = filter_info.get("filterType", "")
            
            if filter_type == "PRICE_FILTER":
                tick_size = float(filter_info.get("tickSize", 0))
            elif filter_type == "LOT_SIZE":
                step_size = float(filter_info.get("stepSize", 0))
        
        return InstrumentCandidate(
            symbol=instrument_info.get("symbol", ""),
            base_asset=instrument_info.get("baseAsset", ""),
            quote_asset=instrument_info.get("quoteAsset", ""),
            tick_size=tick_size,
            step_size=step_size,
            volume_24h=volume_24h,
        )
    
    async def _rate_limit(self) -> None:
        """Ограничивает частоту запросов."""
        now = time.time()
        
        # Удаляем старые временные метки
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if now - ts < 1.0
        ]
        
        # Если превысили лимит - ждём
        if len(self._request_timestamps) >= self.config.requests_per_second:
            sleep_time = 1.0 - (now - self._request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self._request_timestamps.append(now)


async def main() -> None:
    """Точка входа."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Отбор пула инструментов")
    parser.add_argument(
        "--output",
        default="configs/instruments.json",
        help="Путь к выходному файлу",
    )
    parser.add_argument(
        "--max-pool-size",
        type=int,
        default=100,
        help="Максимальный размер пула",
    )
    
    args = parser.parse_args()
    
    config = SelectionConfig(
        max_instruments_in_pool=args.max_pool_size,
    )
    
    selector = InstrumentSelector(config=config)
    
    print("=" * 60)
    print("🚀 Запуск отбора пула инструментов")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    candidates = await selector.run()
    
    selector.save_results(candidates, args.output)
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 60)
    print(f"⏱️  Время выполнения: {elapsed:.1f} сек")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())