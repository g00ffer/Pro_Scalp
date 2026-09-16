"""
Главный цикл торгового робота.

Объединяет все модули в единую систему:
1. Загрузка истории и построение карты уровней
2. Отбор инструментов из пула
3. Сканер готовности к пробою
4. Генерация сигналов
5. Риск-менеджмент
6. Исполнение сделок
7. Управление позициями

Запуск:
    python -m proscalper.app.main

Или:
    python src/proscalper/app/main.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_bot.log"),
    ],
)
logger = logging.getLogger("proscalper.main")


class TradingBotConfig:
    """Конфигурация торгового робота."""
    
    def __init__(
        self,
        instruments_file: str = "configs/instruments.json",
        initial_deposit: float = 1000.0,
        scan_interval_ms: int = 500,
        regime_check_interval_sec: int = 60,
    ):
        self.instruments_file = instruments_file
        self.initial_deposit = initial_deposit
        self.scan_interval_ms = scan_interval_ms
        self.regime_check_interval_sec = regime_check_interval_sec


class TradingBot:
    """
    Главный класс торгового робота.
    
    Объединяет все модули и управляет жизненным циклом торговли.
    
    Использование:
        bot = TradingBot(config)
        await bot.run()
    """
    
    def __init__(self, config: Optional[TradingBotConfig] = None):
        self.config = config or TradingBotConfig()
        
        # Флаг работы
        self._running = False
        
        # Пул инструментов
        self._instrument_pool: List[Dict] = []
        
        # Компоненты (инициализируются в setup())
        self._level_manager = None
        self._compression_manager = None
        self._book_analyzer_manager = None
        self._approach_manager = None
        self._impulse_manager = None
        self._breakout_manager = None
        self._signal_generator_manager = None
        self._risk_manager = None
        self._execution_engine = None
        self._position_manager = None
        self._stops_manager = None
        self._market_regime_manager = None
        self._exhaustion_manager = None
    
    async def setup(self) -> None:
        """
        Инициализация всех компонентов.
        
        Вызывается один раз перед запуском.
        """
        logger.info("🚀 Инициализация торгового робота...")
        
        # Загружаем пул инструментов
        self._load_instrument_pool()
        
        # Инициализируем компоненты
        # В реальной реализации здесь создаются все менеджеры
        
        logger.info(f"✅ Инициализация завершена. Пул инструментов: {len(self._instrument_pool)}")
    
    async def run(self) -> None:
        """
        Главный цикл торговли.
        
        Работает до вызова stop() или получения сигнала остановки.
        """
        await self.setup()
        
        self._running = True
        logger.info("🟢 Торговый робот запущен")
        
        # Регистрируем обработчики сигналов
        self._setup_signal_handlers()
        
        # Запускаем параллельные задачи
        tasks = [
            asyncio.create_task(self._scan_loop()),
            asyncio.create_task(self._regime_check_loop()),
            asyncio.create_task(self._position_management_loop()),
        ]
        
        try:
            # Ждём завершения
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Завершение работы робота."""
        logger.info("🔴 Остановка торгового робота...")
        
        self._running = False
        
        # Закрываем все позиции
        await self._close_all_positions()
        
        # Сохраняем статистику
        self._save_stats()
        
        logger.info("✅ Торговый робот остановлен")
    
    def stop(self) -> None:
        """Останавливает робота."""
        self._running = False
    
    def _load_instrument_pool(self) -> None:
        """Загружает пул инструментов из файла."""
        instruments_path = Path(self.config.instruments_file)
        
        if not instruments_path.exists():
            logger.warning(
                f"Файл пула инструментов не найден: {instruments_path}. "
                f"Запустите скрипт отбора: python scripts/instrument_selector.py"
            )
            self._instrument_pool = []
            return
        
        try:
            with open(instruments_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._instrument_pool = data.get("instruments", [])
            logger.info(f"Загружено инструментов из пула: {len(self._instrument_pool)}")
            
            # Логируем топ-5 инструментов
            for inst in self._instrument_pool[:5]:
                symbol = inst.get("symbol", "")
                score = inst.get("total_score", 0)
                logger.info(f"  {symbol}: score={score:.3f}")
        
        except Exception as e:
            logger.error(f"Ошибка загрузки пула инструментов: {e}")
            self._instrument_pool = []
    
    def _setup_signal_handlers(self) -> None:
        """Регистрирует обработчики сигналов ОС."""
        import signal
        
        def handle_stop(signum, frame):
            logger.info(f"Получен сигнал остановки: {signum}")
            self.stop()
        
        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)
    
    async def _scan_loop(self) -> None:
        """
        Цикл сканирования инструментов на готовность к пробою.
        
        Это основной цикл торговли. Работает каждые 500мс.
        """
        logger.info("📡 Запуск цикла сканирования...")
        
        while self._running:
            try:
                await self._scan_instruments()
            except Exception as e:
                logger.error(f"Ошибка в цикле сканирования: {e}")
            
            await asyncio.sleep(self.config.scan_interval_ms / 1000)
    
    async def _scan_instruments(self) -> None:
        """
        Сканирует инструменты на готовность к пробою.
        
        Логика:
        1. Для каждого инструмента из пула проверяем готовность
        2. Если готовность высокая → рассчитываем параметры входа
        3. Если параметры валидны → генерируем сигнал
        4. Передаём сигнал в RiskManager
        """
        for instrument in self._instrument_pool:
            symbol = instrument.get("symbol", "")
            
            if not symbol:
                continue
            
            # Проверяем режим рынка
            # В реальной реализации здесь вызывается market_regime_manager
            
            # Проверяем готовность к пробою
            # В реальной реализации здесь вызывается approach_manager
            
            # Если готовность высокая → генерируем сигнал
            # В реальной реализации здесь вызывается signal_generator
            
            # Передаём сигнал в RiskManager
            # В реальной реализации здесь вызывается risk_manager
            
            pass  # Заглушка для первой версии
    
    async def _regime_check_loop(self) -> None:
        """
        Цикл проверки режима рынка.
        
        Работает каждые 60 секунд.
        """
        logger.info("📊 Запуск цикла проверки режима рынка...")
        
        while self._running:
            try:
                await self._check_market_regime()
            except Exception as e:
                logger.error(f"Ошибка в цикле проверки режима: {e}")
            
            await asyncio.sleep(self.config.regime_check_interval_sec)
    
    async def _check_market_regime(self) -> None:
        """
        Проверяет режим рынка для всех инструментов.
        
        Если режим не подходит для торговли — пропускаем инструмент.
        """
        for instrument in self._instrument_pool:
            symbol = instrument.get("symbol", "")
            
            if not symbol:
                continue
            
            # В реальной реализации здесь вызывается market_regime_manager
            pass  # Заглушка для первой версии
    
    async def _position_management_loop(self) -> None:
        """
        Цикл управления позициями.
        
        Работает каждые 100мс. Проверяет стопы, тейки, трейлинг.
        """
        logger.info("📋 Запуск цикла управления позициями...")
        
        while self._running:
            try:
                await self._manage_positions()
            except Exception as e:
                logger.error(f"Ошибка в цикле управления позициями: {e}")
            
            await asyncio.sleep(0.1)  # 100мс
    
    async def _manage_positions(self) -> None:
        """
        Управляет открытыми позициями.
        
        Проверяет:
        - Стоп-лоссы
        - Тейк-профиты
        - Трейлинг-стопы
        - Исчерпание импульса
        """
        # В реальной реализации здесь вызываются:
        # - position_manager.check_actions()
        # - stops_manager.on_price_update()
        # - exhaustion_manager.update()
        pass  # Заглушка для первой версии
    
    async def _close_all_positions(self) -> None:
        """Закрывает все открытые позиции."""
        logger.info("Закрытие всех позиций...")
        
        # В реальной реализации здесь вызывается:
        # - position_manager.close_all()
        # - execution_engine.cancel_all_orders()
        pass  # Заглушка для первой версии
    
    def _save_stats(self) -> None:
        """Сохраняет статистику торговли."""
        stats = {
            "shutdown_ts": time.time(),
            "instrument_pool_size": len(self._instrument_pool),
        }
        
        # В реальной реализации здесь собирается полная статистика из всех менеджеров
        
        stats_path = Path("logs/trading_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Статистика сохранена в: {stats_path}")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику робота."""
        return {
            "running": self._running,
            "instrument_pool_size": len(self._instrument_pool),
        }


async def main() -> None:
    """Точка входа."""
    # Создаём конфиг
    config = TradingBotConfig(
        instruments_file="configs/instruments.json",
        initial_deposit=1000.0,
        scan_interval_ms=500,
        regime_check_interval_sec=60,
    )
    
    # Создаём и запускаем робота
    bot = TradingBot(config)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Получен KeyboardInterrupt, останавливаем...")
        bot.stop()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        bot.stop()
        raise


if __name__ == "__main__":
    # Создаём директорию для логов
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Запускаем
    asyncio.run(main())