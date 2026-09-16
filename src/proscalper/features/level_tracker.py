"""
Трекер уровней поддержки/сопротивления.

Отвечает за жизненный цикл уровней:
- Управление состояниями (FORMING → ACTIVE → TESTED → BROKEN → RETESTABLE)
- Cooldown после ложных пробоев
- История пробоев (успешные/неуспешные)
- Уведомления о смене состояния

Разделение ответственности:
- levels.py — детекция уровней (фракталы, касания)
- level_tracker.py — жизненный цикл и состояния

Используется в связке с:
- LevelDetector (создание уровней)
- BreakoutDetector (пробои)
- SignalGenerator (генерация сигналов)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from proscalper.features.levels import (
    Level,
    LevelSide,
    LevelState,
    LevelDetector,
    LevelDetectorConfig,
)


class LevelEventType(Enum):
    """Типы событий уровня."""
    CREATED = auto()            # Уровень создан
    ACTIVATED = auto()          # Уровень стал активным
    TESTED = auto()             # Уровень протестирован
    BROKEN = auto()             # Уровень пробит
    FAILED_BREAK = auto()       # Ложный пробой
    RETESTABLE = auto()         # Уровень доступен для ретеста
    RETESTED = auto()           # Уровень ретестнут
    EXPIRED = auto()            # Уровень устарел
    INVALID = auto()            # Уровень разрушен


@dataclass
class LevelEvent:
    """Событие уровня."""
    event_type: LevelEventType
    level_id: str
    symbol: str
    ts_ns: int
    details: Dict = field(default_factory=dict)


@dataclass
class LevelTrackerConfig:
    """Конфигурация трекера уровней."""
    
    # Время перехода в RETESTABLE после пробоя
    retestable_delay_sec: float = 30.0
    
    # Время жизни уровня после пробоя
    broken_lifetime_sec: float = 3600.0  # 1 час
    
    # Время жизни уровня после ложного пробоя
    failed_break_cooldown_sec: float = 300.0  # 5 минут
    
    # Максимальное количество ложных пробоев
    max_fakeout_count: int = 3
    
    # Максимальное время жизни уровня
    max_level_lifetime_sec: float = 86400.0  # 24 часа
    
    # Минимальное время между событиями
    min_event_interval_sec: float = 1.0


class LevelTracker:
    """
    Трекер уровней для одного символа.
    
    Управляет жизненным циклом уровней:
    1. Созданные уровни сначала имеют состояние FORMING
    2. После набора касаний переходят в ACTIVE
    3. При пробое переходят в BROKEN
    4. Через задержку становятся RETESTABLE
    5. После ложного пробоя получают cooldown
    
    Использование:
        tracker = LevelTracker("BTCUSDT", config=LevelTrackerConfig())
        
        # Регистрируем обработчик событий
        tracker.on_level_event(my_handler)
        
        # При пробое уровня
        tracker.on_breakout(level_id, is_confirmed=True)
        
        # При ложном пробое
        tracker.on_failed_breakout(level_id)
        
        # Периодически обновляем состояния
        tracker.update_states()
    """
    
    def __init__(
        self,
        symbol: str,
        config: Optional[LevelTrackerConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.config = config or LevelTrackerConfig()
        
        # Отслеживаемые уровни
        self._tracked_levels: Dict[str, Level] = {}
        
        # Время пробоя каждого уровня
        self._breakout_ts: Dict[str, int] = {}
        
        # Время последнего события для каждого уровня
        self._last_event_ts: Dict[str, int] = {}
        
        # Обработчики событий
        self._event_handlers: List[Callable[[LevelEvent], None]] = []
        
        # Статистика
        self._total_events: int = 0
        self._breakout_count: int = 0
        self._failed_break_count: int = 0
    
    def track_level(self, level: Level) -> None:
        """Начинает отслеживание уровня."""
        self._tracked_levels[level.id] = level
        self._last_event_ts[level.id] = time.time_ns()
    
    def untrack_level(self, level_id: str) -> None:
        """Прекращает отслеживание уровня."""
        self._tracked_levels.pop(level_id, None)
        self._breakout_ts.pop(level_id, None)
        self._last_event_ts.pop(level_id, None)
    
    def get_level(self, level_id: str) -> Optional[Level]:
        """Возвращает отслеживаемый уровень."""
        return self._tracked_levels.get(level_id)
    
    def get_active_levels(self) -> List[Level]:
        """Возвращает все активные уровни."""
        return [
            level for level in self._tracked_levels.values()
            if level.state == LevelState.ACTIVE
        ]
    
    def get_retestable_levels(self) -> List[Level]:
        """Возвращает все уровни, доступные для ретеста."""
        return [
            level for level in self._tracked_levels.values()
            if level.state == LevelState.RETESTABLE
        ]
    
    def get_all_levels(self) -> List[Level]:
        """Возвращает все отслеживаемые уровни."""
        return list(self._tracked_levels.values())
    
    def on_level_event(
        self,
        handler: Callable[[LevelEvent], None],
    ) -> None:
        """Регистрирует обработчик событий уровней."""
        self._event_handlers.append(handler)
    
    def on_breakout(
        self,
        level_id: str,
        is_confirmed: bool = True,
        strength: float = 0.0,
    ) -> None:
        """
        Обработка пробоя уровня.
        
        Вызывается из BreakoutDetector при подтверждении пробоя.
        
        Если пробой сильный → уровень сразу BROKEN
        Если пробой средний → уровень BROKEN, потом RETESTABLE
        Если пробой слабый → ложный пробой (вызвать on_failed_breakout)
        """
        level = self._tracked_levels.get(level_id)
        if level is None:
            return
        
        now_ns = time.time_ns()
        
        # Проверяем минимальный интервал между событиями
        if not self._can_emit_event(level_id, now_ns):
            return
        
        if not is_confirmed:
            self.on_failed_breakout(level_id)
            return
        
        # Переводим в BROKEN
        level.state = LevelState.BROKEN
        level.breakout_attempts += 1
        level.successful_breakouts += 1
        self._breakout_ts[level_id] = now_ns
        self._breakout_count += 1
        
        # Уведомляем о пробое
        self._emit_event(
            LevelEventType.BROKEN,
            level_id,
            now_ns,
            {"strength": strength},
        )
    
    def on_failed_breakout(self, level_id: str) -> None:
        """
        Обработка ложного пробоя уровня.
        
        Вызывается из BreakoutDetector при отклонении пробоя.
        """
        level = self._tracked_levels.get(level_id)
        if level is None:
            return
        
        now_ns = time.time_ns()
        
        # Проверяем минимальный интервал между событиями
        if not self._can_emit_event(level_id, now_ns):
            return
        
        # Переводим в FAILED_BREAK
        level.state = LevelState.FAILED_BREAK
        level.fakeout_count += 1
        self._failed_break_count += 1
        
        # Если слишком много ложных пробоев — уровень невалиден
        if level.fakeout_count >= self.config.max_fakeout_count:
            level.state = LevelState.INVALID
        
        # Уведомляем о ложном пробое
        self._emit_event(
            LevelEventType.FAILED_BREAK,
            level_id,
            now_ns,
            {"fakeout_count": level.fakeout_count},
        )
    
    def on_retest(self, level_id: str, success: bool = True) -> None:
        """
        Обработка ретеста уровня.
        
        Вызывается из RetestDetector.
        """
        level = self._tracked_levels.get(level_id)
        if level is None:
            return
        
        now_ns = time.time_ns()
        
        if not self._can_emit_event(level_id, now_ns):
            return
        
        if success:
            # Ретест успешен — уровень снова активен
            level.state = LevelState.ACTIVE
            level.touches += 1
            self._emit_event(
                LevelEventType.RETESTED,
                level_id,
                now_ns,
                {"touches": level.touches},
            )
        else:
            # Ретест не удался — уровень остаётся в текущем состоянии
            self._emit_event(
                LevelEventType.TESTED,
                level_id,
                now_ns,
                {"result": "failed"},
            )
    
    def update_states(self) -> None:
        """
        Обновляет состояния уровней.
        
        Вызывается периодически (каждую секунду).
        Обрабатывает:
        - Переход BROKEN → RETESTABLE
        - Устаревание уровней
        - Очистку невалидных уровней
        """
        now_ns = time.time_ns()
        now_sec = now_ns / 1_000_000_000
        
        levels_to_remove = []
        
        for level_id, level in self._tracked_levels.items():
            # Проверяем переход BROKEN → RETESTABLE
            if level.state == LevelState.BROKEN:
                breakout_ts = self._breakout_ts.get(level_id, 0)
                if breakout_ts > 0:
                    time_since_breakout_sec = (now_ns - breakout_ts) / 1_000_000_000
                    
                    if time_since_breakout_sec >= self.config.retestable_delay_sec:
                        # Переводим в RETESTABLE
                        level.state = LevelState.RETESTABLE
                        
                        if self._can_emit_event(level_id, now_ns):
                            self._emit_event(
                                LevelEventType.RETESTABLE,
                                level_id,
                                now_ns,
                                {"time_since_breakout_sec": time_since_breakout_sec},
                            )
                    
                    elif time_since_breakout_sec >= self.config.broken_lifetime_sec:
                        # Уровень устарел после пробоя
                        level.state = LevelState.INVALID
            
            # Проверяем устаревание после ложного пробоя
            elif level.state == LevelState.FAILED_BREAK:
                if self._can_emit_event(level_id, now_ns):
                    # Проверяем, прошёл ли cooldown
                    last_event = self._last_event_ts.get(level_id, 0)
                    cooldown_sec = self.config.failed_break_cooldown_sec
                    
                    if (now_ns - last_event) / 1_000_000_000 >= cooldown_sec:
                        # Cooldown прошёл — уровень снова активен
                        level.state = LevelState.ACTIVE
                        
                        self._emit_event(
                            LevelEventType.ACTIVATED,
                            level_id,
                            now_ns,
                            {"reason": "cooldown_expired"},
                        )
            
            # Проверяем максимальное время жизни
            age_sec = level.age_sec(now_ns)
            if age_sec > self.config.max_level_lifetime_sec:
                level.state = LevelState.INVALID
            
            # Собираем невалидные уровни для удаления
            if level.state == LevelState.INVALID:
                levels_to_remove.append(level_id)
        
        # Удаляем невалидные уровни
        for level_id in levels_to_remove:
            self.untrack_level(level_id)
    
    def get_stats(self) -> Dict:
        """Возвращает статистику трекера."""
        return {
            "tracked_levels": len(self._tracked_levels),
            "active_levels": len(self.get_active_levels()),
            "retestable_levels": len(self.get_retestable_levels()),
            "total_events": self._total_events,
            "breakout_count": self._breakout_count,
            "failed_break_count": self._failed_break_count,
        }
    
    def reset(self) -> None:
        """Сбрасывает состояние трекера."""
        self._tracked_levels.clear()
        self._breakout_ts.clear()
        self._last_event_ts.clear()
        self._total_events = 0
        self._breakout_count = 0
        self._failed_break_count = 0
    
    def _can_emit_event(self, level_id: str, now_ns: int) -> bool:
        """Проверяет, можно ли отправить событие."""
        last_ts = self._last_event_ts.get(level_id, 0)
        interval_sec = (now_ns - last_ts) / 1_000_000_000
        return interval_sec >= self.config.min_event_interval_sec
    
    def _emit_event(
        self,
        event_type: LevelEventType,
        level_id: str,
        now_ns: int,
        details: Dict,
    ) -> None:
        """Отправляет событие уровня."""
        event = LevelEvent(
            event_type=event_type,
            level_id=level_id,
            symbol=self.symbol,
            ts_ns=now_ns,
            details=details,
        )
        
        self._last_event_ts[level_id] = now_ns
        self._total_events += 1
        
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception:
                pass  # Ошибка в обработчике не должна ронять трекер


class LevelTrackerManager:
    """
    Менеджер LevelTracker'ов для множества символов.
    
    Использование:
        manager = LevelTrackerManager()
        
        # Получаем трекер для символа
        tracker = manager.get_or_create("BTCUSDT")
        
        # Обновляем состояния всех уровней
        manager.update_all_states()
    """
    
    def __init__(self):
        self._trackers: Dict[str, LevelTracker] = {}
    
    def get_or_create(
        self,
        symbol: str,
        config: Optional[LevelTrackerConfig] = None,
    ) -> LevelTracker:
        """Возвращает трекер для символа, создавая при необходимости."""
        symbol = symbol.upper()
        
        if symbol not in self._trackers:
            self._trackers[symbol] = LevelTracker(
                symbol=symbol,
                config=config,
            )
        
        return self._trackers[symbol]
    
    def update_all_states(self) -> None:
        """Обновляет состояния всех уровней."""
        for tracker in self._trackers.values():
            tracker.update_states()
    
    def get_active_levels(self, symbol: str) -> List[Level]:
        """Возвращает активные уровни для символа."""
        tracker = self._trackers.get(symbol.upper())
        if tracker is None:
            return []
        return tracker.get_active_levels()
    
    def get_retestable_levels(self, symbol: str) -> List[Level]:
        """Возвращает уровни для ретеста."""
        tracker = self._trackers.get(symbol.upper())
        if tracker is None:
            return []
        return tracker.get_retestable_levels()
    
    def reset_all(self) -> None:
        """Сбрасывает все трекеры."""
        for tracker in self._trackers.values():
            tracker.reset()
    
    def all_symbols(self) -> List[str]:
        """Возвращает список всех символов."""
        return list(self._trackers.keys())