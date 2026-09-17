"""
Часы и временные утилиты системы.

Предоставляет единый интерфейс времени для всех модулей:
- SystemClock — реальное время (для живого режима)
- VirtualClock — виртуальное время (для бэктеста)
- синхронизация с биржевым временем (offset local ↔ exchange)
- монотонные часы для измерения задержек (не зависят от NTP-сдвигов)
- утилиты конвертации (ms ↔ ns, форматирование)

Зачем это нужно:
- все события имеют два таймстампа: биржевой и локальный
- локальные часы могут отличаться от биржевых на 10-500мс
- для задержек (латентность) нужны МОНОТОННЫЕ часы —
  они не прыгают при синхронизации NTP
- в бэктесте время управляется движком (не реальное)

Используется в связке с:
- core/events.py (ts_exchange_ns, ts_local_ns)
- backtest/engine.py (VirtualClock для прогона)
- exchange/binance_futures/rest.py (get_server_time)
- journal/market_state_snapshot.py (секция таймингов)

Принципы:
- интерфейс единый: clock.now_ns() работает везде
- монотонные часы — только для измерения дельт, не для меток событий
- синхронизация с биржей обновляется периодически
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol


# ============================================================
# Интерфейс часов
# ============================================================

class Clock(Protocol):
    """
    Интерфейс часов.

    Реализации:
    - SystemClock: реальное время
    - VirtualClock: управляемое время (бэктест)
    """

    def now_ns(self) -> int:
        """Текущее время в наносекундах (эпоха Unix)."""
        ...

    def now_ms(self) -> int:
        """Текущее время в миллисекундах (эпоха Unix)."""
        ...

    def now_sec(self) -> float:
        """Текущее время в секундах (эпоха Unix)."""
        ...

    def monotonic_ns(self) -> int:
        """
        Монотонное время в наносекундах.

        ВАЖНО: подходит ТОЛЬКО для измерения интервалов.
        Не соответствует эпохе Unix. Не сравнивать с now_ns().
        """
        ...


# ============================================================
# Системные часы (живой режим)
# ============================================================

class SystemClock:
    """
    Часы на основе системного времени.

    Использование:
        clock = SystemClock()
        ts = clock.now_ns()

        # Измерение задержки
        start = clock.monotonic_ns()
        do_something()
        elapsed_ms = (clock.monotonic_ns() - start) / 1_000_000
    """

    def now_ns(self) -> int:
        return time.time_ns()

    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def now_sec(self) -> float:
        return time.time()

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


# ============================================================
# Виртуальные часы (бэктест)
# ============================================================

class VirtualClock:
    """
    Управляемые часы для бэктеста.

    Время задаётся движком бэктеста при обработке каждого события.
    Монотонные часы следуют за виртуальным временем.

    Использование:
        clock = VirtualClock(start_ns=1726500000_000_000_000)

        # При обработке события
        clock.set_time(event.ts_exchange_ns)

        # Стратегия видит виртуальное время
        ts = clock.now_ns()
    """

    def __init__(self, start_ns: int = 0) -> None:
        self._time_ns = start_ns

    def set_time(self, ts_ns: int) -> None:
        """Устанавливает текущее время (вызывается движком бэктеста)."""
        if ts_ns < self._time_ns:
            # Время не должно идти назад — это признак ошибки
            # в данных (нарушение хронологического порядка)
            raise ValueError(
                f"VirtualClock: попытка установить время назад: "
                f"текущее={self._time_ns}, новое={ts_ns}"
            )
        self._time_ns = ts_ns

    def now_ns(self) -> int:
        return self._time_ns

    def now_ms(self) -> int:
        return self._time_ns // 1_000_000

    def now_sec(self) -> float:
        return self._time_ns / 1_000_000_000

    def monotonic_ns(self) -> int:
        # В бэктесте монотонное время = виртуальное время
        return self._time_ns

    @property
    def time_ns(self) -> int:
        """Доступ к текущему времени (для движка)."""
        return self._time_ns


# ============================================================
# Синхронизация с биржевым временем
# ============================================================

@dataclass
class ExchangeTimeSync:
    """
    Результат синхронизации локального времени с биржевым.

    Хранит смещение (offset) между локальными часами и сервером биржи:
    - offset_ns > 0: локальные часы позади биржевых
    - offset_ns < 0: локальные часы впереди биржевых

    Использование:
        sync = ExchangeTimeSync(
            offset_ns=45_000_000,   # локальные отстают на 45мс
            synced_ts_ns=time.time_ns(),
        )

        # Перевод локального времени в биржевое
        exchange_ts = sync.local_to_exchange(local_ts)

        # Перевод биржевого времени в локальное
        local_ts = sync.exchange_to_local(exchange_ts)
    """

    # Смещение: биржевое время - локальное время (в наносекундах)
    offset_ns: int = 0

    # Когда была выполнена синхронизация
    synced_ts_ns: int = 0

    # Задержка сети в момент синхронизации (round-trip)
    rtt_ms: float = 0.0

    # Точность синхронизации (примерная)
    accuracy_ms: float = 0.0

    @property
    def offset_ms(self) -> float:
        return self.offset_ns / 1_000_000

    def local_to_exchange(self, local_ts_ns: int) -> int:
        """Переводит локальный таймстамп в биржевой."""
        return local_ts_ns + self.offset_ns

    def exchange_to_local(self, exchange_ts_ns: int) -> int:
        """Переводит биржевой таймстамп в локальный."""
        return exchange_ts_ns - self.offset_ns

    def is_stale(self, max_age_sec: float = 60.0) -> bool:
        """Проверяет, не устарела ли синхронизация."""
        if self.synced_ts_ns == 0:
            return True
        age_ns = time.time_ns() - self.synced_ts_ns
        return age_ns > max_age_sec * 1_000_000_000


async def sync_with_exchange(
    rest_client,
    samples: int = 3,
) -> ExchangeTimeSync:
    """
    Синхронизирует локальное время с биржевым сервером.

    Алгоритм (упрощённый NTP):
    1. Замеряем локальное время до запроса (t1)
    2. Отправляем GET /fapi/v1/time
    3. Замеряем локальное время после ответа (t2)
    4. Биржевое время = server_time из ответа
    5. Предполагаем: серверное время получено в момент (t1 + t2) / 2
    6. offset = server_time - (t1 + t2) / 2

    Повторяем samples раз, берём медиану (устойчиво к выбросам).

    Args:
        rest_client: клиент с методом get_server_time()
        samples: количество замеров (больше = точнее)

    Returns:
        ExchangeTimeSync с рассчитанным смещением
    """
    offsets = []
    rtts = []

    for _ in range(max(1, samples)):
        t1_ns = time.time_ns()
        server_time_ms = await rest_client.get_server_time()
        t2_ns = time.time_ns()

        # Биржевое время в момент получения ответа
        server_time_ns = server_time_ms * 1_000_000

        # Середина окна запроса
        mid_ns = (t1_ns + t2_ns) // 2

        # Смещение
        offset = server_time_ns - mid_ns
        offsets.append(offset)

        # RTT
        rtt_ms = (t2_ns - t1_ns) / 1_000_000
        rtts.append(rtt_ms)

    # Медиана смещений (устойчиво к выбросам)
    offsets.sort()
    median_offset = offsets[len(offsets) // 2]

    # Минимальный RTT (наилучший замер)
    min_rtt = min(rtts) if rtts else 0.0

    # Точность ≈ половина RTT (как в NTP)
    accuracy_ms = min_rtt / 2.0

    return ExchangeTimeSync(
        offset_ns=median_offset,
        synced_ts_ns=time.time_ns(),
        rtt_ms=min_rtt,
        accuracy_ms=accuracy_ms,
    )


# ============================================================
# Утилиты конвертации
# ============================================================

NS_PER_MS = 1_000_000
NS_PER_SEC = 1_000_000_000


def ms_to_ns(ms: float) -> int:
    """Миллисекунды → наносекунды."""
    return int(ms * NS_PER_MS)


def ns_to_ms(ns: int) -> float:
    """Наносекунды → миллисекунды."""
    return ns / NS_PER_MS


def sec_to_ns(sec: float) -> int:
    """Секунды → наносекунды."""
    return int(sec * NS_PER_SEC)


def ns_to_sec(ns: int) -> float:
    """Наносекунды → секунды."""
    return ns / NS_PER_SEC


def format_ts_ns(ts_ns: int) -> str:
    """
    Форматирует наносекундный таймстамп в читаемый вид.

    Пример: "2026-09-17 14:22:03.772"
    """
    if ts_ns <= 0:
        return "N/A"

    sec = ts_ns / NS_PER_SEC
    ms = (ts_ns % NS_PER_SEC) // NS_PER_MS

    local = time.localtime(sec)
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', local)}.{ms:03d}"


def format_duration_ns(duration_ns: int) -> str:
    """
    Форматирует длительность в наносекундах.

    Примеры:
        500_000     → "0.500 ms"
        1_500_000   → "1.500 ms"
        45_000      → "45 µs"
        120         → "120 ns"
    """
    if duration_ns < 1_000:
        return f"{duration_ns} ns"
    if duration_ns < 1_000_000:
        return f"{duration_ns / 1_000:.1f} µs"
    if duration_ns < 1_000_000_000:
        return f"{duration_ns / 1_000_000:.3f} ms"
    return f"{duration_ns / 1_000_000_000:.2f} s"


def elapsed_since_ns(start_ns: int) -> int:
    """
    Возвращает прошедшее время с момента start_ns.

    Внимание: использует time.time_ns() — может прыгать при
    синхронизации времени. Для измерения задержек лучше
    использовать monotonic_ns().
    """
    return time.time_ns() - start_ns


# ============================================================
# Дефолтный экземпляр
# ============================================================

# Глобальный экземпляр системных часов для использования по умолчанию
_default_clock: Optional[Clock] = None


def get_clock() -> Clock:
    """
    Возвращает глобальный экземпляр часов.

    По умолчанию — SystemClock. Для бэктеста можно установить
    VirtualClock через set_clock().
    """
    global _default_clock
    if _default_clock is None:
        _default_clock = SystemClock()
    return _default_clock


def set_clock(clock: Clock) -> None:
    """
    Устанавливает глобальный экземпляр часов.

    Использование в бэктесте:
        virtual = VirtualClock(start_ns=...)
        set_clock(virtual)

        # Все модули, использующие get_clock(), увидят
        # виртуальное время
    """
    global _default_clock
    _default_clock = clock


def reset_clock() -> None:
    """Сбрасывает к системным часам."""
    global _default_clock
    _default_clock = SystemClock()