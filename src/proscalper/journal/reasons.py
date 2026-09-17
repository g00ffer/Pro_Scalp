"""
Стандартизированные коды причин (Reason Codes).

Единый реестр всех причин, используемых в журналах решений
(DecisionJournal), журналах инцидентов (IncidentJournal),
сигнальном слое и риск-менеджменте.

Принципы:
- каждый код — строковая константа (для прямой записи в JSONL)
- группировка по категориям (сигналы, отклонения, риски, выходы)
- коды используются в DecisionJournal.reasons / reject_reasons
- коды самодокументируемые (по имени понятна причина)
- реестр позволяет валидировать и фильтровать

Используется в связке с:
- journal/decision_journal.py (log_* методы)
- journal/incident_journal.py (report_* методы)
- signals/signal_generator.py (создание/отклонение сигналов)
- risk/risk_manager.py (одобрение/отклонение риска)
- execution/order_manager.py (закрытие позиций)
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set


# ============================================================
# Причины создания сигнала (почему сигнал принят)
# ============================================================

class SignalReason:
    """Коды причин создания сигнала."""
    
    # Основные сетапы
    BREAKOUT = "BREAKOUT"                          # пробой уровня
    RETEST = "RETEST"                              # ретест пробитого уровня
    FALSE_BREAKOUT = "FALSE_BREAKOUT"              # ложный пробой (закол)
    
    # Подтверждения уровня
    LEVEL_ACTIVE = "LEVEL_ACTIVE"                  # уровень активен и значим
    LEVEL_STRONG = "LEVEL_STRONG"                  # уровень с высоким strength
    LEVEL_MULTI_TOUCH = "LEVEL_MULTI_TOUCH"        # уровень с множеством касаний
    
    # Подтверждения через стакан
    COMPRESSION = "COMPRESSION"                    # консолидация/сжатие у уровня
    WALLS_CONSUMED = "WALLS_CONSUMED"              # стены на уровне проедены
    IMBALANCE_CONFIRMED = "IMBALANCE_CONFIRMED"    # дисбаланс в сторону пробоя
    
    # Подтверждения через ленту
    DELTA_CONFIRMED = "DELTA_CONFIRMED"            # нетто-дельта в сторону пробоя
    IMPULSE_CONFIRMED = "IMPULSE_CONFIRMED"        # импульс подтверждён
    VOLUME_BURST = "VOLUME_BURST"                  # всплеск объёма
    
    # Кросс-маркетные подтверждения
    CROSS_MARKET_CONFIRMED = "CROSS_MARKET_CONFIRMED"  # спот подтверждает
    
    # Общие
    STRATEGY_ENTRY = "STRATEGY_ENTRY"              # вход по стратегии


# ============================================================
# Причины отклонения сигнала
# ============================================================

class RejectReason:
    """Коды причин отклонения сигнала."""
    
    # Проблемы с дельтой
    NO_POSITIVE_DELTA = "NO_POSITIVE_DELTA"        # нет положительной дельты для лонга
    NO_NEGATIVE_DELTA = "NO_NEGATIVE_DELTA"        # нет отрицательной дельты для шорта
    DELTA_TOO_WEAK = "DELTA_TOO_WEAK"              # дельта недостаточна
    
    # Проблемы со стаканом
    WALL_ON_PATH = "WALL_ON_PATH"                  # значимая стена на пути пробоя
    SPOOF_DETECTED = "SPOOF_DETECTED"              # стена классифицирована как спуфинг
    ICEBERG_DETECTED = "ICEBERG_DETECTED"          # обнаружен айсберг
    LOW_LIQUIDITY = "LOW_LIQUIDITY"                # недостаточная ликвидность
    HIGH_SPREAD = "HIGH_SPREAD"                    # спред слишком широкий
    
    # Проблемы с лентой
    LOW_TRADE_RATE = "LOW_TRADE_RATE"              # слишком мало сделок
    VOLUME_IMBALANCE_ADVERSE = "VOLUME_IMBALANCE_ADVERSE"  # объём против направления
    
    # Кросс-маркетные
    CROSS_MARKET_CONTRADICTED = "CROSS_MARKET_CONTRADICTED"  # спот противоречит
    
    # Системные ограничения
    SYMBOL_DISABLED = "SYMBOL_DISABLED"            # символ отключён (защита)
    MAX_POSITIONS_REACHED = "MAX_POSITIONS_REACHED"  # лимит позиций
    MAX_DAILY_TRADES_EXCEEDED = "MAX_DAILY_TRADES_EXCEEDED"  # дневной лимит сделок
    MAX_DAILY_LOSS_REACHED = "MAX_DAILY_LOSS_REACHED"  # дневной лимит убытков
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"      # аварийная остановка активна
    
    # Проблемы данных
    OUT_OF_SYNC = "OUT_OF_SYNC"                    # стакан рассинхронизирован
    STALE_BOOKTICKER = "STALE_BOOKTICKER"          # устаревший bookTicker
    NO_LEVEL_DATA = "NO_LEVEL_DATA"                # нет данных по уровням
    BOOK_NOT_INITIALIZED = "BOOK_NOT_INITIALIZED"  # стакан не инициализирован
    
    # Проблемы импульса
    IMPULSE_TOO_WEAK = "IMPULSE_TOO_WEAK"          # импульс недостаточен
    NO_COMPRESSION = "NO_COMPRESSION"              # нет консолидации перед пробоем
    
    # Общие
    FILTER_REJECTED = "FILTER_REJECTED"            # отклонён фильтром
    RISK_REJECTED = "RISK_REJECTED"                # отклонён риск-менеджером


# ============================================================
# Причины одобрения/отклонения риска
# ============================================================

class RiskReason:
    """Коды причин риск-менеджмента."""
    
    # Одобрения
    RISK_WITHIN_LIMITS = "RISK_WITHIN_LIMITS"      # риск в пределах лимитов
    POSITION_SIZE_OK = "POSITION_SIZE_OK"          # размер позиции корректен
    
    # Отклонения
    MAX_NOTIONAL_EXCEEDED = "MAX_NOTIONAL_EXCEEDED"  # превышен макс. номинал
    MIN_NOTIONAL_NOT_MET = "MIN_NOTIONAL_NOT_MET"    # ниже мин. номинала
    DAILY_LOSS_LIMIT_HIT = "DAILY_LOSS_LIMIT_HIT"    # дневной лимит убытков
    CONSECUTIVE_LOSSES_LIMIT = "CONSECUTIVE_LOSSES_LIMIT"  # лимит серии убытков
    MAX_LEVERAGE_EXCEEDED = "MAX_LEVERAGE_EXCEEDED"  # превышено плечо
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"      # недостаточно маржи


# ============================================================
# Причины выхода из позиции
# ============================================================

class ExitReason:
    """Коды причин закрытия позиции."""
    
    # Стопы
    STOP_LOSS = "STOP_LOSS"                        # срабатывание стоп-лосса
    TRAILING_STOP = "TRAILING_STOP"                # срабатывание трейлинг-стопа
    BREAK_EVEN = "BREAK_EVEN"                      # перенос стопа в безубыток
    
    # Тейки
    TAKE_PROFIT = "TAKE_PROFIT"                    # тейк-профит
    PARTIAL_PROFIT = "PARTIAL_PROFIT"              # частичная фиксация прибыли
    
    # Стратегические
    HOLD_EXPIRED = "HOLD_EXPIRED"                  # истечение времени удержания
    REVERSE_BREAK = "REVERSE_BREAK"                # обратный пробой
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"              # сигнал устарел
    
    # Ручные и аварийные
    MANUAL_CLOSE = "MANUAL_CLOSE"                  # ручное закрытие
    FORCED_CLOSE = "FORCED_CLOSE"                  # принудительное закрытие
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"        # аварийное закрытие
    PROTECTION_TIMEOUT = "PROTECTION_TIMEOUT"      # таймаут защиты
    KILL_SWITCH = "KILL_SWITCH"                    # аварийная остановка системы
    
    # Технические
    SESSION_END = "SESSION_END"                    # конец торговой сессии
    SYMBOL_DELISTED = "SYMBOL_DELISTED"            # делистинг инструмента


# ============================================================
# Причины отклонения ордеров биржей
# ============================================================

class OrderRejectReason:
    """Коды причин отклонения ордеров."""
    
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"    # недостаточно средств
    RATE_LIMIT = "RATE_LIMIT"                        # лимит запросов
    MARKET_CLOSED = "MARKET_CLOSED"                  # рынок закрыт
    INVALID_QUANTITY = "INVALID_QUANTITY"            # невалидный объём
    INVALID_PRICE = "INVALID_PRICE"                  # невалидная цена
    PRICE_OUT_OF_BOUNDS = "PRICE_OUT_OF_BOUNDS"      # цена вне допустимого диапазона
    DUPLICATE_ORDER = "DUPLICATE_ORDER"              # дубликат ордера
    SELF_TRADE = "SELF_TRADE"                        # попытка самоторговли
    ORDER_NOT_FOUND = "ORDER_NOT_FOUND"              # ордер не найден
    POSITION_CLOSED = "POSITION_CLOSED"              # позиция уже закрыта


# ============================================================
# Причины защиты (защитный слой)
# ============================================================

class ProtectionReason:
    """Коды причин защитных действий."""
    
    STOP_LEG_REJECTED = "STOP_LEG_REJECTED"          # нога стопа отклонена
    STOP_PLACE_FAILED = "STOP_PLACE_FAILED"          # не удалось поставить стоп
    ENTRY_NO_FILL_TIMEOUT = "ENTRY_NO_FILL_TIMEOUT"  # вход не исполнился
    PARTIAL_FILL_UNPROTECTED = "PARTIAL_FILL_UNPROTECTED"  # частичный филл без защиты
    BATCH_ORDER_FAILED = "BATCH_ORDER_FAILED"        # пакетный ордер не прошёл
    WATCHDOG_TIMEOUT = "WATCHDOG_TIMEOUT"            # таймаут вотчдога


# ============================================================
# Реестр всех кодов
# ============================================================

# Все известные коды по категориям
_ALL_REASONS: Dict[str, FrozenSet[str]] = {
    "signal": frozenset([
        SignalReason.BREAKOUT,
        SignalReason.RETEST,
        SignalReason.FALSE_BREAKOUT,
        SignalReason.LEVEL_ACTIVE,
        SignalReason.LEVEL_STRONG,
        SignalReason.LEVEL_MULTI_TOUCH,
        SignalReason.COMPRESSION,
        SignalReason.WALLS_CONSUMED,
        SignalReason.IMBALANCE_CONFIRMED,
        SignalReason.DELTA_CONFIRMED,
        SignalReason.IMPULSE_CONFIRMED,
        SignalReason.VOLUME_BURST,
        SignalReason.CROSS_MARKET_CONFIRMED,
        SignalReason.STRATEGY_ENTRY,
    ]),
    "reject": frozenset([
        RejectReason.NO_POSITIVE_DELTA,
        RejectReason.NO_NEGATIVE_DELTA,
        RejectReason.DELTA_TOO_WEAK,
        RejectReason.WALL_ON_PATH,
        RejectReason.SPOOF_DETECTED,
        RejectReason.ICEBERG_DETECTED,
        RejectReason.LOW_LIQUIDITY,
        RejectReason.HIGH_SPREAD,
        RejectReason.LOW_TRADE_RATE,
        RejectReason.VOLUME_IMBALANCE_ADVERSE,
        RejectReason.CROSS_MARKET_CONTRADICTED,
        RejectReason.SYMBOL_DISABLED,
        RejectReason.MAX_POSITIONS_REACHED,
        RejectReason.MAX_DAILY_TRADES_EXCEEDED,
        RejectReason.MAX_DAILY_LOSS_REACHED,
        RejectReason.KILL_SWITCH_ACTIVE,
        RejectReason.OUT_OF_SYNC,
        RejectReason.STALE_BOOKTICKER,
        RejectReason.NO_LEVEL_DATA,
        RejectReason.BOOK_NOT_INITIALIZED,
        RejectReason.IMPULSE_TOO_WEAK,
        RejectReason.NO_COMPRESSION,
        RejectReason.FILTER_REJECTED,
        RejectReason.RISK_REJECTED,
    ]),
    "risk": frozenset([
        RiskReason.RISK_WITHIN_LIMITS,
        RiskReason.POSITION_SIZE_OK,
        RiskReason.MAX_NOTIONAL_EXCEEDED,
        RiskReason.MIN_NOTIONAL_NOT_MET,
        RiskReason.DAILY_LOSS_LIMIT_HIT,
        RiskReason.CONSECUTIVE_LOSSES_LIMIT,
        RiskReason.MAX_LEVERAGE_EXCEEDED,
        RiskReason.INSUFFICIENT_MARGIN,
    ]),
    "exit": frozenset([
        ExitReason.STOP_LOSS,
        ExitReason.TRAILING_STOP,
        ExitReason.BREAK_EVEN,
        ExitReason.TAKE_PROFIT,
        ExitReason.PARTIAL_PROFIT,
        ExitReason.HOLD_EXPIRED,
        ExitReason.REVERSE_BREAK,
        ExitReason.SIGNAL_EXPIRED,
        ExitReason.MANUAL_CLOSE,
        ExitReason.FORCED_CLOSE,
        ExitReason.EMERGENCY_FLATTEN,
        ExitReason.PROTECTION_TIMEOUT,
        ExitReason.KILL_SWITCH,
        ExitReason.SESSION_END,
        ExitReason.SYMBOL_DELISTED,
    ]),
    "order_reject": frozenset([
        OrderRejectReason.INSUFFICIENT_BALANCE,
        OrderRejectReason.RATE_LIMIT,
        OrderRejectReason.MARKET_CLOSED,
        OrderRejectReason.INVALID_QUANTITY,
        OrderRejectReason.INVALID_PRICE,
        OrderRejectReason.PRICE_OUT_OF_BOUNDS,
        OrderRejectReason.DUPLICATE_ORDER,
        OrderRejectReason.SELF_TRADE,
        OrderRejectReason.ORDER_NOT_FOUND,
        OrderRejectReason.POSITION_CLOSED,
    ]),
    "protection": frozenset([
        ProtectionReason.STOP_LEG_REJECTED,
        ProtectionReason.STOP_PLACE_FAILED,
        ProtectionReason.ENTRY_NO_FILL_TIMEOUT,
        ProtectionReason.PARTIAL_FILL_UNPROTECTED,
        ProtectionReason.BATCH_ORDER_FAILED,
        ProtectionReason.WATCHDOG_TIMEOUT,
    ]),
}

# Плоское множество всех кодов
_ALL_CODES: FrozenSet[str] = frozenset(
    code for codes in _ALL_REASONS.values() for code in codes
)


# ============================================================
# Валидация и утилиты
# ============================================================

def is_valid_reason(code: str) -> bool:
    """Проверяет, является ли код известным."""
    return code in _ALL_CODES


def validate_reasons(codes: List[str]) -> List[str]:
    """
    Возвращает список неизвестных кодов.
    Пустой список = все коды валидны.
    """
    return [c for c in codes if c not in _ALL_CODES]


def get_category(code: str) -> Optional[str]:
    """Возвращает категорию кода или None."""
    for category, codes in _ALL_REASONS.items():
        if code in codes:
            return category
    return None


def get_all_codes() -> FrozenSet[str]:
    """Возвращает все зарегистрированные коды."""
    return _ALL_CODES


def get_codes_by_category(category: str) -> FrozenSet[str]:
    """Возвращает коды указанной категории."""
    return _ALL_REASONS.get(category, frozenset())


def get_all_categories() -> List[str]:
    """Возвращает список всех категорий."""
    return list(_ALL_REASONS.keys())