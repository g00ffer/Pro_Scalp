#!/bin/bash
# Скрипт быстрой настройки нового рабочего места

set -e

echo "🚀 Настройка рабочего места proscalper..."

# Проверка Python
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 не найден. Установите его."
    exit 1
fi

# Создание виртуального окружения
if [ ! -d ".venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3.12 -m venv .venv
fi

# Активация
source .venv/bin/activate

# Обновление pip
echo "📦 Обновление pip..."
pip install --upgrade pip

# Установка зависимостей
echo "📦 Установка зависимостей..."
pip install -e ".[dev]"

# Создание локальных конфигов из шаблонов
if [ ! -f "configs/default.yaml" ]; then
    echo "⚙️  Создание configs/default.yaml из шаблона..."
    cp configs/default.yaml.example configs/default.yaml
    echo "   ⚠️  Отредактируйте configs/default.yaml под свои нужды"
fi

if [ ! -f ".env" ]; then
    echo "🔐 Создание .env из шаблона..."
    cp secrets/.env.example .env
    echo "   ⚠️  Обязательно заполните API ключи в .env"
fi

# Создание директорий
mkdir -p data logs secrets configs/local

echo ""
echo "✅ Настройка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "   1. Отредактируйте .env и добавьте API ключи"
echo "   2. Отредактируйте configs/default.yaml"
echo "   3. Запустите сборщик: python -m proscalper.app.collector configs/default.yaml"
echo ""
echo "🔐 Помните: .env и configs/local/ НЕ добавляются в git!"
