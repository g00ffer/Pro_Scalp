#!/bin/bash
# Обновление структуры проекта: добавление новых файлов сверх исходной архитектуры

set -e

echo "📦 Обновление структуры проекта..."

# Создаём недостающие файлы в market_data
if [ ! -f "src/proscalper/market_data/bar_aggregator.py" ]; then
    echo "  + src/proscalper/market_data/bar_aggregator.py"
    touch src/proscalper/market_data/bar_aggregator.py
fi

# Создаём недостающие файлы в signals
if [ ! -f "src/proscalper/signals/approach.py" ]; then
    echo "  + src/proscalper/signals/approach.py"
    touch src/proscalper/signals/approach.py
fi

# Проверяем, что все нужные файлы существуют
echo ""
echo "🔍 Проверка файлов сигнального слоя:"

FILES=(
    "src/proscalper/features/compression.py"
    "src/proscalper/features/book_analyzer.py"
    "src/proscalper/features/impulse_score.py"
    "src/proscalper/signals/approach.py"
    "src/proscalper/signals/breakout.py"
    "src/proscalper/signals/signal_generator.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ "$size" -gt 100 ]; then
            echo "  ✅ $file ($size байт)"
        else
            echo "  ⚠️  $file (пустой, нужно заполнить)"
        fi
    else
        echo "  ❌ $file (отсутствует)"
    fi
done

echo ""
echo "✅ Обновление структуры завершено"