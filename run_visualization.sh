#!/bin/bash

# Скрипт для запуска интерактивной визуализации

echo "============================================================"
echo "ЗАПУСК ИНТЕРАКТИВНОЙ ВИЗУАЛИЗАЦИИ ПАТЕНТНОГО ЛАНДШАФТА"
echo "============================================================"
echo ""

# Проверяем наличие кэша
CACHE_FILE="data/clustering_cache.json"

if [ ! -f "$CACHE_FILE" ]; then
    echo "⚠️  Кэш не найден: $CACHE_FILE"
    echo ""
    echo "Генерируем кэш кластеризации (это займёт ~5-10 минут)..."
    echo ""
    cd src/visualization
    python3 generate_clustering_cache.py
    cd ../..
    echo ""
fi

echo "✓ Кэш найден: $CACHE_FILE"
echo ""
echo "Запуск веб-приложения..."
echo ""

cd src/visualization
python3 app.py
