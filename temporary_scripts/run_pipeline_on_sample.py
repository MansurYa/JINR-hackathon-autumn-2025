#!/usr/bin/env python3
"""
Скрипт для запуска полного пайплайна на тестовой выборке из 50 документов.
Включает: генерацию тегов, эмбедингов тегов, эмбедингов документов и кластеризацию.
"""

import os
import sys
import json

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_clustering import hierarchical_clustering_pipeline


def run_full_pipeline():
    """
    Запускает полный пайплайн обработки на тестовой выборке.
    """
    print("=" * 60)
    print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА НА 50 ДОКУМЕНТАХ")
    print("=" * 60)
    print()

    # Параметры
    sample_file = 'temporary_data/sample_50_documents.jsonl'
    tags_output = 'temporary_data/sample_50_tags.jsonl'
    tag_embeddings_output = 'temporary_data/sample_50_tag_embeddings.npz'
    doc_embeddings_output = 'temporary_data/sample_50_doc_embeddings.npz'
    clustering_output = 'temporary_data/sample_50_clustering_results.json'

    # Проверяем наличие входного файла
    if not os.path.exists(sample_file):
        print(f"❌ Ошибка: файл {sample_file} не найден!")
        print("Сначала запустите temporary_scripts/create_sample_dataset.py")
        return

    # Запускаем пайплайн
    hierarchical_clustering_pipeline(
        documents_file=sample_file,
        tags_output_file=tags_output,
        tag_embeddings_file=tag_embeddings_output,
        doc_embeddings_file=doc_embeddings_output,
        clustering_results_file=clustering_output,
        n_clusters=5,
        skip_stage_0=False,  # Генерируем теги
        skip_stage_1=False,  # Генерируем эмбединги тегов
        skip_stage_2=False,  # Генерируем эмбединги документов
        use_umap=True,       # Используем UMAP для визуализации
        umap_output='temporary_data/sample_50_umap.png'
    )

    print()
    print("=" * 60)
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН!")
    print("=" * 60)
    print()
    print("Результаты:")
    print(f"  - Теги документов: {tags_output}")
    print(f"  - Эмбединги тегов: {tag_embeddings_output}")
    print(f"  - Эмбединги документов: {doc_embeddings_output}")
    print(f"  - Результаты кластеризации: {clustering_output}")
    print(f"  - UMAP визуализация: temporary_data/sample_50_umap.png")


if __name__ == "__main__":
    run_full_pipeline()
