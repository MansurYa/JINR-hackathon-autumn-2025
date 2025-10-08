"""
Скрипт для генерации полного кэша иерархической кластеризации.
Выполняет кластеризацию для всех уровней (1948 -> 1) и сохраняет результаты.
"""

import json
import os
import sys
import numpy as np
import colorsys
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from document_embeddings_generator import load_document_embeddings_npz
from hierarchical_clustering import load_document_tags


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """
    Конвертирует RGB (0-1) в HEX цвет.

    Args:
        rgb: Кортеж (r, g, b) со значениями 0-1

    Returns:
        HEX строка цвета
    """
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


def generate_base_colors(n: int) -> List[str]:
    """
    Генерирует базовые цвета для начальных кластеров.

    Args:
        n: Количество цветов

    Returns:
        Список HEX цветов
    """
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append(rgb_to_hex(rgb))
    return colors


def blend_colors(color1: str, color2: str) -> str:
    """
    Смешивает два HEX цвета для получения среднего.

    Args:
        color1: HEX цвет 1
        color2: HEX цвет 2

    Returns:
        Усредненный HEX цвет
    """
    # Конвертируем в RGB
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    # Усредняем
    r = (r1 + r2) // 2
    g = (g1 + g2) // 2
    b = (b1 + b2) // 2

    return f'#{r:02x}{g:02x}{b:02x}'


def compute_centroid(embeddings: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Вычисляет центроид (среднюю точку) для набора эмбедингов.

    Args:
        embeddings: Массив всех эмбедингов
        indices: Индексы документов для вычисления центроида

    Returns:
        Вектор центроида
    """
    cluster_embeddings = embeddings[indices]
    return np.mean(cluster_embeddings, axis=0)


def get_top_tags(doc_tags: Dict[int, Dict[str, float]], doc_ids: List[int], top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Получает топ-N тегов для набора документов.

    Args:
        doc_tags: Словарь тегов документов
        doc_ids: Список ID документов
        top_n: Количество топ-тегов

    Returns:
        Список кортежей (тег, вес)
    """
    tag_scores = defaultdict(float)

    for doc_id in doc_ids:
        if doc_id in doc_tags:
            for tag, weight in doc_tags[doc_id].items():
                tag_scores[tag] += weight

    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_tags[:top_n]


def generate_cluster_name(top_tags: List[Tuple[str, float]], num_clusters: int, total_docs: int) -> str:
    """
    Генерирует название кластера в зависимости от количества кластеров.

    Args:
        top_tags: Топ теги кластера
        num_clusters: Текущее количество кластеров
        total_docs: Общее количество документов

    Returns:
        Название кластера
    """
    if num_clusters < 5:
        # Топ-3 тега с многоточием
        tags = [tag for tag, _ in top_tags[:3]]
        return ", ".join(tags) + ", ..."
    elif num_clusters < 10:
        # Топ-2 тега с многоточием
        tags = [tag for tag, _ in top_tags[:2]]
        return ", ".join(tags) + ", ..."
    else:
        # Топ-1 тег без многоточия
        return top_tags[0][0] if top_tags else "Unknown"


def generate_full_clustering_cache(
    embeddings: np.ndarray,
    document_ids: List[int],
    doc_tags: Dict[int, Dict[str, float]],
    output_file: str,
    method: str = 'average',
    metric: str = 'cosine'
) -> None:
    """
    Генерирует полный кэш кластеризации для всех уровней.

    Args:
        embeddings: Массив эмбедингов документов
        document_ids: Список ID документов
        doc_tags: Словарь тегов документов
        output_file: Путь к выходному файлу
        method: Метод linkage
        metric: Метрика расстояния
    """
    n_docs = len(document_ids)
    print(f"\n{'='*60}")
    print(f"Генерация кэша кластеризации для {n_docs} документов")
    print(f"{'='*60}\n")

    # Создаём базовые цвета для максимального количества кластеров (когда n_clusters = n_docs)
    print("Генерация базовых цветов...")
    base_colors = generate_base_colors(min(n_docs, 360))  # Ограничиваем 360 для лучшего распределения hue
    if n_docs > 360:
        # Если документов больше 360, дублируем цвета
        base_colors = base_colors * (n_docs // 360 + 1)
    base_colors = base_colors[:n_docs]

    # Словарь для хранения всех уровней кластеризации
    cache = {}

    # Маппинг cluster_id -> color для каждого уровня
    cluster_colors_history = {}

    # Начинаем с максимального количества кластеров (каждый документ = кластер)
    # Идём от n_docs до 1
    print("\nВыполнение иерархической кластеризации...\n")

    # Для оптимизации: вычислим все уровни за один проход
    # Используем linkage matrix для получения истории слияний
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

    linkage_matrix = scipy_linkage(embeddings, method=method, metric=metric)

    # Генерируем кэш для разных уровней кластеризации
    # Делаем выборочно для оптимизации (не все 1948 уровней, а ключевые точки)
    cluster_counts = []

    # Логарифмическая шкала для выбора уровней
    cluster_counts.append(1)  # Один кластер
    cluster_counts.append(n_docs)  # Максимум кластеров

    # Промежуточные точки (логарифмическая шкала)
    for i in range(1, 100):
        n = int(np.exp(np.log(n_docs) * i / 100))
        if n not in cluster_counts and 1 < n < n_docs:
            cluster_counts.append(n)

    cluster_counts = sorted(set(cluster_counts))

    print(f"Генерируем кэш для {len(cluster_counts)} уровней кластеризации...")

    for idx, n_clusters in enumerate(cluster_counts):
        if idx % 10 == 0:
            print(f"  Прогресс: {idx}/{len(cluster_counts)} уровней...")

        # Получаем метки кластеров для текущего уровня
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1  # -1 для 0-индексации

        # Группируем документы по кластерам
        clusters = defaultdict(list)
        for doc_idx, label in enumerate(labels):
            clusters[label].append(doc_idx)

        # Для текущего уровня кластеризации вычисляем информацию о каждом кластере
        level_data = {
            'n_clusters': n_clusters,
            'clusters': []
        }

        # Определяем цвета кластеров
        if n_clusters == n_docs:
            # Базовый случай: каждый документ = кластер со своим цветом
            cluster_colors = {i: base_colors[i] for i in range(n_docs)}
        else:
            # Наследуем и смешиваем цвета от предыдущего уровня
            cluster_colors = {}

            # Для каждого кластера на текущем уровне находим цвета его "детей" с предыдущего уровня
            # Упрощение: используем первый цвет из документов кластера
            for cluster_id, doc_indices in clusters.items():
                # Берём цвета документов из базового уровня
                doc_colors = [base_colors[i] for i in doc_indices[:5]]  # Берём до 5 цветов для смешивания
                if len(doc_colors) == 1:
                    cluster_colors[cluster_id] = doc_colors[0]
                else:
                    # Смешиваем цвета
                    blended = doc_colors[0]
                    for c in doc_colors[1:]:
                        blended = blend_colors(blended, c)
                    cluster_colors[cluster_id] = blended

        cluster_colors_history[n_clusters] = cluster_colors

        # Для каждого кластера вычисляем информацию
        for cluster_id in sorted(clusters.keys()):
            doc_indices = clusters[cluster_id]
            doc_ids = [document_ids[i] for i in doc_indices]

            # Центроид кластера
            centroid = compute_centroid(embeddings, doc_indices)

            # Топ теги
            top_tags = get_top_tags(doc_tags, doc_ids, top_n=10)

            # Название кластера
            name = generate_cluster_name(top_tags, n_clusters, n_docs)

            # Цвет кластера
            color = cluster_colors.get(cluster_id, '#808080')

            cluster_info = {
                'cluster_id': int(cluster_id),
                'name': name,
                'color': color,
                'centroid': centroid.tolist(),
                'doc_count': len(doc_ids),
                'doc_ids': [int(doc_id) for doc_id in doc_ids],
                'top_tags': [[tag, float(weight)] for tag, weight in top_tags]
            }

            level_data['clusters'].append(cluster_info)

        cache[n_clusters] = level_data

    print(f"\n✓ Кэш сгенерирован для {len(cache)} уровней кластеризации")

    # Сохраняем кэш
    print(f"\nСохранение кэша в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"✓ Кэш успешно сохранён!")
    print(f"  Размер файла: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"  Уровней кластеризации: {len(cache)}")
    print(f"  Диапазон: {min(cluster_counts)} - {max(cluster_counts)} кластеров")


def main():
    """Основная функция."""
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ КЭША КЛАСТЕРИЗАЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ")
    print("="*60)

    # Пути к файлам
    embeddings_file = "../../data/document_embeddings.npz"
    tags_file = "../../data/full_dataset_tags.jsonl"
    output_file = "../../data/clustering_cache.json"

    # Загрузка данных
    print("\nЗагрузка данных...")
    document_ids, embeddings = load_document_embeddings_npz(embeddings_file)
    doc_tags = load_document_tags(tags_file)

    print(f"✓ Загружено {len(document_ids)} документов")
    print(f"✓ Загружено тегов для {len(doc_tags)} документов")

    # Генерация кэша
    generate_full_clustering_cache(
        embeddings=embeddings,
        document_ids=document_ids,
        doc_tags=doc_tags,
        output_file=output_file,
        method='average',
        metric='cosine'
    )

    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ КЭША ЗАВЕРШЕНА!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
