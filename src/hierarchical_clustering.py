"""
Модуль для иерархической кластеризации документов.
ЭТАП 4 пайплайна обработки данных.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict

# Добавляем путь к src для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from document_embeddings_generator import load_document_embeddings_npz

# Импорты для кластеризации
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_document_tags(tags_file: str) -> Dict[int, Dict[str, float]]:
    """
    Загружает теги документов из JSONL файла в формат {document_id: {tag: weight}}.

    Args:
        tags_file: Путь к файлу с тегами документов

    Returns:
        Словарь {document_id: {tag: weight}}
    """
    if not os.path.exists(tags_file):
        raise FileNotFoundError(f"Файл с тегами не найден: {tags_file}")

    doc_tags = {}

    with open(tags_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                doc_id = data["document_id"]
                tags = data["tags"]
                doc_tags[doc_id] = tags
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON на строке {line_num}: {e}")
                continue

    return doc_tags


def compute_cluster_names_weighted_frequency(
    cluster_labels: np.ndarray,
    document_ids: List[int],
    doc_tags: Dict[int, Dict[str, float]],
    top_n: int = 3
) -> Dict[int, Dict]:
    """
    Вычисляет названия кластеров по алгоритму Weighted Tag Frequency.

    Алгоритм:
    1. Для каждого кластера суммируем веса тегов по всем документам
    2. Сортируем теги по убыванию суммарного веса
    3. Выбираем топ-N тегов как название кластера

    Args:
        cluster_labels: Массив меток кластеров для каждого документа
        document_ids: Список ID документов (соответствует порядку в cluster_labels)
        doc_tags: Словарь {document_id: {tag: weight}}
        top_n: Количество топ-тегов для названия кластера

    Returns:
        Словарь {cluster_id: {
            'name': str,
            'top_tags': [(tag, weight), ...],
            'doc_count': int,
            'doc_ids': [...]
        }}
    """
    cluster_info = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        # Находим индексы документов в этом кластере
        mask = cluster_labels == cluster_id
        cluster_doc_indices = np.where(mask)[0]
        cluster_doc_ids = [document_ids[i] for i in cluster_doc_indices]

        # Суммируем веса тегов по всем документам кластера
        tag_scores = defaultdict(float)

        for doc_id in cluster_doc_ids:
            if doc_id in doc_tags:
                for tag, weight in doc_tags[doc_id].items():
                    tag_scores[tag] += weight

        # Сортируем теги по убыванию веса
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)

        # Топ-N тегов
        top_tags = sorted_tags[:top_n]

        # Название кластера = топ-3 тега через запятую
        cluster_name = ", ".join([tag for tag, _ in top_tags])

        cluster_info[cluster_id] = {
            'name': cluster_name,
            'top_tags': sorted_tags[:10],  # Топ-10 для детального анализа
            'doc_count': len(cluster_doc_ids),
            'doc_ids': cluster_doc_ids
        }

    return cluster_info


def plot_dendrogram(
    embeddings: np.ndarray,
    document_ids: List[int],
    method: str = 'average',
    metric: str = 'cosine',
    figsize: Tuple[int, int] = (12, 8)
) -> np.ndarray:
    """
    Строит дендрограмму для иерархической кластеризации.

    Args:
        embeddings: Массив эмбедингов документов
        document_ids: Список ID документов
        method: Метод linkage ('average', 'complete', 'ward', 'single')
        metric: Метрика расстояния ('cosine', 'euclidean', 'manhattan')
        figsize: Размер фигуры

    Returns:
        Linkage matrix для дальнейшего использования
    """
    print("\n" + "=" * 60)
    print("Построение дендрограммы...")
    print("=" * 60)
    print(f"Метод linkage: {method}")
    print(f"Метрика: {metric}")
    print(f"Количество документов: {len(document_ids)}")

    # Вычисляем linkage matrix
    linkage_matrix = linkage(embeddings, method=method, metric=metric)

    # Строим дендрограмму
    plt.figure(figsize=figsize)
    dendrogram(
        linkage_matrix,
        labels=[f"Doc {doc_id}" for doc_id in document_ids],
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title(f"Иерархическая кластеризация (method={method}, metric={metric})")
    plt.xlabel("ID документа")
    plt.ylabel("Расстояние")
    plt.tight_layout()

    # Сохраняем дендрограмму
    output_file = "../temporary_data/dendrogram.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Дендрограмма сохранена: {output_file}")

    plt.show()

    return linkage_matrix


def perform_clustering(
    embeddings: np.ndarray,
    n_clusters: int = None,
    distance_threshold: float = None,
    method: str = 'average',
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Выполняет иерархическую кластеризацию.

    Args:
        embeddings: Массив эмбедингов документов
        n_clusters: Количество кластеров (если задано)
        distance_threshold: Порог расстояния (если задано)
        method: Метод linkage
        metric: Метрика расстояния

    Returns:
        Массив меток кластеров
    """
    print("\n" + "=" * 60)
    print("Кластеризация документов...")
    print("=" * 60)

    if n_clusters is not None:
        print(f"Количество кластеров: {n_clusters}")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage=method
        )
    elif distance_threshold is not None:
        print(f"Порог расстояния: {distance_threshold}")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric=metric,
            linkage=method
        )
    else:
        raise ValueError("Необходимо указать либо n_clusters, либо distance_threshold")

    labels = clustering.fit_predict(embeddings)

    print(f"✓ Кластеризация завершена")
    print(f"  Найдено кластеров: {len(np.unique(labels))}")

    return labels


def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = 'cosine'
) -> Dict[str, float]:
    """
    Оценивает качество кластеризации с помощью различных метрик.

    Args:
        embeddings: Массив эмбедингов
        labels: Метки кластеров
        metric: Метрика расстояния

    Returns:
        Словарь с метриками качества
    """
    n_clusters = len(np.unique(labels))

    if n_clusters < 2 or n_clusters >= len(embeddings):
        print("⚠️ Невозможно вычислить метрики: недостаточно кластеров")
        return {}

    metrics = {}

    # Silhouette Score (от -1 до 1, чем больше - тем лучше)
    metrics['silhouette'] = silhouette_score(embeddings, labels, metric=metric)

    # Davies-Bouldin Index (чем меньше - тем лучше)
    metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)

    # Calinski-Harabasz Score (чем больше - тем лучше)
    metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)

    return metrics


def print_clustering_results(
    cluster_info: Dict[int, Dict],
    metrics: Dict[str, float] = None
) -> None:
    """
    Выводит результаты кластеризации в удобном формате.

    Args:
        cluster_info: Информация о кластерах
        metrics: Метрики качества кластеризации
    """
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ")
    print("=" * 60)

    if metrics:
        print("\nМетрики качества:")
        print(f"  Silhouette Score:        {metrics.get('silhouette', 'N/A'):.4f} (↑ лучше)")
        print(f"  Davies-Bouldin Index:    {metrics.get('davies_bouldin', 'N/A'):.4f} (↓ лучше)")
        print(f"  Calinski-Harabasz Score: {metrics.get('calinski_harabasz', 'N/A'):.2f} (↑ лучше)")

    print(f"\nНайдено кластеров: {len(cluster_info)}")

    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        print("\n" + "-" * 60)
        print(f"Кластер {cluster_id}: \"{info['name']}\"")
        print(f"  Количество документов: {info['doc_count']}")
        print(f"  Документы: {info['doc_ids']}")
        print(f"  Топ-5 тегов:")
        for i, (tag, weight) in enumerate(info['top_tags'][:5], 1):
            print(f"    {i}. {tag:40s} (вес: {weight:.3f})")


def plot_umap_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    document_ids: List[int],
    cluster_info: Dict[int, Dict],
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Строит 2D визуализацию кластеров с помощью UMAP.

    Args:
        embeddings: Массив эмбедингов документов
        labels: Метки кластеров
        document_ids: Список ID документов
        cluster_info: Информация о кластерах
        figsize: Размер фигуры
    """
    try:
        import umap
    except ImportError:
        print("⚠️ Модуль umap-learn не установлен. Пропускаем UMAP визуализацию.")
        print("   Установите: pip install umap-learn")
        return

    print("\n" + "=" * 60)
    print("Построение UMAP визуализации...")
    print("=" * 60)

    # Снижаем размерность до 2D
    reducer = umap.UMAP(
        n_neighbors=min(5, len(embeddings) - 1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # Строим график
    plt.figure(figsize=figsize)

    # Определяем цвета для каждого кластера
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        cluster_points = embedding_2d[mask]
        cluster_doc_ids = [document_ids[i] for i, m in enumerate(mask) if m]

        # Название кластера (первый тег)
        cluster_name = cluster_info[cluster_id]['name'].split(',')[0]

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[color],
            label=f"Cluster {cluster_id}: {cluster_name}",
            s=100,
            alpha=0.7,
            edgecolors='black'
        )

        # Подписываем точки ID документов
        for i, (x, y) in enumerate(cluster_points):
            doc_id = cluster_doc_ids[i]
            plt.annotate(
                f"Doc {doc_id}",
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

    plt.title("UMAP визуализация кластеров документов")
    plt.xlabel("UMAP компонента 1")
    plt.ylabel("UMAP компонента 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Сохраняем график
    output_file = "../temporary_data/umap_clusters.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ UMAP визуализация сохранена: {output_file}")

    plt.show()


def save_clustering_results(
    cluster_info: Dict[int, Dict],
    output_file: str
) -> None:
    """
    Сохраняет результаты кластеризации в JSON файл.

    Args:
        cluster_info: Информация о кластерах
        output_file: Путь к выходному файлу
    """
    # Конвертируем в сериализуемый формат
    serializable_info = {}
    for cluster_id, info in cluster_info.items():
        serializable_info[int(cluster_id)] = {
            'name': info['name'],
            'top_tags': [[tag, float(weight)] for tag, weight in info['top_tags']],
            'doc_count': info['doc_count'],
            'doc_ids': [int(doc_id) for doc_id in info['doc_ids']]
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_info, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Результаты сохранены: {output_file}")


def interactive_clustering(
    embeddings: np.ndarray,
    document_ids: List[int],
    doc_tags: Dict[int, Dict[str, float]],
    linkage_matrix: np.ndarray = None,
    method: str = 'average',
    metric: str = 'cosine'
) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    Интерактивный режим кластеризации с выбором параметров.

    Args:
        embeddings: Массив эмбедингов
        document_ids: Список ID документов
        doc_tags: Словарь тегов документов
        linkage_matrix: Linkage matrix (опционально)
        method: Метод linkage
        metric: Метрика расстояния

    Returns:
        Кортеж (метки кластеров, информация о кластерах)
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE CLUSTERING PARAMETERS")
    print("=" * 60)
    print("\nChoose clustering method:")
    print("  1. Specify number of clusters (n_clusters)")
    print("  2. Specify distance threshold (distance_threshold)")

    try:
        choice = sys.stdin.readline().strip()
    except Exception:
        choice = '1'

    labels = None

    if choice == '1':
        print("Enter number of clusters: ", end='', flush=True)
        try:
            n_clusters_str = sys.stdin.readline().strip()
            n_clusters = int(n_clusters_str)
        except Exception:
            print("Invalid input, using n_clusters=3 by default")
            n_clusters = 3

        labels = perform_clustering(
            embeddings=embeddings,
            n_clusters=n_clusters,
            method=method,
            metric=metric
        )
    elif choice == '2':
        print("Enter distance threshold: ", end='', flush=True)
        try:
            distance_threshold_str = sys.stdin.readline().strip()
            distance_threshold = float(distance_threshold_str)
        except Exception:
            print("Invalid input, using n_clusters=3 by default")
            labels = perform_clustering(
                embeddings=embeddings,
                n_clusters=3,
                method=method,
                metric=metric
            )
        else:
            labels = perform_clustering(
                embeddings=embeddings,
                distance_threshold=distance_threshold,
                method=method,
                metric=metric
            )
    else:
        print("Invalid choice, using n_clusters=3 by default")
        labels = perform_clustering(
            embeddings=embeddings,
            n_clusters=3,
            method=method,
            metric=metric
        )

    # Вычисляем названия кластеров
    cluster_info = compute_cluster_names_weighted_frequency(
        cluster_labels=labels,
        document_ids=document_ids,
        doc_tags=doc_tags,
        top_n=3
    )

    # Оцениваем качество кластеризации
    metrics = evaluate_clustering(embeddings, labels, metric=metric)

    # Выводим результаты
    print_clustering_results(cluster_info, metrics)

    return labels, cluster_info


def main():
    """
    Основная функция для запуска интерактивной кластеризации.
    """
    print("\n" + "=" * 60)
    print("ЭТАП 4: Иерархическая кластеризация документов")
    print("=" * 60)

    # Пути к файлам
    embeddings_file = "../temporary_data/document_embeddings.npz"
    tags_file = "../temporary_data/fake_data_tags.jsonl"
    output_file = "../temporary_data/clustering_results.json"

    # Шаг 1: Загружаем данные
    print("\nШаг 1: Загрузка данных...")
    document_ids, embeddings = load_document_embeddings_npz(embeddings_file)
    doc_tags = load_document_tags(tags_file)

    print(f"✓ Загружено {len(document_ids)} документов")
    print(f"✓ Загружено тегов для {len(doc_tags)} документов")

    # Шаг 2: Строим дендрограмму
    print("\nШаг 2: Построение дендрограммы...")
    linkage_matrix = plot_dendrogram(
        embeddings=embeddings,
        document_ids=document_ids,
        method='average',
        metric='cosine'
    )

    # Шаг 3: Интерактивная кластеризация
    labels, cluster_info = interactive_clustering(
        embeddings=embeddings,
        document_ids=document_ids,
        doc_tags=doc_tags,
        linkage_matrix=linkage_matrix,
        method='average',
        metric='cosine'
    )

    # Шаг 4: Визуализация UMAP
    print("\nШаг 4: Построение UMAP визуализации...")
    plot_umap_visualization(
        embeddings=embeddings,
        labels=labels,
        document_ids=document_ids,
        cluster_info=cluster_info
    )

    # Шаг 5: Сохраняем результаты
    save_clustering_results(cluster_info, output_file)

    # Финальный вывод: количество кластеров и их названия
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ КЛАСТЕРИЗАЦИИ")
    print("=" * 60)
    print(f"\nКоличество кластеров: {len(cluster_info)}\n")

    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        print(f"Кластер {cluster_id}: {info['name']}")

    print("\n" + "=" * 60)
    print("ЭТАП 4 завершён успешно!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
