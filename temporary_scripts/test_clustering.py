"""
Тестовый скрипт для проверки иерархической кластеризации.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_clustering import (
    load_document_embeddings_npz,
    load_document_tags,
    plot_dendrogram,
    perform_clustering,
    compute_cluster_names_weighted_frequency,
    evaluate_clustering,
    print_clustering_results,
    save_clustering_results,
    plot_umap_visualization
)


def test_clustering_with_different_k():
    """
    Тестирует кластеризацию с разным количеством кластеров.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ: Кластеризация с разным количеством кластеров")
    print("=" * 60)

    # Загружаем данные
    embeddings_file = "../temporary_data/document_embeddings.npz"
    tags_file = "../temporary_data/fake_data_tags.jsonl"

    document_ids, embeddings = load_document_embeddings_npz(embeddings_file)
    doc_tags = load_document_tags(tags_file)

    print(f"\n✓ Загружено {len(document_ids)} документов")

    # Строим дендрограмму
    print("\n" + "=" * 60)
    print("Построение дендрограммы...")
    print("=" * 60)
    linkage_matrix = plot_dendrogram(
        embeddings=embeddings,
        document_ids=document_ids,
        method='average',
        metric='cosine'
    )

    # Тестируем разное количество кластеров
    for n_clusters in [2, 3, 4, 5]:
        print("\n" + "=" * 60)
        print(f"Тест с n_clusters = {n_clusters}")
        print("=" * 60)

        # Кластеризация
        labels = perform_clustering(
            embeddings=embeddings,
            n_clusters=n_clusters,
            method='average',
            metric='cosine'
        )

        # Вычисляем названия кластеров
        cluster_info = compute_cluster_names_weighted_frequency(
            cluster_labels=labels,
            document_ids=document_ids,
            doc_tags=doc_tags,
            top_n=3
        )

        # Оцениваем качество
        metrics = evaluate_clustering(embeddings, labels, metric='cosine')

        # Выводим результаты
        print_clustering_results(cluster_info, metrics)

        # Сохраняем результаты
        output_file = f"../temporary_data/clustering_results_k{n_clusters}.json"
        save_clustering_results(cluster_info, output_file)


def test_single_clustering():
    """
    Простой тест с фиксированным количеством кластеров.
    """
    print("\n" + "=" * 60)
    print("ТЕСТ: Кластеризация с 3 кластерами")
    print("=" * 60)

    # Загружаем данные
    embeddings_file = "../temporary_data/document_embeddings.npz"
    tags_file = "../temporary_data/fake_data_tags.jsonl"

    document_ids, embeddings = load_document_embeddings_npz(embeddings_file)
    doc_tags = load_document_tags(tags_file)

    print(f"\n✓ Загружено {len(document_ids)} документов")

    # Кластеризация
    labels = perform_clustering(
        embeddings=embeddings,
        n_clusters=3,
        method='average',
        metric='cosine'
    )

    # Вычисляем названия кластеров
    cluster_info = compute_cluster_names_weighted_frequency(
        cluster_labels=labels,
        document_ids=document_ids,
        doc_tags=doc_tags,
        top_n=3
    )

    # Оцениваем качество
    metrics = evaluate_clustering(embeddings, labels, metric='cosine')

    # Выводим результаты
    print_clustering_results(cluster_info, metrics)

    # Визуализация с UMAP
    plot_umap_visualization(
        embeddings=embeddings,
        labels=labels,
        document_ids=document_ids,
        cluster_info=cluster_info
    )

    # Сохраняем результаты
    output_file = "../temporary_data/clustering_results.json"
    save_clustering_results(cluster_info, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Тестирование кластеризации")
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'multiple'],
        help='Режим тестирования: single (3 кластера) или multiple (2-5 кластеров)'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'single':
            test_single_clustering()
        else:
            test_clustering_with_different_k()

        print("\n" + "=" * 60)
        print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
