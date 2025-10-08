"""
Модуль для генерации эмбедингов документов на основе тегов с весами.
ЭТАП 3 пайплайна обработки данных.
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Tuple

# Добавляем путь к src для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_embeddings_generator import load_tag_embeddings_npz, create_tag_to_embedding_dict


def load_document_tags(tags_file: str) -> List[Dict]:
    """
    Загружает теги документов из JSONL файла.

    Args:
        tags_file: Путь к файлу с тегами документов

    Returns:
        Список словарей с document_id и tags
    """
    if not os.path.exists(tags_file):
        raise FileNotFoundError(f"Файл с тегами не найден: {tags_file}")

    documents_tags = []

    with open(tags_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                documents_tags.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON на строке {line_num}: {e}")
                continue

    return documents_tags


def compute_document_embedding(
    tags_with_weights: Dict[str, float],
    tag_to_embedding: Dict[str, np.ndarray],
    embedding_dim: int
) -> np.ndarray:
    """
    Вычисляет эмбединг документа как взвешенную сумму эмбедингов тегов.

    Args:
        tags_with_weights: Словарь {тег: вес}
        tag_to_embedding: Словарь {тег: эмбединг}
        embedding_dim: Размерность эмбединга

    Returns:
        Нормализованный эмбединг документа (numpy array)
    """
    # Инициализируем нулевой вектор
    document_embedding = np.zeros(embedding_dim, dtype=np.float32)

    # Считаем взвешенную сумму
    for tag, weight in tags_with_weights.items():
        if tag in tag_to_embedding:
            document_embedding += tag_to_embedding[tag] * weight
        else:
            print(f"⚠️ Тег '{tag}' не найден в предпосчитанных эмбедингах")

    # Нормализуем результат
    norm = np.linalg.norm(document_embedding)
    if norm > 0:
        document_embedding = document_embedding / norm

    return document_embedding


def generate_document_embeddings(
    tags_file: str,
    tag_embeddings_file: str
) -> Tuple[List[int], np.ndarray]:
    """
    Генерирует эмбединги для всех документов.

    Args:
        tags_file: Путь к файлу с тегами документов (fake_data_tags.jsonl)
        tag_embeddings_file: Путь к файлу с эмбедингами тегов (.npz)

    Returns:
        Кортеж (список document_id, массив эмбедингов)
    """
    print("=" * 60)
    print("ЭТАП 3: Генерация эмбедингов документов")
    print("=" * 60)

    # Шаг 1: Загружаем предпосчитанные эмбединги тегов
    print("\nШаг 1: Загрузка эмбедингов тегов...")
    tags, tag_embeddings = load_tag_embeddings_npz(tag_embeddings_file)
    tag_to_emb = create_tag_to_embedding_dict(tags, tag_embeddings)
    embedding_dim = tag_embeddings.shape[1]

    print(f"✓ Загружено {len(tags)} тегов с эмбедингами")
    print(f"  Размерность эмбедингов: {embedding_dim}")

    # Шаг 2: Загружаем теги документов
    print("\nШаг 2: Загрузка тегов документов...")
    documents_tags = load_document_tags(tags_file)
    print(f"✓ Загружено {len(documents_tags)} документов с тегами")

    # Шаг 3: Вычисляем эмбединги документов
    print("\nШаг 3: Вычисление эмбедингов документов...")
    document_ids = []
    document_embeddings = []

    for doc_data in documents_tags:
        doc_id = doc_data["document_id"]
        tags_with_weights = doc_data["tags"]

        # Вычисляем эмбединг документа
        doc_embedding = compute_document_embedding(
            tags_with_weights=tags_with_weights,
            tag_to_embedding=tag_to_emb,
            embedding_dim=embedding_dim
        )

        document_ids.append(doc_id)
        document_embeddings.append(doc_embedding)

    # Преобразуем в numpy array
    document_embeddings = np.array(document_embeddings, dtype=np.float32)

    print(f"✓ Вычислено {len(document_ids)} эмбедингов документов")
    print(f"  Размерность: {document_embeddings.shape}")

    return document_ids, document_embeddings


def save_document_embeddings_npz(
    document_ids: List[int],
    embeddings: np.ndarray,
    output_file: str
) -> None:
    """
    Сохраняет ID документов и их эмбединги в .npz файл.

    Args:
        document_ids: Список ID документов
        embeddings: Массив эмбедингов
        output_file: Путь к выходному .npz файлу
    """
    if len(document_ids) != len(embeddings):
        raise ValueError(
            f"Количество ID ({len(document_ids)}) не совпадает "
            f"с количеством эмбедингов ({len(embeddings)})"
        )

    print("\n" + "=" * 60)
    print("Сохранение эмбедингов документов...")
    print("=" * 60)
    print(f"Файл: {output_file}")
    print(f"Количество документов: {len(document_ids)}")
    print(f"Размер эмбедингов: {embeddings.shape}")

    # Сохраняем в .npz формат
    np.savez_compressed(
        output_file,
        document_ids=np.array(document_ids, dtype=np.int32),
        embeddings=embeddings.astype(np.float32)
    )

    # Проверяем размер файла
    file_size = os.path.getsize(output_file)
    file_size_kb = file_size / 1024

    print(f"✓ Эмбединги документов сохранены")
    print(f"  Размер файла: {file_size_kb:.2f} KB")
    print("=" * 60)


def load_document_embeddings_npz(npz_file: str) -> Tuple[List[int], np.ndarray]:
    """
    Загружает ID документов и их эмбединги из .npz файла.

    Args:
        npz_file: Путь к .npz файлу

    Returns:
        Кортеж (список document_ids, массив эмбедингов)
    """
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Файл не найден: {npz_file}")

    data = np.load(npz_file)
    document_ids = data['document_ids'].tolist()
    embeddings = data['embeddings']

    print(f"✓ Загружено {len(document_ids)} документов с эмбедингами")
    print(f"  Размерность эмбедингов: {embeddings.shape}")

    return document_ids, embeddings


def create_id_to_embedding_dict(
    document_ids: List[int],
    embeddings: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Создает словарь document_id → эмбединг для быстрого доступа.

    Args:
        document_ids: Список ID документов
        embeddings: Массив эмбедингов

    Returns:
        Словарь {document_id: эмбединг}
    """
    return {doc_id: embeddings[i] for i, doc_id in enumerate(document_ids)}


def process_documents_to_embeddings(
    tags_file: str,
    tag_embeddings_file: str,
    output_file: str
) -> int:
    """
    Полный пайплайн ЭТАПА 3: от тегов документов к эмбедингам документов.

    Args:
        tags_file: Путь к файлу с тегами документов
        tag_embeddings_file: Путь к файлу с эмбедингами тегов
        output_file: Путь к выходному .npz файлу

    Returns:
        Количество обработанных документов
    """
    # Генерируем эмбединги документов
    document_ids, embeddings = generate_document_embeddings(
        tags_file=tags_file,
        tag_embeddings_file=tag_embeddings_file
    )

    if len(document_ids) == 0:
        print("⚠️ Не найдено ни одного документа для обработки!")
        return 0

    # Сохраняем в .npz файл
    save_document_embeddings_npz(
        document_ids=document_ids,
        embeddings=embeddings,
        output_file=output_file
    )

    print("\n" + "=" * 60)
    print("ЭТАП 3 завершён успешно!")
    print("=" * 60)
    print(f"Обработано документов: {len(document_ids)}")
    print(f"Файл с эмбедингами: {output_file}")
    print("=" * 60)

    # Опционально: выводим статистику
    print("\nПримеры document_id → эмбединг:")
    for i in range(min(3, len(document_ids))):
        doc_id = document_ids[i]
        emb_preview = embeddings[i][:5]
        print(f"  Document {doc_id}: [{emb_preview[0]:.4f}, {emb_preview[1]:.4f}, ...]")

    return len(document_ids)


if __name__ == "__main__":
    # Тестовый запуск
    tags_file = "../temporary_data/fake_data_tags.jsonl"
    tag_embeddings_file = "../temporary_data/unique_tag_embeddings.npz"
    output_file = "../temporary_data/document_embeddings.npz"

    try:
        count = process_documents_to_embeddings(
            tags_file=tags_file,
            tag_embeddings_file=tag_embeddings_file,
            output_file=output_file
        )
        print(f"\n✓ ЭТАП 3 завершён успешно! Обработано документов: {count}")

        # Проверяем загрузку
        print("\n" + "=" * 60)
        print("Проверка загрузки эмбедингов документов...")
        print("=" * 60)
        doc_ids, doc_embeddings = load_document_embeddings_npz(output_file)
        print(f"✓ Успешно загружено {len(doc_ids)} документов")

        # Создаем словарь для демонстрации
        id_to_emb = create_id_to_embedding_dict(doc_ids, doc_embeddings)
        print(f"✓ Создан словарь document_id→эмбединг с {len(id_to_emb)} элементами")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
