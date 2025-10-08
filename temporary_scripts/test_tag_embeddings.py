"""
Тестовый скрипт для проверки загрузки и работы с эмбедингами тегов.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tag_embeddings_generator import load_tag_embeddings_npz, create_tag_to_embedding_dict
import numpy as np


def test_load_embeddings():
    """Тест загрузки эмбедингов"""
    print("="*60)
    print("ТЕСТ: Загрузка эмбедингов тегов")
    print("="*60)

    npz_file = "../temporary_data/unique_tag_embeddings.npz"

    # Загружаем
    tags, embeddings = load_tag_embeddings_npz(npz_file)

    print(f"\n✓ Загружено тегов: {len(tags)}")
    print(f"✓ Размерность эмбедингов: {embeddings.shape}")
    print(f"✓ Тип данных: {embeddings.dtype}")

    # Создаем словарь
    tag_to_emb = create_tag_to_embedding_dict(tags, embeddings)

    print(f"\n✓ Создан словарь тег→эмбединг")
    print(f"\nПримеры тегов:")
    for i, tag in enumerate(tags[:5]):
        print(f"  {i+1}. {tag}")

    return tags, embeddings, tag_to_emb


def test_cosine_similarity(tag_to_emb):
    """Тест вычисления косинусного сходства между тегами"""
    print("\n" + "="*60)
    print("ТЕСТ: Косинусное сходство между тегами")
    print("="*60)

    # Выбираем несколько тегов для сравнения
    test_tags = [
        "Machine learning",
        "Artificial Intelligence",
        "High energy physics",
        "Nuclear reactions"
    ]

    # Фильтруем только те теги, которые есть в словаре
    available_tags = [tag for tag in test_tags if tag in tag_to_emb]

    if len(available_tags) < 2:
        print("⚠️ Недостаточно тегов для сравнения")
        return

    print(f"\nСравниваем теги:")
    for tag in available_tags:
        print(f"  - {tag}")

    print(f"\nМатрица косинусного сходства:")
    print("-" * 60)

    for tag1 in available_tags:
        emb1 = tag_to_emb[tag1]
        similarities = []

        for tag2 in available_tags:
            emb2 = tag_to_emb[tag2]
            # Косинусное сходство (эмбединги уже нормализованы)
            similarity = float(np.dot(emb1, emb2))
            similarities.append(similarity)

        print(f"{tag1:30s}: " + " ".join([f"{s:.3f}" for s in similarities]))


def test_linear_combination(tags, embeddings, tag_to_emb):
    """
    Тест линейной комбинации эмбедингов (как будет использоваться на ЭТАПЕ 3).
    """
    print("\n" + "="*60)
    print("ТЕСТ: Линейная комбинация эмбедингов")
    print("="*60)

    # Симулируем теги одного документа с весами
    doc_tags = {
        "Machine learning": 0.6,
        "Artificial Intelligence": 0.3,
        "Data Cleaning": 0.1
    }

    print(f"\nДокумент с тегами:")
    for tag, weight in doc_tags.items():
        print(f"  {tag}: {weight}")

    # Вычисляем линейную комбинацию
    combined_embedding = np.zeros(embeddings.shape[1], dtype=np.float32)

    for tag, weight in doc_tags.items():
        if tag in tag_to_emb:
            combined_embedding += tag_to_emb[tag] * weight

    # Нормализуем результат
    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

    print(f"\n✓ Комбинированный эмбединг вычислен")
    print(f"  Размерность: {combined_embedding.shape}")
    print(f"  Норма: {np.linalg.norm(combined_embedding):.4f}")
    print(f"  Первые 5 значений: {combined_embedding[:5]}")

    # Находим наиболее похожие теги
    print(f"\nНаиболее похожие исходные теги:")
    similarities = []
    for tag in tags:
        if tag in tag_to_emb:
            sim = float(np.dot(combined_embedding, tag_to_emb[tag]))
            similarities.append((tag, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    for tag, sim in similarities[:5]:
        print(f"  {tag:40s}: {sim:.4f}")


def main():
    """Запуск всех тестов"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЭМБЕДИНГОВ ТЕГОВ")
    print("="*60)

    try:
        # Тест 1: Загрузка
        tags, embeddings, tag_to_emb = test_load_embeddings()

        # Тест 2: Косинусное сходство
        test_cosine_similarity(tag_to_emb)

        # Тест 3: Линейная комбинация
        test_linear_combination(tags, embeddings, tag_to_emb)

        print("\n" + "="*60)
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ! ✓")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
