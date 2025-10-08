"""
Тестовый скрипт для проверки работы генератора эмбедингов.
"""

import sys
import os

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings_generator import EmbeddingsGenerator


def test_single_embedding():
    """Тест генерации одного эмбединга."""
    print("\n" + "="*60)
    print("ТЕСТ 1: Генерация одного эмбединга")
    print("="*60)

    generator = EmbeddingsGenerator()

    text = "This is a patent about machine learning algorithms for data processing."
    print(f"Текст: {text}")

    embedding = generator.generate_embedding(text)
    print(f"Размерность эмбединга: {embedding.shape}")
    print(f"Первые 10 значений: {embedding[:10]}")
    print("✓ Тест пройден!")


def test_batch_embeddings():
    """Тест батч-генерации эмбедингов."""
    print("\n" + "="*60)
    print("ТЕСТ 2: Батч-генерация эмбедингов")
    print("="*60)

    generator = EmbeddingsGenerator()

    texts = [
        "A method for processing data using neural networks.",
        "System and method for image recognition using deep learning.",
        "Database optimization techniques for large-scale applications.",
        "Software implementation of quantum computing algorithms."
    ]

    print(f"Количество текстов: {len(texts)}")
    embeddings = generator.generate_embeddings_batch(texts, batch_size=2)

    print(f"Размер батча эмбедингов: {embeddings.shape}")
    print("✓ Тест пройден!")

    return generator, texts, embeddings


def test_cosine_similarity(generator, texts, embeddings):
    """Тест вычисления косинусного сходства."""
    print("\n" + "="*60)
    print("ТЕСТ 3: Косинусное сходство")
    print("="*60)

    # Сравниваем первый текст со всеми остальными
    print(f"\nБазовый текст: '{texts[0]}'")
    print("\nСходство с другими текстами:")

    for i in range(1, len(texts)):
        similarity = generator.cosine_similarity(embeddings[0], embeddings[i])
        print(f"  {i}. '{texts[i][:60]}...' - {similarity:.4f}")

    print("✓ Тест пройден!")


def test_save_and_load_jsonl(generator, texts, embeddings):
    """Тест сохранения и загрузки эмбедингов в JSONL."""
    print("\n" + "="*60)
    print("ТЕСТ 4: Сохранение и загрузка JSONL")
    print("="*60)

    filename = "../temporary_data/test_embeddings.jsonl"

    # Добавляем метаданные
    metadata = [
        {"type": "patent", "id": 1},
        {"type": "patent", "id": 2},
        {"type": "article", "id": 3},
        {"type": "software", "id": 4}
    ]

    # Сохранение
    print(f"Сохранение в файл: {filename}")
    generator.save_embeddings_to_jsonl(filename, texts, embeddings, metadata)
    print("✓ Файл сохранён!")

    # Загрузка
    print(f"\nЗагрузка из файла: {filename}")
    loaded_texts, loaded_embeddings, loaded_metadata = generator.load_embeddings_from_jsonl(filename)

    print(f"Загружено текстов: {len(loaded_texts)}")
    print(f"Размер эмбедингов: {loaded_embeddings.shape}")
    print(f"Загружено метаданных: {len(loaded_metadata)}")
    print(f"Пример метаданных: {loaded_metadata[0]}")
    print("✓ Тест пройден!")


def test_patent_similarity():
    """Тест на реальных примерах патентных текстов."""
    print("\n" + "="*60)
    print("ТЕСТ 5: Сходство патентных текстов")
    print("="*60)

    generator = EmbeddingsGenerator()

    # Примеры патентных текстов (симулированные)
    patent_texts = {
        "Patent A (ML)": "A machine learning system for automated classification of documents using neural networks and deep learning techniques.",
        "Patent B (ML similar)": "Method and apparatus for document classification using artificial neural networks and machine learning algorithms.",
        "Patent C (Image)": "Image processing system using convolutional neural networks for object detection and recognition.",
        "Patent D (Database)": "Database management system with optimized query processing and distributed storage architecture."
    }

    texts = list(patent_texts.values())
    labels = list(patent_texts.keys())

    print("Генерация эмбедингов для патентов...")
    embeddings = generator.generate_embeddings_batch(texts)

    print("\nМатрица сходства патентов:")
    print("-" * 60)

    for i, label_i in enumerate(labels):
        print(f"\n{label_i}:")
        for j, label_j in enumerate(labels):
            if i != j:
                similarity = generator.cosine_similarity(embeddings[i], embeddings[j])
                print(f"  vs {label_j}: {similarity:.4f}")

    print("\n✓ Тест пройден!")


def main():
    """Запуск всех тестов."""
    print("\n" + "="*60)
    print("ЗАПУСК ТЕСТОВ ГЕНЕРАТОРА ЭМБЕДИНГОВ")
    print("="*60)

    try:
        # Тест 1: Один эмбединг
        test_single_embedding()

        # Тест 2-3: Батч и косинусное сходство
        generator, texts, embeddings = test_batch_embeddings()
        test_cosine_similarity(generator, texts, embeddings)

        # Тест 4: Сохранение/загрузка
        test_save_and_load_jsonl(generator, texts, embeddings)

        # Тест 5: Патентные тексты
        test_patent_similarity()

        print("\n" + "="*60)
        print("ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ! ✓")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
