"""
Модуль для генерации эмбедингов уникальных тегов.
ЭТАП 2 пайплайна обработки данных.
"""

import json
import os
import sys
import numpy as np
from typing import Set, List, Dict, Tuple

# Добавляем путь к src для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings_generator import EmbeddingsGenerator


def load_config(config_path: str = "../config.json") -> Dict:
    """
    Загружает конфигурацию из config.json.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Словарь с конфигурацией
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if "embedding_model" not in config:
        raise ValueError("Отсутствует поле 'embedding_model' в config.json")

    return config


def collect_unique_tags(tags_file: str) -> List[str]:
    """
    Собирает все уникальные теги из файла с тегами.

    Args:
        tags_file: Путь к JSONL файлу с тегами

    Returns:
        Список уникальных тегов (отсортированный)
    """
    if not os.path.exists(tags_file):
        raise FileNotFoundError(f"Файл с тегами не найден: {tags_file}")

    print("="*60)
    print("Сбор уникальных тегов...")
    print("="*60)

    unique_tags: Set[str] = set()

    with open(tags_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                tags_dict = data.get("tags", {})

                for tag in tags_dict.keys():
                    unique_tags.add(tag)

            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON на строке {line_num}: {e}")
                continue

    # Сортируем для детерминированности
    sorted_tags = sorted(unique_tags)

    print(f"✓ Найдено уникальных тегов: {len(sorted_tags)}")
    print(f"  Примеры тегов: {', '.join(sorted_tags[:5])}...")

    return sorted_tags


def generate_tag_embeddings(
    tags: List[str],
    embedding_model: str,
    batch_size: int = 16,
    add_prefix: bool = True
) -> np.ndarray:
    """
    Генерирует эмбединги для списка тегов.

    Args:
        tags: Список тегов
        embedding_model: Название модели эмбедингов
        batch_size: Размер батча (меньше для Mac M1, чтобы не переполнить память)
        add_prefix: Добавлять ли префикс "Scientific theme: " к тегам

    Returns:
        NumPy массив с эмбедингами (shape: [len(tags), embedding_dim])
    """
    print("\n" + "="*60)
    print("Генерация эмбедингов для тегов...")
    print("="*60)
    print(f"Модель: {embedding_model}")
    print(f"Количество тегов: {len(tags)}")
    print(f"Batch size: {batch_size}")
    print(f"Префикс 'Scientific theme:': {'Да' if add_prefix else 'Нет'}")
    print("="*60)

    # Инициализируем генератор эмбедингов
    generator = EmbeddingsGenerator(model_name=embedding_model)

    # Добавляем префикс к тегам, если требуется
    texts_for_embedding = tags
    if add_prefix:
        texts_for_embedding = [f"Scientific theme: {tag}" for tag in tags]
        print(f"\nПримеры текстов для эмбедингов:")
        for i in range(min(3, len(texts_for_embedding))):
            print(f"  {i+1}. '{texts_for_embedding[i]}'")

    # Генерируем эмбединги батчами
    embeddings = generator.generate_embeddings_batch(
        texts=texts_for_embedding,
        batch_size=batch_size,
        normalize=True
    )

    print(f"✓ Эмбединги сгенерированы")
    print(f"  Размерность: {embeddings.shape}")

    return embeddings


def save_tag_embeddings_npz(
    tags: List[str],
    embeddings: np.ndarray,
    output_file: str
) -> None:
    """
    Сохраняет теги и эмбединги в .npz файл.

    Args:
        tags: Список тегов
        embeddings: NumPy массив с эмбедингами
        output_file: Путь к выходному .npz файлу
    """
    if len(tags) != len(embeddings):
        raise ValueError(
            f"Количество тегов ({len(tags)}) не совпадает "
            f"с количеством эмбедингов ({len(embeddings)})"
        )

    print("\n" + "="*60)
    print("Сохранение эмбедингов...")
    print("="*60)
    print(f"Файл: {output_file}")
    print(f"Количество тегов: {len(tags)}")
    print(f"Размер эмбедингов: {embeddings.shape}")

    # Сохраняем в .npz формат
    np.savez_compressed(
        output_file,
        tags=np.array(tags, dtype=object),  # Массив строк
        embeddings=embeddings.astype(np.float32)  # float32 для экономии памяти
    )

    # Проверяем размер файла
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)

    print(f"✓ Эмбединги сохранены")
    print(f"  Размер файла: {file_size_mb:.2f} MB")
    print("="*60)


def load_tag_embeddings_npz(npz_file: str) -> Tuple[List[str], np.ndarray]:
    """
    Загружает теги и эмбединги из .npz файла.

    Args:
        npz_file: Путь к .npz файлу

    Returns:
        Кортеж (список тегов, массив эмбедингов)
    """
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Файл не найден: {npz_file}")

    data = np.load(npz_file, allow_pickle=True)
    tags = data['tags'].tolist()
    embeddings = data['embeddings']

    print(f"✓ Загружено {len(tags)} тегов с эмбедингами")
    print(f"  Размерность эмбедингов: {embeddings.shape}")

    return tags, embeddings


def create_tag_to_embedding_dict(
    tags: List[str],
    embeddings: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Создает словарь тег → эмбединг для быстрого доступа.

    Args:
        tags: Список тегов
        embeddings: Массив эмбедингов

    Returns:
        Словарь {тег: эмбединг}
    """
    return {tag: embeddings[i] for i, tag in enumerate(tags)}


def process_tags_to_embeddings(
    tags_file: str,
    output_file: str,
    config_path: str = "../config.json",
    batch_size: int = 16,
    add_prefix: bool = True
) -> int:
    """
    Полный пайплайн ЭТАПА 2: от тегов к эмбедингам.

    Args:
        tags_file: Путь к JSONL файлу с тегами
        output_file: Путь к выходному .npz файлу
        config_path: Путь к файлу конфигурации
        batch_size: Размер батча для генерации эмбедингов
        add_prefix: Добавлять ли префикс "Scientific theme: " к тегам

    Returns:
        Количество обработанных уникальных тегов
    """
    print("\n" + "="*60)
    print("ЭТАП 2: Генерация эмбедингов для уникальных тегов")
    print("="*60)

    # Загружаем конфигурацию
    config = load_config(config_path)
    embedding_model = config["embedding_model"]

    # Шаг 1: Собираем уникальные теги
    unique_tags = collect_unique_tags(tags_file)

    if len(unique_tags) == 0:
        print("⚠️ Не найдено ни одного тега!")
        return 0

    # Шаг 2: Генерируем эмбединги
    embeddings = generate_tag_embeddings(
        tags=unique_tags,
        embedding_model=embedding_model,
        batch_size=batch_size,
        add_prefix=add_prefix
    )

    # Шаг 3: Сохраняем в .npz файл
    save_tag_embeddings_npz(
        tags=unique_tags,
        embeddings=embeddings,
        output_file=output_file
    )

    print("\n" + "="*60)
    print("ЭТАП 2 завершён успешно!")
    print("="*60)
    print(f"Обработано уникальных тегов: {len(unique_tags)}")
    print(f"Файл с эмбедингами: {output_file}")
    print("="*60)

    # Опционально: выводим статистику
    print("\nПримеры тегов с эмбедингами:")
    for i in range(min(3, len(unique_tags))):
        tag = unique_tags[i]
        emb_preview = embeddings[i][:5]
        print(f"  '{tag}': [{emb_preview[0]:.4f}, {emb_preview[1]:.4f}, ...]")

    return len(unique_tags)


if __name__ == "__main__":
    # Тестовый запуск
    tags_file = "../temporary_data/fake_data_tags.jsonl"
    output_file = "../temporary_data/unique_tag_embeddings.npz"

    try:
        count = process_tags_to_embeddings(
            tags_file=tags_file,
            output_file=output_file,
            batch_size=16,  # Маленький batch для Mac M1
            add_prefix=True  # Добавляем префикс "Scientific theme:"
        )
        print(f"\n✓ ЭТАП 2 завершён успешно! Обработано тегов: {count}")

        # Проверяем загрузку
        print("\n" + "="*60)
        print("Проверка загрузки эмбедингов...")
        print("="*60)
        tags, embeddings = load_tag_embeddings_npz(output_file)
        print(f"✓ Успешно загружено {len(tags)} тегов")

        # Создаем словарь для демонстрации
        tag_to_emb = create_tag_to_embedding_dict(tags, embeddings)
        print(f"✓ Создан словарь тег→эмбединг с {len(tag_to_emb)} элементами")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
