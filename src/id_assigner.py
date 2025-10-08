"""
Модуль для присвоения уникальных ID документам в JSONL файле.
ЭТАП 0 пайплайна обработки данных.
"""

import json
import os
from typing import Set


def find_min_free_id(existing_ids: Set[int]) -> int:
    """
    Находит минимальный свободный ID (заполняет "дырки").

    Args:
        existing_ids: Множество существующих ID

    Returns:
        Минимальный свободный ID (int >= 0)
    """
    if not existing_ids:
        return 0

    # Сортируем ID и ищем первую "дырку"
    sorted_ids = sorted(existing_ids)

    for i, id_val in enumerate(sorted_ids):
        if i != id_val:
            return i

    # Если "дырок" нет, возвращаем следующий после максимального
    return sorted_ids[-1] + 1


def assign_ids_to_documents(filename: str) -> int:
    """
    Присваивает уникальные ID документам в JSONL файле, которые их не имеют.
    Модифицирует исходный файл.

    Args:
        filename: Путь к JSONL файлу

    Returns:
        Количество документов, которым был присвоен ID
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден")

    # ПЕРВЫЙ ПРОХОД: собираем все существующие ID
    print(f"Первый проход: сбор существующих ID из {filename}...")
    existing_ids: Set[int] = set()

    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                document = json.loads(line)
                if 'id' in document:
                    existing_ids.add(int(document['id']))
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON на строке {line_num}: {e}")
                continue

    print(f"Найдено {len(existing_ids)} документов с существующими ID")

    # ВТОРОЙ ПРОХОД: добавляем ID документам, у которых его нет
    print(f"Второй проход: добавление ID документам без ID...")

    temp_filename = filename + '.tmp'
    documents_with_new_id = 0

    with open(filename, 'r', encoding='utf-8') as f_in, \
         open(temp_filename, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                document = json.loads(line)

                # Если ID нет, добавляем его
                if 'id' not in document:
                    new_id = find_min_free_id(existing_ids)
                    document['id'] = new_id
                    existing_ids.add(new_id)
                    documents_with_new_id += 1
                    print(f"  Документ '{document.get('name', 'Unknown')[:50]}...' получил ID: {new_id}")

                # Записываем документ во временный файл
                f_out.write(json.dumps(document, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON на строке {line_num}: {e}")
                continue

    # Заменяем исходный файл временным
    os.replace(temp_filename, filename)
    print(f"\nГотово! Присвоено ID {documents_with_new_id} документам")
    print(f"Всего документов с ID: {len(existing_ids)}")

    return documents_with_new_id


if __name__ == "__main__":
    # Тестовый запуск
    test_file = "../temporary_data/fake_data.jsonl"

    print("="*60)
    print("ЭТАП 0: Присвоение уникальных ID документам")
    print("="*60)

    try:
        count = assign_ids_to_documents(test_file)
        print(f"\n✓ ЭТАП 0 завершён успешно!")
        print(f"  Обработано документов: {count}")
        print("\n→ Переход к ЭТАПУ 1: извлечение тегов с помощью LLM (TODO)")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
