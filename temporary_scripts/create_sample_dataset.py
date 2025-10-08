#!/usr/bin/env python3
"""
Скрипт для создания тестовой выборки из N документов.
"""

import json
import random


def create_sample_dataset(input_file: str, output_file: str, sample_size: int = 50, seed: int = 42):
    """
    Создаёт тестовую выборку из N случайных документов.

    :param input_file: Путь к исходному датасету (JSONL)
    :param output_file: Путь для сохранения выборки (JSONL)
    :param sample_size: Количество документов в выборке
    :param seed: Seed для воспроизводимости случайной выборки
    """
    random.seed(seed)

    # Загружаем все документы
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    print(f"Всего документов в датасете: {len(documents)}")

    # Создаём случайную выборку
    if sample_size > len(documents):
        print(f"Предупреждение: запрошено {sample_size} документов, но доступно только {len(documents)}")
        sample_size = len(documents)

    sample = random.sample(documents, sample_size)

    # Сохраняем выборку
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in sample:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"Создана выборка из {len(sample)} документов: {output_file}")


if __name__ == "__main__":
    create_sample_dataset(
        input_file='data/fake_data.jsonl',
        output_file='temporary_data/sample_50_documents.jsonl',
        sample_size=50
    )
