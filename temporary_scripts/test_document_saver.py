import sys
import os
import time

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.document_saver import save_document  # теперь это сработает

FILENAME = "test_output.jsonl"
NUM_LINES = 1_000_000


def main():
    start_time = time.time()

    for _ in range(NUM_LINES):
        save_document(
            filename=FILENAME,
            name="Анализ рынка технологий 2024",
            type_of_document="article",
            date="2024.01.20",
            authors="Сидоров П.К., Nicolas Aquire , П.В. Пушкин",
            identifier="10.1016/j.cell.2020.01.001",
            text_of_document="В этой статье мы..."
        )

    elapsed = time.time() - start_time
    print(f"✅ Создан файл '{FILENAME}' с {NUM_LINES} строками.")
    print(f"⏱ Время выполнения: {elapsed:.2f} сек.")


if __name__ == "__main__":
    main()
