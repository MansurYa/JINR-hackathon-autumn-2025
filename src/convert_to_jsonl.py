"""
Конвертирует многострочный JSON в JSONL формат.
"""

import json
import sys

def convert_to_jsonl(input_file: str, output_file: str):
    """
    Читает файл с многострочными JSON объектами и конвертирует в JSONL.
    """
    print(f"Конвертация {input_file} → {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Разбиваем по объектам (предполагаем, что объекты разделены "}\n{")
    # Добавляем обратно скобки
    objects = content.strip().split('}\n{')

    count = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        for i, obj_str in enumerate(objects):
            # Восстанавливаем скобки
            if i == 0:
                obj_str = obj_str + '}'
            elif i == len(objects) - 1:
                obj_str = '{' + obj_str
            else:
                obj_str = '{' + obj_str + '}'

            try:
                # Парсим и записываем в одну строку
                obj = json.loads(obj_str)
                out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                count += 1
                if count % 100 == 0:
                    print(f"  Обработано: {count} документов...")
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка на объекте {i}: {e}")
                continue

    print(f"✓ Конвертировано {count} документов")

if __name__ == "__main__":
    input_file = "../data/full_dataset.jsonl"
    output_file = "../data/full_dataset_clean.jsonl"

    convert_to_jsonl(input_file, output_file)
