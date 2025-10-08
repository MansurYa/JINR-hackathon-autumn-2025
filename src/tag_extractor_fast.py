"""
Многопоточная версия модуля для извлечения научных тегов из документов с использованием LLM.
ЭТАП 1 пайплайна обработки данных (параллельная версия).
"""

import json
import os
import re
from typing import Dict, Optional
from openrouter_client import OpenRouterClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


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

    required_fields = ["api_key", "model_name", "temperature", "max_response_tokens"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Отсутствует обязательное поле в config.json: {field}")

    return config


def load_prompt(prompt_path: str = "../prompt.txt") -> str:
    """
    Загружает системный промпт из файла.

    Args:
        prompt_path: Путь к файлу с промптом

    Returns:
        Системный промпт
    """
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Файл с промптом не найден: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def is_english(text: str) -> bool:
    """
    Проверяет, является ли текст английским (простая эвристика).

    Args:
        text: Текст для проверки

    Returns:
        True, если текст на английском
    """
    # Проверяем наличие кириллицы
    cyrillic_pattern = re.compile('[а-яА-ЯёЁ]')
    return not bool(cyrillic_pattern.search(text))


def parse_tags_and_weights(response: str) -> Optional[Dict[str, float]]:
    """
    Парсит ответ LLM и извлекает теги с весами.

    Ожидаемый формат: "Tag1: 0.6, Tag2: 0.2, Tag3: 0.15, Tag4: 0.05"

    Args:
        response: Ответ от LLM

    Returns:
        Словарь {тег: вес} или None при ошибке
    """
    try:
        # Удаляем лишние пробелы и переносы строк
        response = response.strip()

        # Разделяем по запятой
        tag_weight_pairs = response.split(',')

        tags_dict = {}
        for pair in tag_weight_pairs:
            pair = pair.strip()
            if ':' not in pair:
                continue

            # Разделяем на тег и вес
            parts = pair.split(':')
            if len(parts) != 2:
                continue

            tag = parts[0].strip()
            weight_str = parts[1].strip()

            # Парсим вес
            try:
                weight = float(weight_str)
            except ValueError:
                print(f"⚠️ Не удалось распарсить вес: {weight_str}")
                continue

            # Проверяем, что тег на английском
            if not is_english(tag):
                print(f"⚠️ Тег не на английском языке (пропускаем): {tag}")
                continue

            # Проверяем вес
            if weight <= 0.0 or weight > 1.0:
                print(f"⚠️ Некорректный вес для тега '{tag}': {weight}")
                continue

            tags_dict[tag] = weight

        if not tags_dict:
            print("⚠️ Не удалось извлечь ни одного валидного тега")
            return None

        # Пересчитываем веса, чтобы сумма была 1.0
        total_weight = sum(tags_dict.values())
        if abs(total_weight - 1.0) > 0.01:  # Допускаем небольшую погрешность
            print(f"⚠️ Сумма весов не равна 1.0 ({total_weight}), пересчитываем...")
            tags_dict = {tag: weight / total_weight for tag, weight in tags_dict.items()}

        return tags_dict

    except Exception as e:
        print(f"❌ Ошибка парсинга ответа LLM: {e}")
        print(f"Ответ: {response}")
        return None


def process_single_document(document: Dict, system_prompt: str, client: OpenRouterClient, line_num: int) -> Optional[Dict]:
    """
    Обрабатывает один документ (для использования в потоке).

    Args:
        document: Документ из JSONL
        system_prompt: Системный промпт для LLM
        client: Клиент OpenRouter API
        line_num: Номер строки в файле

    Returns:
        Словарь с результатом или None при ошибке
    """
    try:
        # Проверяем наличие необходимых полей
        if 'id' not in document:
            print(f"⚠️ Документ на строке {line_num} не имеет ID, пропускаем")
            return None

        doc_id = document['id']
        doc_name = document.get('name', 'Unknown')
        doc_text = document.get('text_of_document', '')

        if not doc_text:
            print(f"⚠️ Документ ID={doc_id} не имеет текста, пропускаем")
            return None

        print(f"\n📄 Обработка документа ID={doc_id}: '{doc_name[:50]}...'")

        # Формируем запрос к LLM
        user_message = f"Document title: {doc_name}\n\nDocument text:\n{doc_text}"

        # Отправляем запрос к LLM
        try:
            response = client.call_api(system_prompt, user_message)
        except Exception as e:
            print(f"❌ Ошибка при обращении к LLM для документа ID={doc_id}: {e}")
            return None

        # Парсим теги и веса
        tags_dict = parse_tags_and_weights(response)

        if tags_dict is None or len(tags_dict) == 0:
            print(f"⚠️ Не удалось извлечь теги для документа ID={doc_id}, пропускаем")
            return None

        # Формируем результат
        result = {
            "document_id": doc_id,
            "tags": tags_dict
        }

        print(f"✓ Извлечено {len(tags_dict)} тегов для документа ID={doc_id}")
        print(f"  Теги: {', '.join([f'{tag}: {weight:.2f}' for tag, weight in list(tags_dict.items())[:3]])}...")

        return result

    except Exception as e:
        print(f"❌ Ошибка при обработке документа на строке {line_num}: {e}")
        return None


def extract_tags_from_documents(
    input_file: str,
    output_file: str,
    config_path: str = "../config.json",
    prompt_path: str = "../prompt.txt",
    num_threads: int = 32
) -> int:
    """
    Извлекает теги из документов в JSONL файле и сохраняет результаты в отдельный файл (многопоточная версия).

    Args:
        input_file: Путь к входному JSONL файлу с документами
        output_file: Путь к выходному JSONL файлу с тегами
        config_path: Путь к файлу конфигурации
        prompt_path: Путь к файлу с системным промптом
        num_threads: Количество потоков для параллельной обработки

    Returns:
        Количество успешно обработанных документов
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Входной файл не найден: {input_file}")

    # Загружаем конфигурацию и промпт
    config = load_config(config_path)
    system_prompt = load_prompt(prompt_path)

    # Переопределяем количество потоков из конфига, если указано
    if "num_threads" in config:
        num_threads = config["num_threads"]

    # Создаем клиента OpenRouter
    client = OpenRouterClient(
        api_key=config["api_key"],
        model_name=config["model_name"],
        max_response_tokens=config["max_response_tokens"],
        temperature=config["temperature"]
    )

    print("="*60)
    print("ЭТАП 1: Извлечение тегов из документов с помощью LLM (FAST)")
    print("="*60)
    print(f"Модель: {config['model_name']}")
    print(f"Температура: {config['temperature']}")
    print(f"Количество потоков: {num_threads}")
    print(f"Входной файл: {input_file}")
    print(f"Выходной файл: {output_file}")
    print("="*60)

    # Загружаем все документы из файла
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
                documents.append((document, line_num))
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка парсинга JSON на строке {line_num}: {e}")

    total_documents = len(documents)
    print(f"\n📚 Загружено документов: {total_documents}")

    # Счетчики
    processed_documents = 0
    skipped_documents = 0
    write_lock = Lock()

    # Открываем выходной файл для записи
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Многопоточная обработка
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Отправляем задачи в пул потоков
            futures = {
                executor.submit(process_single_document, doc, system_prompt, client, line_num): (doc, line_num)
                for doc, line_num in documents
            }

            # Обрабатываем результаты по мере готовности
            for future in as_completed(futures):
                result = future.result()

                if result is not None:
                    # Потокобезопасная запись в файл
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                    processed_documents += 1
                else:
                    skipped_documents += 1

    print("\n" + "="*60)
    print("ЭТАП 1 завершён!")
    print("="*60)
    print(f"Всего документов: {total_documents}")
    print(f"Успешно обработано: {processed_documents}")
    print(f"Пропущено: {skipped_documents}")
    print(f"Результаты сохранены в: {output_file}")
    print("="*60)

    return processed_documents


if __name__ == "__main__":
    # Тестовый запуск
    input_file = "../temporary_data/fake_data.jsonl"
    output_file = "../temporary_data/fake_data_tags.jsonl"

    try:
        count = extract_tags_from_documents(input_file, output_file)
        print(f"\n✓ ЭТАП 1 завершён успешно! Обработано документов: {count}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
