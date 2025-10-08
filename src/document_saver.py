"""
Модуль для сохранения документов в формате JSONL.
"""

import json
import os
from typing import Optional


VALID_DOCUMENT_TYPES = {"article", "patent", "databases", "software"}


def save_document(
    filename: str,
    name: str,
    type_of_document: str,
    date: str,
    authors: str,
    identifier: str,
    text_of_document: str
) -> None:
    """
    Сохраняет документ в JSONL файл.

    :param filename: Имя файла для записи (должно заканчиваться на .jsonl)
    :param name: Название документа
    :param type_of_document: Тип документа (article, patent, databases, software)
    :param date: Дата в формате YYYY.MM.DD
    :param authors: Список авторов в любом виде
    :param identifier: ДОИ или другой идентификатор
    :param text_of_document: Текст документа
    :raises ValueError: если тип документа неверный или filename не заканчивается на .jsonl
    """
    # Проверка расширения файла
    if not filename.endswith('.jsonl'):
        raise ValueError("Имя файла должно заканчиваться на .jsonl")

    # Проверка типа документа
    if type_of_document not in VALID_DOCUMENT_TYPES:
        raise ValueError(
            f"Неверный тип документа: {type_of_document}. "
            f"Допустимые типы: {', '.join(VALID_DOCUMENT_TYPES)}"
        )

    # Создание объекта документа
    document = {
        "name": name,
        "type_of_document": type_of_document,
        "date": date,
        "authors": authors,
        "identifier": identifier,
        "text_of_document": text_of_document
    }

    # Сохранение в файл (добавление в конец)
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(document, ensure_ascii=False) + '\n')


def save_documents_batch(
    filename: str,
    documents: list[dict]
) -> None:
    """
    Сохраняет пакет документов в JSONL файл.

    :param filename: Имя файла для записи (должно заканчиваться на .jsonl)
    :param documents: Список словарей с документами
    :raises ValueError: если filename не заканчивается на .jsonl или тип документа неверный
    """
    # Проверка расширения файла
    if not filename.endswith('.jsonl'):
        raise ValueError("Имя файла должно заканчиваться на .jsonl")

    # Проверка всех документов перед записью
    for i, doc in enumerate(documents):
        if doc.get("type_of_document") not in VALID_DOCUMENT_TYPES:
            raise ValueError(
                f"Документ #{i}: неверный тип документа '{doc.get('type_of_document')}'. "
                f"Допустимые типы: {', '.join(VALID_DOCUMENT_TYPES)}"
            )

    # Сохранение всех документов
    with open(filename, 'a', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
