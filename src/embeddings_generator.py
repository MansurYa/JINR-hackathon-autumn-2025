"""
Модуль для генерации эмбедингов текстов с использованием модели BAAI/bge-base-en-v1.5.
Оптимизирован для работы на Mac M1 с использованием MPS (Metal Performance Shaders).
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Union, List


class EmbeddingsGenerator:
    """
    Класс для генерации эмбедингов текстов с использованием предобученной модели.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):  # "BAAI/bge-large-en-v1.5"
        """
        Инициализация генератора эмбедингов.

        :param model_name: Название модели из HuggingFace
        """
        self.model_name = model_name
        self.device = self._get_device()
        print(f"Используется устройство: {self.device}")

        # Загрузка модели
        print(f"Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Модель успешно загружена!")

    def _get_device(self) -> str:
        """
        Определяет оптимальное устройство для вычислений.

        :return: Название устройства ('mps', 'cuda', или 'cpu')
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Генерирует эмбединг для одного текста.

        :param text: Входной текст
        :param normalize: Нормализовать ли эмбединг (для косинусного сходства)
        :return: Numpy массив с эмбедингом
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Генерирует эмбединги для списка текстов (батч-обработка).

        :param texts: Список текстов
        :param batch_size: Размер батча для обработки
        :param normalize: Нормализовать ли эмбединги
        :return: Numpy массив с эмбедингами (shape: [len(texts), embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True
        )
        return embeddings

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя эмбедингами.

        :param emb1: Первый эмбединг
        :param emb2: Второй эмбединг
        :return: Косинусное сходство (от -1 до 1)
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def save_embeddings_to_jsonl(
        self,
        filename: str,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[dict] = None
    ) -> None:
        """
        Сохраняет эмбединги в JSONL файл.

        :param filename: Имя файла для сохранения
        :param texts: Список текстов
        :param embeddings: Массив эмбедингов
        :param metadata: Дополнительные метаданные для каждого текста (опционально)
        """
        if not filename.endswith('.jsonl'):
            raise ValueError("Имя файла должно заканчиваться на .jsonl")

        with open(filename, 'w', encoding='utf-8') as f:
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                record = {
                    "text": text,
                    "embedding": embedding.tolist()
                }

                # Добавление метаданных, если они есть
                if metadata and i < len(metadata):
                    record.update(metadata[i])

                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def load_embeddings_from_jsonl(self, filename: str) -> tuple:
        """
        Загружает эмбединги из JSONL файла.

        :param filename: Имя файла для загрузки
        :return: Кортеж (список текстов, массив эмбедингов, список метаданных)
        """
        texts = []
        embeddings = []
        metadata = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                texts.append(record['text'])
                embeddings.append(record['embedding'])

                # Извлечение метаданных
                meta = {k: v for k, v in record.items() if k not in ['text', 'embedding']}
                metadata.append(meta)

        return texts, np.array(embeddings), metadata


def generate_embedding_for_text(text: str, model_name: str = "BAAI/bge-base-en-v1.5") -> np.ndarray:
    """
    Простая функция для быстрой генерации эмбединга одного текста.

    :param text: Входной текст
    :param model_name: Название модели
    :return: Numpy массив с эмбедингом
    """
    generator = EmbeddingsGenerator(model_name=model_name)
    return generator.generate_embedding(text)
