"""
Скрипт для скачивания модели BAAI/bge-large-en-v1.5
с отображением прогресса загрузки.
"""

import os
from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_model_with_progress(
    model_name: str = "BAAI/bge-large-en-v1.5",
    cache_dir: str = None
):
    """
    Скачивает модель с HuggingFace Hub с отображением прогресса.

    :param model_name: Название модели на HuggingFace
    :param cache_dir: Директория для кэширования (по умолчанию ~/.cache/huggingface)
    """
    print(f"🚀 Начинаем скачивание модели: {model_name}")
    print("-" * 60)

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    print(f"📁 Директория для сохранения: {cache_dir}")
    print()

    try:
        # Скачивание модели с прогресс-баром
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,  # Продолжить скачивание при обрыве
            local_files_only=False,
            tqdm_class=tqdm  # Использование tqdm для прогресса
        )

        print()
        print("✅ Модель успешно скачана!")
        print(f"📍 Путь к модели: {model_path}")
        print()
        print("Теперь вы можете использовать модель в своём коде:")
        print(f'generator = EmbeddingsGenerator(model_name="{model_name}")')

        return model_path

    except Exception as e:
        print(f"❌ Ошибка при скачивании: {e}")
        return None


def check_model_exists(model_name: str = "BAAI/bge-large-en-v1.5") -> bool:
    """
    Проверяет, скачана ли уже модель.

    :param model_name: Название модели
    :return: True если модель уже есть локально
    """
    from sentence_transformers import SentenceTransformer

    try:
        # Попытка загрузить модель без скачивания
        _ = SentenceTransformer(model_name, device='cpu')
        return True
    except:
        return False


if __name__ == "__main__":
    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    print("=" * 60)
    print("  Загрузчик модели для генерации эмбедингов")
    print("=" * 60)
    print()

    # Проверка наличия модели
    print("🔍 Проверяем, есть ли модель локально...")
    if check_model_exists(MODEL_NAME):
        print("✅ Модель уже скачана и готова к использованию!")
        print()
        user_input = input("Скачать заново? (y/n): ").lower()
        if user_input != 'y':
            print("Отмена загрузки.")
            exit(0)

    print()

    # Скачивание модели
    model_path = download_model_with_progress(MODEL_NAME)

    if model_path:
        print()
        print("=" * 60)
        print("🎉 Готово! Модель готова к работе в вашем хакатоне!")
        print("=" * 60)
