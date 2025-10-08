"""
Упрощенный клиент для работы с OpenRouter API.
Используется для извлечения тегов из научных документов.
"""

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)


class OpenRouterError(Exception):
    """Ошибка при работе с OpenRouter API"""
    pass


class OpenRouterClient:
    """Минимальный клиент для работы с OpenRouter API"""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_response_tokens: int = 4095,
        temperature: float = 0.0
    ):
        """
        Инициализация клиента OpenRouter.

        Args:
            api_key: API ключ OpenRouter
            model_name: Название модели (должно содержать '/' для OpenRouter)
            max_response_tokens: Максимальное количество токенов в ответе
            temperature: Температура для генерации (всегда 0.0 для нашей задачи)
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        if not model_name.count("/"):
            raise ValueError(
                f"Название модели для OpenRouter должно содержать '/'. "
                f"Например: 'anthropic/claude-3.5-sonnet'"
            )

        self.api_key = api_key
        self.model_name = model_name
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((
            APIError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            OpenRouterError
        )),
        reraise=True
    )
    def call_api(self, system_prompt: str, user_message: str) -> str:
        """
        Вызывает OpenRouter API с обработкой пустых ответов и автоматическими повторами.

        Args:
            system_prompt: Системный промпт
            user_message: Сообщение пользователя

        Returns:
            Ответ модели в виде строки
        """
        print(f"🤖 Отправка запроса к OpenRouter API ({self.model_name})...")

        client = None
        try:
            # Создаем клиента с отключенным внутренним retry механизмом
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                max_retries=0
            )

            # Формирование сообщений
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]

            # Формирование параметров запроса
            request_parameters = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_response_tokens,
                "temperature": self.temperature,
                "extra_headers": {
                    "HTTP-Referer": "https://jinr-hackathon.com",
                    "X-Title": "JINR Patent Landscape Analyzer"
                },
                "extra_body": {
                    "provider": {
                        "sort": "throughput"
                    }
                }
            }

            # Выполняем запрос
            api_response = client.chat.completions.create(**request_parameters)

            # Проверка базовой структуры ответа
            if not api_response or not hasattr(api_response, 'choices') or not api_response.choices:
                raise OpenRouterError("Пустой ответ от API (отсутствуют choices)")

            # Получение контента из ответа
            content = api_response.choices[0].message.content

            # Проверка: контент не должен быть пустым
            if content is None or (isinstance(content, str) and content.strip() == ""):
                provider = getattr(api_response, 'provider', 'unknown')
                usage_info = ""
                if hasattr(api_response, 'usage'):
                    usage_info = f", tokens: {api_response.usage.completion_tokens}/{api_response.usage.prompt_tokens}"

                error_msg = f"Получен пустой ответ от провайдера {provider}{usage_info}"
                print(f"⚠️ {error_msg}")
                raise OpenRouterError(error_msg)

            print("✓ Ответ получен успешно")
            return content

        except (APIError, APIConnectionError, APITimeoutError, RateLimitError) as api_error:
            error_details = []
            if hasattr(api_error, 'message'):
                error_details.append(f"сообщение: {api_error.message}")
            if hasattr(api_error, 'status_code'):
                error_details.append(f"HTTP статус: {api_error.status_code}")
            if hasattr(api_error, 'code'):
                error_details.append(f"код ошибки: {api_error.code}")

            error_info = ', '.join(error_details) if error_details else 'детали отсутствуют'
            error_message = f"[Ошибка API] {type(api_error).__name__}: {error_info}"
            print(f"❌ {error_message}")
            raise

        except OpenRouterError:
            raise

        except Exception as general_error:
            print(f"🔥 Непредвиденная ошибка: {str(general_error)}")
            raise

        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
