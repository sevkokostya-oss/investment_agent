import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from tools import get_stock_financials, search_news


class DeepseekLLM(LLM):
    api_url: str
    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: int = 30

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.request_timeout,
        )
        response.raise_for_status()

        result = response.json()
        if isinstance(result, dict):
            first_choice = result.get("choices", [{}])[0]
            if isinstance(first_choice, dict):
                if "message" in first_choice:
                    return str(first_choice["message"].get("content", "")).strip()
                return str(first_choice.get("text", "")).strip()
        return json.dumps(result)


def create_agent():
    """Создаёт и возвращает AgentExecutor с финансовыми инструментами."""
    load_dotenv()
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY") if hasattr(st, 'secrets') else None

    if not deepseek_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY не найден. Установите переменную окружения DEEPSEEK_API_KEY или добавьте его в .streamlit/secrets.toml."
        )

    deepseek_api_url = os.getenv(
        "DEEPSEEK_API_URL",
        "https://api.deepseek.ai/v1/chat/completions"
    )

    llm = DeepseekLLM(
        api_url=deepseek_api_url,
        api_key=deepseek_api_key,
        model="gpt-4o",
        temperature=0,
        max_tokens=1024,
    )
    tools = [get_stock_financials, search_news]

    # Строгий промпт для аналитика
    template = """
Ты — опытный финансовый аналитик. Твоя задача — предоставить структурированный анализ акций компании на основе данных.
У тебя есть доступ к следующим инструментам:
{tools}

Имя инструмента: {tool_names}

Формат строго следующий:
Question: вопрос пользователя, на который нужно ответить
Thought: твои размышления о том, что нужно сделать
Action: действие, которое нужно выполнить, должно быть одним из [{tool_names}]
Action Input: входные данные для действия
Observation: результат выполнения действия
... (это Thought/Action/Action Input/Observation может повторяться)
Thought: Я знаю финальный ответ
Final Answer: финальный ответ пользователю

Правила:
- Никогда не давай рекомендаций покупать или продавать акции. Только объективные данные и факты.
- В финальном ответе обязательно укажи источники новостей, если они использовались (название источника и ссылку).
- Структурируй ответ по секциям: Новостной фон, Финансовые показатели, Мультипликаторы, Риски.
- Если данных по какому-то показателю нет, укажи "Н/Д".
- Отвечай на русском языке.
- В конце всегда добавляй дисклеймер: "Это не инвестиционная рекомендация. Предоставленная информация носит исключительно аналитический характер."

Начинай!

Question: {input}
Thought: {agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)

    # Используем классический initialize_agent (ReAct)
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True
    )

    return agent_executor
