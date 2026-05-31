import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from tools import get_stock_financials, search_news

def create_agent():
    """Создаёт и возвращает AgentExecutor с финансовыми инструментами."""
    llm = ChatOpenAI(
        model="gpt-4o",  # или "gpt-3.5-turbo" для экономии
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
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
