import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
from langchain.callbacks.base import BaseCallbackHandler
from agent import create_agent

# Настройка страницы
st.set_page_config(page_title="Investment AI Agent", layout="wide")
st.title("📈 Инвестиционный AI-агент")
st.markdown(
    "Введите тикер компании (например, **AAPL**, **TSLA**, **NVDA**), чтобы получить "
    "структурированный аналитический отчёт."
)

# Callback-обработчик для отображения текущего статуса в Streamlit
class StreamlitStatusCallback(BaseCallbackHandler):
    def __init__(self, status_container):
        self.status_container = status_container

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "инструмент")
        if "search_news" in tool_name:
            self.status_container.text("🔍 Ищу новости...")
        elif "get_stock_financials" in tool_name:
            self.status_container.text("📊 Загружаю финансовые показатели...")
        else:
            self.status_container.text(f"⚙️ Выполняется {tool_name}...")

    def on_agent_action(self, action, **kwargs):
        self.status_container.text(f"🤔 Агент размышляет: {action.log[:120]}...")

# Загружаем агента (кэшируем, чтобы не пересоздавать при каждом взаимодействии)
@st.cache_resource(show_spinner=False)
def load_agent():
    return create_agent()

agent_executor = load_agent()

# Пользовательский интерфейс
ticker_input = st.text_input("Тикер", placeholder="Например, AAPL").strip().upper()
analyze_btn = st.button("Анализировать")

if analyze_btn and ticker_input:
    status_container = st.empty()   # сюда будут выводиться статусы
    status_container.text("🚀 Начинаю анализ...")

    callback = StreamlitStatusCallback(status_container)

    with st.spinner("Агент выполняет запросы..."):
        try:
            result = agent_executor.invoke(
                {"input": f"Проанализируй компанию {ticker_input}"},
                callbacks=[callback]
            )
        except Exception as e:
            st.error(f"Ошибка при выполнении агента: {e}")
            st.stop()

    status_container.text("✅ Отчёт сформирован.")
    final_output = result.get("output", "Нет ответа от агента.")

    # График цены за последний год
    st.subheader("📉 График цены за последний год")
    try:
        stock = yf.Ticker(ticker_input)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist = stock.history(start=start_date, end=end_date)
        if not hist.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hist.index, hist['Close'], label='Цена закрытия', color='#1f77b4')
            ax.set_title(f"Динамика цены {ticker_input} за 1 год", fontsize=14)
            ax.set_xlabel("Дата")
            ax.set_ylabel("Цена (USD)")
            ax.legend()
            ax.grid(alpha=0.3)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig)
        else:
            st.warning("Не удалось загрузить исторические данные для графика.")
    except Exception as e:
        st.warning(f"Ошибка построения графика: {e}")

    # Отчёт агента
    st.subheader("📋 Аналитический отчёт")
    st.markdown(final_output)

    # Ход мыслей агента (по шагам)
    with st.expander("🧠 Ход мыслей агента"):
        intermediate_steps = result.get("intermediate_steps", [])
        if not intermediate_steps:
            st.write("Нет записей.")
        else:
            for i, (action, observation) in enumerate(intermediate_steps):
                st.markdown(f"**Шаг {i+1}:**")
                st.markdown(f"🔧 Инструмент: `{action.tool}`")
                st.markdown(f"📥 Вход: `{action.tool_input}`")
                obs_text = str(observation)
                st.markdown(f"📤 Результат: {obs_text[:500]}{'...' if len(obs_text)>500 else ''}")
                st.markdown("---")

    # Обязательный дисклеймер
    st.markdown("---")
    st.markdown(
        "<span style='color:red; font-weight:bold;'>"
        "⚠️ Это не инвестиционная рекомендация. Предоставленная информация носит "
        "исключительно аналитический характер и не должна рассматриваться как совет "
        "к покупке или продаже ценных бумаг."
        "</span>",
        unsafe_allow_html=True
    )

elif analyze_btn and not ticker_input:
    st.warning("Пожалуйста, введите тикер.")