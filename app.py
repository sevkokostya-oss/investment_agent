import json
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
from langchain_core.callbacks.base import BaseCallbackHandler
from agent import create_agent
from tools import get_stock_financials

# Настройка страницы
st.set_page_config(
    page_title="Investment AI Agent",
    page_icon="📈",
    layout="wide"
)

# Пользовательские стили для более презентабельного дизайна
st.markdown(
    """
    <style>
    section.main { background: linear-gradient(180deg, #071026 0%, #132554 55%, #192e5f 100%); }
    .stButton>button { background-color: #00bfa6; color: #ffffff; border: none; }
    .stButton>button:hover { background-color: #02907b; }
    .stTextInput>div>div>input { border-radius: 12px; }
    .metric-card { padding: 1.25rem; border-radius: 24px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16); box-shadow: 0 20px 40px rgba(0,0,0,0.15); }
    .card-title { color: #e4efff; font-weight: 700; margin-bottom: 0.5rem; }
    .card-value { color: #ffffff; font-size: 2rem; font-weight: 800; }
    .card-subtitle { color: #b2d2ff; }
    .analytics-box { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 24px; padding: 1.5rem; }
    .section-title { color: #ffffff; font-weight: 800; }
    .news-card { padding: 1.15rem; border-radius: 18px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14); margin-bottom: 1rem; }
    .news-card:hover { transform: translateY(-2px); }
    .news-link { color: #a5c8ff; text-decoration: none; }
    .news-link:hover { text-decoration: underline; }
    .disclaimer-box { background: #0b1d3f; border-left: 4px solid #ff6b6b; padding: 1rem; border-radius: 16px; color: #f4f7ff; }
    .stExpanderHeader { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Инвестиционный AI-агент")
st.markdown(
    "Интерактивная панель для быстрого анализа акций: новости, ключевые метрики и ясный аналитический отчёт."
)

# Помощь и образцы
with st.container():
    hints_col1, hints_col2 = st.columns([3, 1])
    with hints_col1:
        st.markdown("**Введите тикер компании и нажмите «Анализировать».**")
        st.markdown(
            "Примеры: `AAPL`, `TSLA`, `NVDA`, `MSFT`, `GOOG`"
        )
    with hints_col2:
        st.markdown("**Преимущества:**")
        st.markdown("- Быстрый обзор ключевых финансовых метрик")
        st.markdown("- Визуализация цены за последний год")
        st.markdown("- Чёткая структура отчёта для презентаций")

# Сбор информации от пользователя
ticker_input = st.text_input("Тикер", placeholder="Например, AAPL").strip().upper()
analyze_btn = st.button("Анализировать")

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

@st.cache_resource(show_spinner=False)
def load_agent():
    return create_agent()

try:
    agent_executor = load_agent()
except Exception as e:
    agent_executor = None
    st.error(f"Ошибка инициализации агента: {e}")

# Вспомогательные функции для отображения метрик
def format_large_number(value):
    try:
        if value is None:
            return "Н/Д"
        value = float(value)
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value/1_000:.2f}K"
        return f"{value:.2f}"
    except Exception:
        return str(value)


def show_metric_card(title, value, subtitle=None, ratio=None):
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-value'>{value}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='card-subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    if ratio is not None:
        st.markdown(f"<div class='card-subtitle'>Изменение: {ratio}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_price_chart(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    hist = stock.history(start=start_date, end=end_date)
    if hist.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    ax.plot(hist.index, hist['Close'], color='#00bfa6', linewidth=2)
    ax.fill_between(hist.index, hist['Close'], color='#00bfa6', alpha=0.15)
    ax.set_title(f"Динамика цены {ticker} за последний год", color='#eaf6ff', fontsize=18)
    ax.set_xlabel("Дата", color='#b2c8ff')
    ax.set_ylabel("Цена (USD)", color='#b2c8ff')
    ax.tick_params(colors='#b2c8ff')
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#4f6db2')
    ax.spines['left'].set_color('#4f6db2')
    fig.patch.set_facecolor('#071026')
    ax.set_facecolor('#071026')
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


if analyze_btn:
    if not ticker_input:
        st.warning("Пожалуйста, введите тикер.")
    elif agent_executor is None:
        st.error("Агент не инициализирован. Проверьте DEEPSEEK_API_KEY и попробуйте снова.")
    else:
        status_container = st.empty()
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

        # Попытка получить метрики напрямую для карточек
        financials = None
        try:
            financials = json.loads(get_stock_financials(ticker_input))
        except Exception:
            financials = None

        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("<h2 class='section-title'>Ключевые метрики</h2>", unsafe_allow_html=True)
                if financials and not financials.get('error'):
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        show_metric_card(
                            "Цена",
                            format_large_number(financials.get('current_price')),
                            subtitle="Текущая цена"
                        )
                    with k2:
                        show_metric_card(
                            "P/E",
                            financials.get('pe_ratio') or 'Н/Д',
                            subtitle="Коэффициент"
                        )
                    with k3:
                        show_metric_card(
                            "P/B",
                            financials.get('pb_ratio') or 'Н/Д',
                            subtitle="Коэффициент"
                        )
                    with k4:
                        show_metric_card(
                            "ROE",
                            f"{financials.get('roe_percent')}%" if financials.get('roe_percent') is not None else 'Н/Д',
                            subtitle="Доходность капитала"
                        )
                else:
                    st.warning("Не удалось получить прямые финансовые метрики для карточек.")

                st.markdown("<h2 class='section-title'>Аналитический отчёт</h2>", unsafe_allow_html=True)
                st.markdown(final_output)

            with col2:
                st.markdown("<div class='analytics-box'>", unsafe_allow_html=True)
                st.markdown("<h3 class='card-title'>Информация о компании</h3>", unsafe_allow_html=True)
                if financials and not financials.get('error'):
                    company_name = financials.get('company_name', ticker_input)
                    market_cap = format_large_number(financials.get('market_cap'))
                    div_yield = f"{financials.get('dividend_yield_percent')}%" if financials.get('dividend_yield_percent') is not None else 'Н/Д'
                    debt_ratio = financials.get('debt_to_equity') or 'Н/Д'
                    revenue_cagr = f"{financials.get('revenue_3y_cagr_percent')}%" if financials.get('revenue_3y_cagr_percent') is not None else 'Н/Д'

                    st.markdown(f"**Компания:** {company_name}")
                    st.markdown(f"**Тикер:** {ticker_input}")
                    st.markdown(f"**Рыночная капитализация:** {market_cap}")
                    st.markdown(f"**Дивидендная доходность:** {div_yield}")
                    st.markdown(f"**Debt / Equity:** {debt_ratio}")
                    st.markdown(f"**CAGR выручки 3 года:** {revenue_cagr}")
                else:
                    st.write("Нет данных по компании.")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # График цены
        chart_buf = render_price_chart(ticker_input)
        if chart_buf:
            st.markdown("<h2 class='section-title'>Динамика цены за последний год</h2>", unsafe_allow_html=True)
            st.image(chart_buf, use_container_width=True)
        else:
            st.warning("Не удалось загрузить исторические данные для графика.")

        # Ход мыслей агента
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
                    st.markdown(f"📤 Результат: {obs_text[:500]}{'...' if len(obs_text) > 500 else ''}")
                    st.markdown("---")

        st.markdown("---")
        st.markdown(
            "<div class='disclaimer-box'>"
            "⚠️ Это не инвестиционная рекомендация. Предоставленная информация носит исключительно аналитический характер и не должна" 
            " рассматриваться как совет к покупке или продаже ценных бумаг."
            "</div>",
            unsafe_allow_html=True
        )

