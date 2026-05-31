import json
import streamlit as st
import yfinance as yf
from serpapi import GoogleSearch
from langchain_core.tools import tool

@tool
def get_stock_financials(ticker: str) -> str:
    """
    Получает финансовые показатели для заданного тикера.
    Возвращает JSON с текущей ценой, P/E, P/B, рыночной капитализацией,
    дивидендной доходностью, ростом выручки за 3 года (CAGR), ROE, Debt/Equity.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet

        # Текущая цена
        current_price = (info.get('currentPrice') or
                         info.get('regularMarketPrice') or
                         info.get('previousClose'))

        # Рыночная капитализация
        market_cap = info.get('marketCap')

        # P/E (trailing/forward)
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')

        # P/B
        pb_ratio = info.get('priceToBook')

        # Дивидендная доходность в процентах
        div_yield = info.get('dividendYield')
        if div_yield is not None:
            div_yield = round(div_yield * 100, 2)

        # ROE (Net Income / Shareholders' Equity)
        roe = None
        if (not financials.empty and 'Net Income' in financials.index and
            not balance_sheet.empty and 'Stockholders Equity' in balance_sheet.index):
            net_income = financials.loc['Net Income'].iloc[0]
            equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
            if equity:
                roe = round((net_income / equity) * 100, 2)

        # Debt/Equity
        debt_equity = info.get('debtToEquity')
        if debt_equity is not None:
            # yfinance иногда возвращает в процентах, приводим к коэффициенту
            debt_equity = round(debt_equity / 100, 2) if debt_equity > 10 else debt_equity
        else:
            if (not balance_sheet.empty and 'Total Debt' in balance_sheet.index and
                'Stockholders Equity' in balance_sheet.index):
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
                if equity:
                    debt_equity = round(total_debt / equity, 2)

        # 3-летний CAGR выручки
        revenue_cagr = None
        if not financials.empty and 'Total Revenue' in financials.index:
            revenues = financials.loc['Total Revenue'].dropna().sort_index(ascending=False)
            if len(revenues) >= 4:  # нужно минимум 4 точки, чтобы взять отступ в 3 года
                latest = revenues.iloc[0]
                oldest = revenues.iloc[3]
                if oldest and latest and oldest > 0:
                    cagr = (latest / oldest) ** (1/3) - 1
                    revenue_cagr = round(cagr * 100, 2)

        result = {
            "ticker": ticker.upper(),
            "company_name": info.get('shortName') or info.get('longName') or ticker,
            "current_price": current_price,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
            "dividend_yield_percent": div_yield,
            "roe_percent": roe,
            "debt_to_equity": debt_equity,
            "revenue_3y_cagr_percent": revenue_cagr,
        }
        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        return json.dumps({"error": f"Не удалось получить данные по тикеру {ticker}: {str(e)}"})


@tool
def search_news(query: str) -> str:
    """
    Ищет новости по заданному запросу за последние 7 дней.
    Возвращает JSON-список из не более 5 новостей с полями: title, snippet, source, date, link.
    """
    try:
        serpapi_key = st.secrets["SERPAPI_API_KEY"]
        params = {
            "q": query,
            "tbm": "nws",
            "tbs": "qdr:w",   # последняя неделя
            "api_key": serpapi_key,
            "num": 5,
            "hl": "ru",
            "gl": "us",
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        news_items = results.get("news_results", [])

        articles = []
        for item in news_items[:5]:
            articles.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "source": item.get("source"),
                "date": item.get("date"),
                "link": item.get("link")
            })

        if not articles:
            return json.dumps({"message": "Новости не найдены."})
        return json.dumps(articles, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Ошибка при поиске новостей: {str(e)}"})