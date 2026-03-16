import ast
import math
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ddgs import DDGS
from langchain_core.tools import tool


ALLOWED_MATH_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


class SafeMathEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Tuple,
            ast.List,
        }

    def generic_visit(self, node: ast.AST):
        if type(node) not in self.allowed_nodes:
            raise ValueError(f"Unsupported expression component: {type(node).__name__}")
        super().generic_visit(node)


@tool
def calculate_advanced_math(expression: str) -> str:
    """
    Safely evaluates mathematical expressions with support for arithmetic and common math functions.

    Example: "(100*432.1)+(50*187.2)"
    Example: "sqrt(252)*0.32"
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        SafeMathEvaluator().visit(parsed)
        result = eval(compile(parsed, "<safe_math>", "eval"), {"__builtins__": {}}, ALLOWED_MATH_FUNCS)
        return f"Expression: {expression}\nResult: {result}"
    except Exception as error:
        return f"Error evaluating expression '{expression}': {error}"


@tool
def get_stock_price(ticker: str) -> str:
    """
    Fetches latest stock market snapshot for a ticker.

    Returns latest close, currency, market cap, trailing PE, and 52-week range when available.
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        history = stock.history(period="5d")

        if history.empty:
            return f"No recent market data found for {ticker}."

        latest_close = history["Close"].iloc[-1]
        low_52 = info.get("fiftyTwoWeekLow", "N/A")
        high_52 = info.get("fiftyTwoWeekHigh", "N/A")

        summary = (
            f"Ticker: {ticker}\n"
            f"Latest Close: {latest_close:.2f}\n"
            f"Currency: {info.get('currency', 'N/A')}\n"
            f"Market Cap: {info.get('marketCap', 'N/A')}\n"
            f"Trailing PE: {info.get('trailingPE', 'N/A')}\n"
            f"52W Range: {low_52} - {high_52}"
        )
        return summary
    except Exception as error:
        return f"Error fetching stock snapshot for {ticker}: {error}"


@tool
def get_stock_risk_metrics(tickers_string: str, period: str = "1y") -> str:
    """
    Computes expected annual return, annualized volatility, and Sharpe ratio from daily returns.

    Input tickers_string format: "AAPL, MSFT, NVDA"
    """
    tickers = [item.strip().upper() for item in tickers_string.replace(",", " ").split() if item.strip()]
    if not tickers:
        return "No valid tickers provided."

    valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    normalized_period = period.strip().lower() if period else "1y"
    period_aliases = {
        "annual": "1y",
        "year": "1y",
        "yearly": "1y",
        "monthly": "1mo",
        "weekly": "5d",
        "daily": "1d",
    }
    normalized_period = period_aliases.get(normalized_period, normalized_period)
    if normalized_period not in valid_periods:
        normalized_period = "1y"

    outputs = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period=normalized_period)
            if data.empty:
                outputs.append(f"No historical data found for {ticker}.")
                continue

            daily_returns = data["Close"].pct_change().dropna()
            if daily_returns.empty:
                outputs.append(f"Not enough data points to compute risk metrics for {ticker}.")
                continue

            annual_return = daily_returns.mean() * 252
            annual_vol = daily_returns.std() * np.sqrt(252)
            risk_free = 0.04
            sharpe = (annual_return - risk_free) / annual_vol if annual_vol != 0 else float("nan")

            outputs.append(
                f"Risk Metrics for {ticker} ({normalized_period}):\n"
                f"- Expected Annual Return: {annual_return:.2%}\n"
                f"- Annualized Volatility: {annual_vol:.2%}\n"
                f"- Sharpe Ratio: {sharpe:.2f}"
            )
        except Exception as error:
            outputs.append(f"Error computing risk metrics for {ticker}: {error}")

    return "\n-------------------\n".join(outputs)


@tool
def get_financial_statements(ticker: str, statement_type: str = "income", period: str = "annual", top_rows: int = 10) -> str:
    """
    Fetches core financial statement lines from Yahoo Finance for a given ticker.

    statement_type: income | balance_sheet | cashflow
    period: annual | quarterly
    """
    ticker = ticker.strip().upper()
    statement_type = statement_type.strip().lower()
    period = period.strip().lower()

    try:
        stock = yf.Ticker(ticker)

        if statement_type == "income":
            frame = stock.financials if period == "annual" else stock.quarterly_financials
        elif statement_type == "balance_sheet":
            frame = stock.balance_sheet if period == "annual" else stock.quarterly_balance_sheet
        elif statement_type == "cashflow":
            frame = stock.cashflow if period == "annual" else stock.quarterly_cashflow
        else:
            return "Invalid statement_type. Use: income, balance_sheet, or cashflow."

        if frame is None or frame.empty:
            return f"No {period} {statement_type} data found for {ticker}."

        trimmed = frame.head(max(1, top_rows)).copy()
        trimmed.columns = [str(column.date()) if hasattr(column, "date") else str(column) for column in trimmed.columns]

        return (
            f"Ticker: {ticker}\n"
            f"Statement: {statement_type} ({period})\n"
            f"Top {len(trimmed)} rows:\n"
            f"{trimmed.to_string()}"
        )
    except Exception as error:
        return f"Error fetching {statement_type} for {ticker}: {error}"
    


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Performs a web search and returns top result snippets.

    Use for qualitative context, definitions, and latest public information.
    """
    query = query.strip()
    if not query:
        return "Query is empty."

    try:
        results = DDGS().text(query, max_results=max(1, min(max_results, 10)))
        if not results:
            return f"No web search results found for '{query}'."

        lines = [f"Top web results for '{query}':"]
        for index, item in enumerate(results, start=1):
            title = item.get("title", "N/A")
            body = item.get("body", "")
            href = item.get("href", "")
            lines.append(f"{index}. {title}\n   URL: {href}\n   Snippet: {body}")
        return "\n".join(lines)
    except Exception as error:
        return f"Error performing web search for '{query}': {error}"


@tool
def get_financial_news(query: str, timelimit: str = "w", max_results: int = 5) -> str:
    """
    Fetches finance-related news articles with optional recency filtering.

    timelimit supports: d (day), w (week), m (month), y (year)
    """
    query = query.strip()
    if not query:
        return "Query is empty."

    valid_timelimits = {"d", "w", "m", "y"}
    if timelimit not in valid_timelimits:
        timelimit = "w"

    try:
        client = DDGS()
        try:
            results = client.news(query, max_results=max(1, min(max_results, 10)), timelimit=timelimit)
        except TypeError:
            results = client.news(query, max_results=max(1, min(max_results, 10)))

        if not results:
            return f"No news found for '{query}'."

        lines = [f"Top finance news for '{query}' (timelimit={timelimit}):"]
        for index, item in enumerate(results, start=1):
            title = item.get("title", "N/A")
            body = item.get("body", "")
            source = item.get("source", "N/A")
            date = item.get("date", "N/A")
            lines.append(f"{index}. [{source}] {title}\n   Date: {date}\n   Snippet: {body}")
        return "\n".join(lines)
    except Exception as error:
        return f"Error fetching financial news for '{query}': {error}"


@tool
def compare_stock_price_changes(tickers_string: str, period: str = "1mo") -> str:
    """
    Compares percentage price change over a given period for multiple tickers.

    period examples: 5d, 1mo, 3mo, 6mo, 1y
    """
    tickers = [item.strip().upper() for item in tickers_string.replace(",", " ").split() if item.strip()]
    if len(tickers) < 2:
        return "Provide at least two tickers for comparison."

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period=period)
            if data.empty:
                rows.append({"ticker": ticker, "status": "No data"})
                continue

            start_price = float(data["Close"].iloc[0])
            end_price = float(data["Close"].iloc[-1])
            pct_change = ((end_price - start_price) / start_price) * 100 if start_price != 0 else float("nan")

            rows.append(
                {
                    "ticker": ticker,
                    "start": round(start_price, 4),
                    "end": round(end_price, 4),
                    "pct_change": round(pct_change, 2),
                }
            )
        except Exception as error:
            rows.append({"ticker": ticker, "status": f"Error: {error}"})

    frame = pd.DataFrame(rows)
    if "pct_change" in frame.columns and not frame["pct_change"].dropna().empty:
        frame = frame.sort_values(by="pct_change", ascending=False, na_position="last")

    return f"Price change comparison ({period}) generated at {datetime.utcnow().isoformat()}Z\n{frame.to_string(index=False)}"
