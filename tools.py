import yfinance as yf
import numpy as np
from bs4 import BeautifulSoup
from langchain_core.tools import tool 
import requests
from ddgs import DDGS
import re


def _extract_news_timelimit(query: str) -> str | None:
    normalized = query.lower()
    if any(phrase in normalized for phrase in ["past week", "last week", "this week", "7 days", "past 7 days"]):
        return "w"
    if any(phrase in normalized for phrase in ["past day", "last day", "today", "24 hours"]):
        return "d"
    if any(phrase in normalized for phrase in ["past month", "last month", "30 days"]):
        return "m"
    if any(phrase in normalized for phrase in ["past year", "last year", "12 months"]):
        return "y"
    return None


def _clean_news_query(query: str) -> str:
    cleaned = re.sub(
        r"\b(past week|last week|this week|past 7 days|7 days|past day|last day|today|24 hours|past month|last month|30 days|past year|last year|12 months)\b",
        "",
        query,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip() or query.strip()

@tool
def get_stock_data(tickers_string: str) -> str:
    """
    Fetches the current stock price, sector, and forward PE ratio for a list of ticker symbols.

    WHEN TO USE: Use this tool when you need current price data or basic fundamental info for known ticker symbols.
    DO NOT use this to find stocks in a sector — use 'get_stocks_by_sector' for that.

    INPUT FORMAT: A single string of comma-separated ticker symbols.
    Example input: "AAPL, MSFT, NVDA"
    Example input: "CAT, HON, UNP"

    OUTPUT FORMAT: A text block with one section per ticker showing:
    - Ticker symbol
    - Current Price
    - Sector
    - Forward PE ratio
    """
    tickers = [t.strip() for t in tickers_string.replace(',', ' ').split() if t.strip()]
    all_summaries = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice', 'N/A')
            summary = (
                f"Ticker: {ticker}\n"
                f"Current Price: ${current_price}\n"
                f"Sector: {info.get('sector', 'N/A')}\n"
                f"Forward PE: {info.get('forwardPE', 'N/A')}"
            )
            all_summaries.append(summary)
        except Exception as e:
            all_summaries.append(f"Error fetching data for {ticker}: {str(e)}")
            
    return "\n-------------------\n".join(all_summaries)

@tool
def get_financial_news(queries_string: str) -> str:
    """
    Searches the web for the latest financial news articles about specific companies or financial topics.

    WHEN TO USE: Use this tool ONLY when you need qualitative news or sentiment information about a company.
    DO NOT use this tool to find stock tickers or calculate financial metrics.

    INPUT FORMAT: A single string of comma-separated search queries. Use full company names, not just ticker symbols.
    Example input: "Caterpillar stock news, Honeywell financials"
    Example input: "Apple stock earnings report"
    BAD input: "CAT" (too ambiguous — could match unrelated results)

    OUTPUT FORMAT: A text block listing the top 3 news articles per query, each with a Title and Snippet.
    """
    queries = [q.strip() for q in queries_string.split(',') if q.strip()]
    all_news = []
    
    for raw_query in queries:
        try:
            timelimit = _extract_news_timelimit(raw_query)
            query = _clean_news_query(raw_query)

            news_client = DDGS()
            try:
                if timelimit:
                    results = news_client.news(query, max_results=3, timelimit=timelimit)
                else:
                    results = news_client.news(query, max_results=3)
            except TypeError:
                # Backward compatibility for ddgs versions that don't support timelimit
                results = news_client.news(query, max_results=3)
            
            if not results:
                all_news.append(f"No news found for '{raw_query}'.")
                continue
            
            window_text = " (filtered to recent window)" if timelimit else ""
            news_summary = f"News for '{raw_query}'{window_text}:\n"
            for i, res in enumerate(results):
                news_summary += f"  {i+1}. Title: {res['title']}\n  Snippet: {res['body']}\n\n"
            all_news.append(news_summary)
        except Exception as e:
            all_news.append(f"Error fetching news for '{raw_query}': {str(e)}")
            
    return "\n-------------------\n".join(all_news)

@tool
def get_risk_metrics(tickers_string: str) -> str:
    """
    Calculates trailing 1-year risk metrics for stocks: Expected Annual Return, Annualized Volatility, and Sharpe Ratio.

    WHEN TO USE: Use this tool whenever you need to calculate or compare risk-adjusted performance of stocks.
    This is the ONLY way to get Sharpe Ratio and Volatility numbers. DO NOT make up these numbers.

    INPUT FORMAT: A single string of comma-separated ticker symbols.
    Example input: "CAT, HON, UNP"
    Example input: "AAPL, MSFT"

    OUTPUT FORMAT: A text block with one section per ticker showing:
    - Expected Annual Return (percentage)
    - Annualized Volatility (percentage)
    - Sharpe Ratio (decimal number)
    """
    tickers = [t.strip() for t in tickers_string.replace(',', ' ').split() if t.strip()]
    all_metrics = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            
            if data.empty:
                all_metrics.append(f"No historical data found for {ticker}.")
                continue
            
            daily_returns = data['Close'].pct_change().dropna()
            
            annual_return = daily_returns.mean() * 252
            annual_volatility = daily_returns.std() * np.sqrt(252)
            
            risk_free_rate = 0.04
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            
            metrics = (
                f"Risk Metrics for {ticker} (Trailing 1-Year):\n"
                f"- Expected Annual Return: {annual_return:.2%}\n"
                f"- Annualized Volatility (Risk): {annual_volatility:.2%}\n"
                f"- Sharpe Ratio: {sharpe_ratio:.2f}"
            )
            all_metrics.append(metrics)
        except Exception as e:
            all_metrics.append(f"Error calculating risk metrics for {ticker}: {str(e)}")
            
    return "\n-------------------\n".join(all_metrics)

@tool
def get_portfolio_correlation(tickers_string: str) -> str:
    """
    Calculates the Pearson correlation coefficient between pairs of stocks based on 1-year daily returns.

    WHEN TO USE: Use this tool when comparing how similarly two or more stocks move together.
    Useful for portfolio diversification analysis. Requires at least 2 tickers.

    INPUT FORMAT: A single string of comma-separated ticker symbols (minimum 2).
    Example input: "AAPL, MSFT, NVDA"

    OUTPUT FORMAT: A text block showing the correlation coefficient for each pair, plus an interpretation
    (highly correlated, low correlation, or inversely correlated).
    """
    tickers = [t.strip() for t in tickers_string.replace(',', ' ').split() if t.strip()]
    
    if len(tickers) < 2:
        return "Error: Please provide at least two tickers to calculate correlation."
    try:
        data = yf.download(" ".join(tickers), period="1y", progress=False)
        close_prices = data['Close']
        
        if close_prices.empty or len(close_prices.columns) < 2:
            return f"Could not fetch complete historical data to compare {tickers_string}."
        
        returns = close_prices.pct_change().dropna()
        corr_matrix = returns.corr()
        
        all_correlations = []
        
        # Loop through unique pairs to apply your original interpretation logic
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                
                # Check if data exists for both in the matrix
                if t1 in corr_matrix.columns and t2 in corr_matrix.columns:
                    correlation = corr_matrix.loc[t1, t2]
                    
                    interpretation = ""
                    if correlation > 0.7:
                        interpretation = "Highly correlated. They tend to move together, offering poor diversification."
                    elif -0.3 <= correlation <= 0.3:
                        interpretation = "Low correlation. Excellent candidates for portfolio diversification."
                    elif correlation < -0.3:
                        interpretation = "Inversely correlated. Strong potential for hedging."
                    else:
                        interpretation = "Moderately correlated."

                    pair_result = (
                        f"Pearson Correlation between {t1} and {t2}: {correlation:.2f}\n"
                        f"Quantitative Analysis: {interpretation}"
                    )
                    all_correlations.append(pair_result)
                    
        return "\n-------------------\n".join(all_correlations)
    except Exception as e:
        return f"Error calculating correlation for {tickers_string}: {str(e)}"

@tool
def get_stocks_by_sector(sector: str) -> str:
    """
    Fetches a list of S&P 500 stock tickers that belong to a specific GICS sector.

    WHEN TO USE: Use this tool FIRST whenever the user mentions a sector name like 'Industrials', 'Technology',
    'Health Care', 'Financials', 'Energy', 'Consumer Discretionary', etc.
    This is the ONLY way to discover which ticker symbols belong to a sector. DO NOT guess ticker symbols.

    INPUT FORMAT: A single sector name string (not ticker symbols).
    Example input: "Industrials"
    Example input: "Information Technology"
    Example input: "Health Care"

    OUTPUT FORMAT: A comma-separated string of all S&P 500 tickers in that sector.
    Example output: "MMM, AOS, ALK, ALLE, AME, ..."
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        
        if not table:
            return "Error: Could not find the constituents table."
        
        rows = table.find_all('tr')[1:]
        matched_tickers = []
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 3:
                continue
            
            symbol = cols[0].text.strip().replace('.', '-')
            row_sector = cols[2].text.strip()
            
            if sector.lower() in row_sector.lower():
                matched_tickers.append(symbol)
                
                    
        if not matched_tickers:
            return f"No stocks found for sector: '{sector}'."
            
        return ", ".join(matched_tickers)
        
    except Exception as e:
        return f"Error fetching sector stocks: {str(e)}"

if __name__ == "__main__":
    print(get_stocks_by_sector.invoke("Industrials"))