import os
from typing import Annotated, Literal, Sequence, TypedDict
import operator
import re
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from tools import (
    get_stock_data, get_risk_metrics, get_portfolio_correlation,
    get_financial_news, get_stocks_by_sector
)


# llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

# supervisor_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# worker_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# ==========================================
# 1. DEFINE THE GRAPH STATE & ROUTING
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

class Route(BaseModel):
    next: Literal["Quant", "Fundamental", "Drafter"] = Field(
        description="The next department to route to. Must be exactly one of: 'Quant', 'Fundamental', or 'Drafter'."
    )

class Grade(BaseModel):
    pass_audit: bool = Field(description="True if the report meets all criteria, False otherwise.")
    critique: str = Field(description="If pass_audit is False, explain exactly what needs to be fixed. If True, leave empty.")

# ==========================================
# 2. BUILD THE WORKER AGENTS
# ==========================================
quant_tools = [get_stock_data, get_risk_metrics, get_portfolio_correlation]
fundamental_tools = [get_financial_news, get_stocks_by_sector]

# --- Comprehensive system prompts for worker agents ---

QUANT_SYSTEM_PROMPT = """You are a Quantitative Analyst working inside a multi-agent financial advisory system.

YOUR ROLE:
You perform numerical and quantitative analysis on stocks. You calculate risk metrics, fetch price data, and compute correlations. You NEVER guess numbers — you ALWAYS use your tools to get real data.

YOUR TOOLS (you have access to exactly these 3 tools):

1. get_stock_data(tickers_string: str) -> str
   - Input: A comma-separated string of ticker symbols, e.g. "CAT, HON, UNP"
   - Output: Current price, sector, and forward PE for each ticker
   - Use when: You need current price data or basic info

2. get_risk_metrics(tickers_string: str) -> str
   - Input: A comma-separated string of ticker symbols, e.g. "CAT, HON, UNP"
   - Output: Expected Annual Return, Annualized Volatility, and Sharpe Ratio for each ticker
   - Use when: You need to calculate or compare risk-adjusted performance (Sharpe Ratio, Volatility)

3. get_portfolio_correlation(tickers_string: str) -> str
   - Input: A comma-separated string of at least 2 ticker symbols, e.g. "CAT, HON"
   - Output: Pearson correlation coefficient between each pair of tickers
   - Use when: You need to check how correlated two or more stocks are

STEP-BY-STEP INSTRUCTIONS:
1. Look at the conversation history to find which ticker symbols have already been identified.
2. Determine what quantitative analysis is needed (risk metrics, price data, correlation, etc.).
3. Call the appropriate tool(s) with the ticker symbols from the conversation.
4. After receiving tool results, write a clear plain-text summary of the findings.

CRITICAL RULES:
- NEVER invent or guess ticker symbols. ONLY use tickers that appear in the conversation history.
- NEVER fabricate numerical values. ALL numbers must come from tool outputs.
- NEVER output raw JSON. Write your final answer as readable plain text.
- If the conversation contains a list of sector tickers, pick 3 to 5 representative ones to analyze (do not try to analyze all 50+ tickers).
- When selecting representative tickers from a large list, pick well-known large-cap names from the sector.
- After your tool calls return data, write a clear summary of the results. Do NOT call more tools unless the data is missing.

EXAMPLE WORKFLOW:
If the conversation history contains tickers "CAT, HON, UNP, GE, RTX" from the Industrials sector and the user wants Sharpe Ratios:
1. Call get_risk_metrics with "CAT, HON, UNP, GE, RTX"
2. Read the output showing each ticker's Annual Return, Volatility, and Sharpe Ratio
3. Write a summary like: "Here are the trailing 1-year risk metrics for the Industrials stocks: CAT has a Sharpe Ratio of X.XX with Volatility of Y.YY%..."
"""

FUNDAMENTAL_SYSTEM_PROMPT = """You are a Fundamental Analyst working inside a multi-agent financial advisory system.

YOUR ROLE:
You handle two specific jobs: (1) finding which stocks belong to a sector, and (2) fetching qualitative news about companies. You NEVER do math or calculate metrics — that is the Quant team's job.

YOUR TOOLS (you have access to exactly these 2 tools):

1. get_stocks_by_sector(sector: str) -> str
   - Input: A sector name string, e.g. "Industrials" or "Information Technology" or "Health Care"
   - Output: A comma-separated string of all S&P 500 tickers in that sector
   - Use when: The user asks about stocks in a sector and specific tickers have NOT been identified yet

2. get_financial_news(queries_string: str) -> str
   - Input: A comma-separated string of search queries using FULL company names (not just ticker symbols)
   - Example good input: "Caterpillar stock news, Honeywell financials"
   - Example BAD input: "CAT" (too ambiguous, will return unrelated results about cats)
   - Output: Top 3 news articles per query with titles and snippets
   - Use when: You need to check recent news sentiment about a specific company

STEP-BY-STEP INSTRUCTIONS:
1. Read the conversation history carefully.
2. Determine what you need to do:
   a. If the user asked about a SECTOR and no tickers are in the history yet → call get_stocks_by_sector
   b. If there's a winning ticker identified and news is needed → call get_financial_news with the FULL company name
3. Call the appropriate tool.
4. After receiving tool results, write a clear plain-text summary.

CRITICAL RULES:
- NEVER calculate, estimate, or report numerical metrics (Sharpe Ratio, Volatility, Returns, Prices). That is the Quant team's job.
- NEVER invent ticker symbols from your own knowledge. ALWAYS use get_stocks_by_sector to find real tickers.
- When you get a long list of sector tickers from get_stocks_by_sector, present 3-5 well-known representative tickers from the list for the team to analyze.
- When searching for news, ALWAYS use the full company name plus "stock" or "financials" — never just the ticker symbol.
- After your tool calls return data, write a clear summary. Do NOT call more tools unless the data is missing.
- NEVER output raw JSON. Write your final answer as readable plain text.

EXAMPLE WORKFLOW - Finding sector stocks:
User asks about "Industrials" sector:
1. Call get_stocks_by_sector("Industrials")
2. Receive a long list like "MMM, AOS, ALK, ALLE, AME, ..."
3. Write: "From the S&P 500 Industrials sector, here are 5 representative stocks for analysis: CAT (Caterpillar), HON (Honeywell), UNP (Union Pacific), GE (GE Aerospace), RTX (RTX Corporation)."

EXAMPLE WORKFLOW - Getting news:
The conversation shows CAT won with the best Sharpe Ratio:
1. Call get_financial_news("Caterpillar stock news")
2. Read the news articles returned
3. Write: "Here is the latest news on Caterpillar (CAT): [summary of articles]..."
"""

quant_agent = create_react_agent(llm, tools=quant_tools)
fundamental_agent = create_react_agent(llm, tools=fundamental_tools)

# --- THE X-RAY TRACE FUNCTION ---
def run_agent_with_trace(agent, messages, dept_name):
    """Runs a worker agent and streams its internal thoughts to the terminal."""
    print(f"\n=================== {dept_name} EXECUTION TRACE ===================")
    final_text = ""
    tool_observations = []
    
    for event in agent.stream({"messages": messages}, stream_mode="values"):
        message = event["messages"][-1]
        
        if isinstance(message, AIMessage):
            if message.tool_calls:
                print(f"\n[{dept_name}] THINKING: I need to fetch external data.")
                for tc in message.tool_calls:
                    print(f"   -> Action: Calling '{tc['name']}' with args: {tc['args']}")
            elif message.content:
                print(f"\n[{dept_name}] SYNTHESIZING:")
                # Truncate the synthesis in the trace so it doesn't flood the terminal
                display = message.content[:300]
                print(f"   -> {display}... [truncated for display]")
                final_text = message.content
                
        elif isinstance(message, ToolMessage):
            obs_text = str(message.content)
            tool_observations.append(obs_text)
            display_text = obs_text[:400] + "... [truncated]" if len(obs_text) > 400 else obs_text
            print(f"\n[{dept_name}] OBSERVATION (Tool Output):\n{display_text}")
            print("-" * 55)
            
    print(f"\n=================== END {dept_name} TRACE ===================\n")
    if tool_observations:
        combined_tool_output = "\n\n".join(tool_observations)
        if final_text:
            return f"RAW_TOOL_OUTPUT:\n{combined_tool_output}\n\nANALYST_SUMMARY:\n{final_text}"
        return f"RAW_TOOL_OUTPUT:\n{combined_tool_output}"
    return final_text

# ==========================================
# 3. DEFINE THE NODES
# ==========================================

# --- SUPERVISOR ITERATION TRACKING ---
MAX_SUPERVISOR_ITERATIONS = 12
_supervisor_stats = {"call_count": 0}

KNOWN_SECTORS = [
    "industrials",
    "information technology",
    "technology",
    "health care",
    "healthcare",
    "financials",
    "energy",
    "consumer discretionary",
    "consumer staples",
    "materials",
    "utilities",
    "real estate",
    "communication services",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_sector_from_query(user_text: str) -> str | None:
    normalized = _normalize_text(user_text)
    for sector in KNOWN_SECTORS:
        if sector in normalized:
            return sector
    return None


def _is_news_request(user_text: str) -> bool:
    normalized = _normalize_text(user_text)
    news_keywords = ["news", "headline", "headlines", "sentiment", "latest", "past week", "last week"]
    return any(keyword in normalized for keyword in news_keywords)


def _is_quant_request(user_text: str) -> bool:
    normalized = _normalize_text(user_text)
    quant_keywords = [
        "sharpe",
        "volatility",
        "risk",
        "return",
        "correlation",
        "compare",
        "comparison",
        "vs",
        "versus",
        "best stock",
        "winner",
        "portfolio",
        "price",
        "pe ratio",
        "valuation",
    ]
    return any(keyword in normalized for keyword in quant_keywords)

def supervisor_node(state: AgentState):
    _supervisor_stats["call_count"] += 1
    
    print(f"\n[Portfolio Manager] Reviewing state and delegating... (iteration {_supervisor_stats['call_count']}/{MAX_SUPERVISOR_ITERATIONS})")
    
    # --- CIRCUIT BREAKER: Force to Drafter if too many iterations ---
    if _supervisor_stats["call_count"] >= MAX_SUPERVISOR_ITERATIONS:
        print("   -> ⚠️ CIRCUIT BREAKER: Max supervisor iterations reached. Forcing route to Drafter.")
        return {"next": "Drafter"}
    
    user_query = _extract_user_query(state)
    requested_sector = _extract_sector_from_query(user_query)
    wants_news = _is_news_request(user_query)
    wants_quant = _is_quant_request(user_query)
    needs_sector = requested_sector is not None

    # --- Robust state checking ---
    # Check what departments have produced output by looking for their tagged messages
    quant_has_spoken = False
    fundamental_has_spoken = False
    has_sector_tickers = False
    has_risk_metrics = False
    has_news = False
    
    for m in state["messages"]:
        content = m.content if isinstance(m.content, str) else str(m.content)
        
        if isinstance(m, AIMessage):
            if "**[Quant Dept]**" in content:
                quant_has_spoken = True
            if "**[Fundamental Dept]**" in content:
                fundamental_has_spoken = True
            if "[FUNDAMENTAL_NEWS_COMPLETE]" in content:
                has_news = True
            if "[FUNDAMENTAL_SECTOR_COMPLETE]" in content:
                has_sector_tickers = True
            if "[QUANT_METRICS_COMPLETE]" in content:
                has_risk_metrics = True
        
        # Check for actual data presence only from AI/Tool outputs, not user query text.
        if isinstance(m, (AIMessage, ToolMessage)):
            if any(kw in content.lower() for kw in ["sharpe ratio", "annualized volatility", "annual return", "risk metrics for"]):
                has_risk_metrics = True
            if any(
                kw in content.lower()
                for kw in [
                    "news for",
                    "title:",
                    "snippet:",
                    "latest news",
                    "key headlines",
                    "headline",
                    "here is the latest news",
                ]
            ):
                has_news = True
            if isinstance(m, AIMessage) and "**[Fundamental Dept]**" in content and any(
                kw in content.lower() for kw in ["representative stocks", "from the s&p", "sector, here are", "for analysis:"]
            ):
                has_sector_tickers = True

    sector_done = (not needs_sector) or has_sector_tickers
    quant_done = (not wants_quant) or has_risk_metrics
    news_done = (not wants_news) or has_news

    if not sector_done:
        print("   -> Delegating task to: Fundamental")
        return {"next": "Fundamental"}

    if not quant_done:
        print("   -> Delegating task to: Quant")
        return {"next": "Quant"}

    if not news_done:
        print("   -> Delegating task to: Fundamental")
        return {"next": "Fundamental"}

    if sector_done and quant_done and news_done:
        print("   -> Delegating task to: Drafter")
        return {"next": "Drafter"}
    
    sys_prompt = f"""You are the Portfolio Manager (Supervisor) of a financial advisory team. Your ONLY job is to decide which department should work next.

    You supervise exactly 3 departments:
    - "Fundamental": Finds stock tickers for a sector, and fetches qualitative news about companies.
    - "Quant": Calculates numerical risk metrics (Sharpe Ratio, Volatility, Returns) and fetches price data.
    - "Drafter": Writes the final executive report combining all findings.

    CURRENT STATE OF WORK COMPLETED:
    - Fundamental Dept has reported: {fundamental_has_spoken}
    - Quant Dept has reported: {quant_has_spoken}
    - Sector tickers have been identified: {has_sector_tickers}
    - Risk metrics have been calculated: {has_risk_metrics}
    - News has been fetched: {has_news}

    ROUTING RULES (follow these in EXACT order — pick the FIRST rule that applies):

    RULE 1: If the user mentions a sector (like "Industrials") AND sector tickers have NOT been identified yet ({has_sector_tickers} is False):
    → Route to "Fundamental" (they will look up the sector tickers)

    RULE 2: If sector tickers HAVE been identified ({has_sector_tickers} is True) AND risk metrics have NOT been calculated ({has_risk_metrics} is False):
    → Route to "Quant" (they will calculate Sharpe Ratio, Volatility, etc.)

    RULE 3: If risk metrics HAVE been calculated ({has_risk_metrics} is True) AND news has NOT been fetched ({has_news} is False) AND the user asked for news:
    → Route to "Fundamental" (they will fetch news for the winning stock)

    RULE 4: If all requested data has been gathered (tickers ✓, metrics ✓, and news ✓ if requested):
    → Route to "Drafter" (they will compile the final report)

    RULE 5: If you are unsure or no rule clearly applies:
    → Route to "Drafter" (better to produce a partial report than loop forever)

    Based on these rules, which department should work next? Respond with exactly one of: "Quant", "Fundamental", or "Drafter".
    """

    messages = [SystemMessage(content=sys_prompt)] + list(state["messages"])
    decision = llm.with_structured_output(Route).invoke(messages)
    
    print(f"   -> Delegating task to: {decision.next}")
    return {"next": decision.next}

def _extract_user_query(state: AgentState) -> str:
    """Extract the original user query from the first HumanMessage."""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            return m.content
    return ""

def _build_prior_context(state: AgentState) -> str:
    """Build a brief summary of what prior departments have found, for injection into worker prompts."""
    context_parts = []
    for m in state["messages"]:
        if isinstance(m, AIMessage) and isinstance(m.content, str):
            if "**[Fundamental Dept]**" in m.content:
                context_parts.append(f"PREVIOUS FINDING FROM FUNDAMENTAL TEAM:\n{m.content}")
            elif "**[Quant Dept]**" in m.content:
                context_parts.append(f"PREVIOUS FINDING FROM QUANT TEAM:\n{m.content}")
    return "\n\n".join(context_parts)


def _extract_quant_metrics_from_text(text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"Risk Metrics for\s+([A-Z\-\.]+)\s*\(Trailing 1-Year\):\s*"
        r"-\s*Expected Annual Return:\s*([^\n]+)\s*"
        r"-\s*Annualized Volatility \(Risk\):\s*([^\n]+)\s*"
        r"-\s*Sharpe Ratio:\s*([^\n]+)",
        re.IGNORECASE,
    )
    metrics = []
    for ticker, annual_return, volatility, sharpe in pattern.findall(text):
        metrics.append(
            {
                "ticker": ticker.strip().upper(),
                "annual_return": annual_return.strip(),
                "volatility": volatility.strip(),
                "sharpe": sharpe.strip(),
            }
        )
    return metrics

def quant_node(state: AgentState):
    user_query = _extract_user_query(state)
    prior_context = _build_prior_context(state)
    
    task_message = f"""USER REQUEST: {user_query}

    {prior_context}

    YOUR TASK: Based on the information above, you must NOW call your tools to calculate the requested quantitative metrics.
    Look at the tickers mentioned in the prior findings above. Pick 3-5 representative tickers and call get_risk_metrics with them.
    DO NOT write a plan. DO NOT describe what you would do. Actually CALL the get_risk_metrics tool RIGHT NOW."""

    messages = [
        SystemMessage(content=QUANT_SYSTEM_PROMPT),
        HumanMessage(content=task_message)
    ]
    final_message = run_agent_with_trace(quant_agent, messages, "QUANT")

    output_marker = ""
    if "RAW_TOOL_OUTPUT:" in final_message and "Risk Metrics for" in final_message:
        output_marker = "\n[QUANT_METRICS_COMPLETE]"

    return {"messages": [AIMessage(content=f"**[Quant Dept]**{output_marker}\n{final_message}")]}

def fundamental_node(state: AgentState):
    user_query = _extract_user_query(state)
    prior_context = _build_prior_context(state)
    requested_sector = _extract_sector_from_query(user_query)
    wants_news = _is_news_request(user_query)
    wants_quant = _is_quant_request(user_query)
    
    has_fundamental = "PREVIOUS FINDING FROM FUNDAMENTAL TEAM" in prior_context
    has_quant = "PREVIOUS FINDING FROM QUANT TEAM" in prior_context
    
    expected_action = "news"

    if wants_news and not wants_quant:
        expected_action = "news"
        task_message = f"""USER REQUEST: {user_query}

    YOUR TASK: This is a DIRECT NEWS REQUEST. You must call get_financial_news with a clean, company/topic-specific query derived from the user request.
    Preserve recency constraints if present (e.g., "past week", "last week").
    DO NOT call get_stocks_by_sector.
    DO NOT run quantitative analysis.
    DO NOT write a plan. Actually CALL get_financial_news RIGHT NOW."""
    elif has_fundamental and has_quant:
        expected_action = "news"
        task_message = f"""USER REQUEST: {user_query}

    {prior_context}

    YOUR TASK: The Quant team has already calculated risk metrics (see above). Look at their results to find which ticker had the BEST Sharpe Ratio (highest number).
    You must NOW call get_financial_news with the FULL company name of that winning stock (e.g. "Caterpillar stock news", NOT just "CAT").
    DO NOT call get_stocks_by_sector. DO NOT write a plan. Actually CALL the get_financial_news tool RIGHT NOW."""
    elif requested_sector:
        expected_action = "sector"
        # First call — need to find sector tickers
        # Extract the sector name from the user query
        task_message = f"""USER REQUEST: {user_query}

    YOUR TASK: The user is asking about a specific sector. You must NOW call get_stocks_by_sector with the sector name to find real ticker symbols.
    For example, if the user mentions "Industrials", call get_stocks_by_sector("Industrials").
    If the user mentions "Consumer Discretionary", call get_stocks_by_sector("Consumer Discretionary").
    DO NOT guess tickers. DO NOT write a plan. Actually CALL the get_stocks_by_sector tool RIGHT NOW with the sector name from the user's request."""
    else:
        expected_action = "news"
        task_message = f"""USER REQUEST: {user_query}

    YOUR TASK: Fetch qualitative company news that directly answers the user.
    Call get_financial_news with a clear query phrase based on the user request.
    DO NOT call get_stocks_by_sector unless the user explicitly asked for sector constituents.
    DO NOT write a plan. Actually CALL get_financial_news RIGHT NOW."""

    messages = [
        SystemMessage(content=FUNDAMENTAL_SYSTEM_PROMPT),
        HumanMessage(content=task_message)
    ]
    final_message = run_agent_with_trace(fundamental_agent, messages, "FUNDAMENTAL")

    # Deterministic fallback: if agent failed to call required tool, call it directly.
    if "RAW_TOOL_OUTPUT:" not in final_message:
        if expected_action == "sector" and requested_sector:
            fallback_output = get_stocks_by_sector.invoke(requested_sector)
            final_message = (
                f"RAW_TOOL_OUTPUT:\n{fallback_output}\n\n"
                f"ANALYST_SUMMARY:\nFetched Consumer/sector constituents for '{requested_sector}'."
            )
        elif expected_action == "news":
            news_query = user_query
            quant_metrics = _extract_quant_metrics_from_text(prior_context)
            if quant_metrics:
                def _safe_float(value: str) -> float:
                    try:
                        return float(value.replace("%", "").strip())
                    except ValueError:
                        return float("-inf")

                winner = max(quant_metrics, key=lambda m: _safe_float(m["sharpe"]))
                news_query = f"{winner['ticker']} stock news"

            fallback_output = get_financial_news.invoke(news_query)
            final_message = (
                f"RAW_TOOL_OUTPUT:\n{fallback_output}\n\n"
                f"ANALYST_SUMMARY:\nFetched latest financial news using query: {news_query}"
            )

    output_marker = ""
    if expected_action == "sector" and "RAW_TOOL_OUTPUT:" in final_message:
        output_marker = "\n[FUNDAMENTAL_SECTOR_COMPLETE]"
    elif expected_action == "news" and "RAW_TOOL_OUTPUT:" in final_message and "News for" in final_message:
        output_marker = "\n[FUNDAMENTAL_NEWS_COMPLETE]"

    return {"messages": [AIMessage(content=f"**[Fundamental Dept]**{output_marker}\n{final_message}")]}

def auditor_node(state: AgentState):
    print("\n[Auditor] Reviewing draft for Quality Assurance...")
    draft = state["messages"][-1].content
    user_query = _extract_user_query(state)
    wants_quant = _is_quant_request(user_query)
    comparison_requested = wants_quant or any(
        keyword in _normalize_text(user_query)
        for keyword in ["compare", "comparison", "vs", "versus", "best", "winner"]
    )
    
    # --- THE CIRCUIT BREAKER ---
    # Count how many times the Auditor has already issued a critique
    critique_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage) and "AUDITOR CRITIQUE" in m.content)
    
    if critique_count >= 2:
        print("   -> ⚠️ CIRCUIT BREAKER TRIPPED: Max revisions reached. Forcing delivery to prevent infinite loop.")
        return {"next": "FINISH"}

    draft_text = draft if isinstance(draft, str) else str(draft)
    has_forbidden_html = bool(re.search(r"<\s*/?\s*(table|tr|td|th|div|span|script|style|b|i|strong|em)\b", draft_text, re.IGNORECASE))
    has_raw_json_like = "{\"name\":" in draft_text or "<|python_tag|>" in draft_text
    has_placeholder_metrics = bool(re.search(r"\b(N/A|TBD|unavailable)\b", draft_text, re.IGNORECASE))
    has_markdown_table = "|" in draft_text and bool(re.search(r"\|\s*-{3,}", draft_text))

    if comparison_requested:
        if has_forbidden_html:
            critique = "Remove raw HTML tags from the draft. Use plain Markdown only."
            print(f"   -> ❌ FAIL: {critique}. Sending back to Drafter.")
            return {"messages": [HumanMessage(content=f"AUDITOR CRITIQUE: Fix this immediately: {critique}")], "next": "Drafter"}
        if has_raw_json_like:
            critique = "Remove raw JSON/tool metadata from the report body."
            print(f"   -> ❌ FAIL: {critique}. Sending back to Drafter.")
            return {"messages": [HumanMessage(content=f"AUDITOR CRITIQUE: Fix this immediately: {critique}")], "next": "Drafter"}
        if not has_markdown_table:
            critique = "Add a valid Markdown comparison table for requested quantitative comparison."
            print(f"   -> ❌ FAIL: {critique}. Sending back to Drafter.")
            return {"messages": [HumanMessage(content=f"AUDITOR CRITIQUE: Fix this immediately: {critique}")], "next": "Drafter"}
        if has_placeholder_metrics:
            critique = "Replace placeholder metric values with actual values from Quant output, or omit unavailable fields."
            print(f"   -> ❌ FAIL: {critique}. Sending back to Drafter.")
            return {"messages": [HumanMessage(content=f"AUDITOR CRITIQUE: Fix this immediately: {critique}")], "next": "Drafter"}
        print("   -> ✅ PASS: Deterministic checks passed for comparison report.")
        return {"next": "FINISH"}

    if has_forbidden_html or has_raw_json_like:
        critique = "Remove raw HTML tags and raw metadata from the report."
        print(f"   -> ❌ FAIL: {critique}. Sending back to Drafter.")
        return {"messages": [HumanMessage(content=f"AUDITOR CRITIQUE: Fix this immediately: {critique}")], "next": "Drafter"}

    sys_prompt = f"""You are a strict Quality Assurance Auditor for a financial advisory firm. Your job is to review draft reports before they are delivered to clients.

    USER REQUEST: {user_query}
    COMPARISON REQUESTED: {comparison_requested}

    REVIEW CRITERIA (the draft MUST meet ALL of these):
    1. If COMPARISON REQUESTED is true, the draft MUST contain a cleanly formatted Markdown table using the '|' character.
    A valid Markdown table looks like this:
    | Ticker | Sharpe Ratio | Volatility |
    |--------|-------------|------------|
    | CAT    | 1.50        | 25.00%     |
    If COMPARISON REQUESTED is false, do NOT require a table.
    2. The draft must NOT contain any raw HTML tags like <table>, <tr>, <td>, etc.
    3. The draft must NOT contain raw JSON or bracketed metadata like {{"name": ...}} or <|python_tag|>.
    4. The draft should contain actual data values, not placeholders like "N/A" or "TBD" for key metrics IF those metrics are included.
    5. For direct news-only requests, the answer should focus on requested company/topic news and should not invent unrelated winner/comparison sections.

    GRADING:
    - If the draft meets ALL criteria → set pass_audit to true and leave critique empty.
    - If the draft fails ANY criterion → set pass_audit to false and explain specifically what needs to be fixed.
    """
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=f"DRAFT:\n{draft}")]
    
    evaluation = llm.with_structured_output(Grade).invoke(messages)
    
    if evaluation.pass_audit:
        print("   -> ✅ PASS: Draft is pristine. Ready for delivery.")
        return {"next": "FINISH"}
    else:
        print(f"   -> ❌ FAIL: {evaluation.critique}. Sending back to Drafter.")
        return {
            "messages": [HumanMessage(content=f"AUDITOR CRITIQUE: Fix this immediately: {evaluation.critique}")],
            "next": "Drafter"
        }

def drafter_node(state: AgentState):
    print("\n[Drafter] Compiling the final executive report...")
    
    user_query = _extract_user_query(state)
    prior_context = _build_prior_context(state)
    wants_news = _is_news_request(user_query)
    wants_quant = _is_quant_request(user_query)
    direct_news_only = wants_news and not wants_quant
    extracted_quant_metrics = _extract_quant_metrics_from_text(prior_context)
    
    sys_prompt = f"""You are the Executive Report Drafter for a financial advisory firm. Your job is to take the raw findings from the Quant and Fundamental teams and compile them into a polished, client-ready report.

    USER REQUEST: {user_query}
    DIRECT_NEWS_ONLY_MODE: {direct_news_only}

    INSTRUCTIONS:
    1. Read through the PRIOR FINDINGS below to find the data from the Quant team (risk metrics, prices) and the Fundamental team (sector tickers, news).
    2. Synthesize this data into a clear, professional executive report.

    CRITICAL FORMATTING RULES:
    1. If DIRECT_NEWS_ONLY_MODE is true:
    - Do NOT include quantitative comparison sections, winner sections, or buy/sell recommendation sections.
    - Write a concise news brief that directly answers the user's query.
    - Include a "Key Headlines" section and a short "Takeaway" section.

    2. If DIRECT_NEWS_ONLY_MODE is false and there is numerical comparison data (like Sharpe Ratios for multiple stocks), present it as a Markdown table.
    Use EXACTLY this format:

    | Ticker | Sharpe Ratio | Volatility | Annual Return |
    |--------|-------------|------------|---------------|
    | CAT    | 1.50        | 25.00%     | 15.20%        |
    | HON    | 1.20        | 22.00%     | 12.50%        |

    3. DO NOT use HTML tags like <table>, <tr>, <td>. Only use Markdown pipe tables.
    4. DO NOT include raw JSON, code blocks, or tool call metadata.
    5. DO NOT invent data. Only use numbers that appear in the Quant team's report.
    6. If a metric is unavailable, omit that column/field instead of writing N/A for key comparison values.
    7. If you include a winner/recommendation section, it must be because the user asked for comparative or investment analysis.

    REPORT STRUCTURE GUIDANCE:
    - For DIRECT_NEWS_ONLY_MODE: "## Amazon News Update", "### Key Headlines", "### Takeaway"
    - For mixed/quant requests: sector analysis + quantitative comparison + winner + news + recommendation
    """
    
    if direct_news_only:
        task_message = f"""USER REQUEST: {user_query}

    {prior_context}

    Now compile a concise, user-focused NEWS BRIEF only. Do not include stock comparison tables, winners, or safe-buy recommendations unless explicitly requested."""
    else:
        metrics_block = ""
        if extracted_quant_metrics:
            rows = [
                f"- {m['ticker']}: Annual Return={m['annual_return']}, Volatility={m['volatility']}, Sharpe={m['sharpe']}"
                for m in extracted_quant_metrics
            ]
            metrics_block = "\n\nEXTRACTED QUANT METRICS (USE THESE EXACT VALUES):\n" + "\n".join(rows)

        task_message = f"""USER REQUEST: {user_query}

    {prior_context}
    {metrics_block}

    Now compile the final executive report based on the findings above. Use the EXACT numbers from the Quant team. Format the comparison as a Markdown table."""

    # Check for any auditor critiques to include
    critiques = [m.content for m in state["messages"] if isinstance(m, HumanMessage) and "AUDITOR CRITIQUE" in m.content]
    if critiques:
        task_message += f"\n\nNOTE: The auditor rejected your previous draft. Fix this issue: {critiques[-1]}"

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=task_message)
    ]
    response = llm.invoke(messages)
        
    return {"messages": [AIMessage(content=response.content)]}


workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Quant", quant_node)
workflow.add_node("Fundamental", fundamental_node)
workflow.add_node("Drafter", drafter_node)
workflow.add_node("Auditor", auditor_node)

workflow.add_edge(START, "Supervisor")

# Supervisor routing
workflow.add_conditional_edges("Supervisor", lambda state: state["next"], {
    "Quant": "Quant",
    "Fundamental": "Fundamental",
    "Drafter": "Drafter"
})

workflow.add_edge("Quant", "Supervisor")
workflow.add_edge("Fundamental", "Supervisor")
workflow.add_edge("Drafter", "Auditor")

# Auditor routing
workflow.add_conditional_edges("Auditor", lambda state: state["next"], {
    "Drafter": "Drafter",
    "FINISH": END
})

master_system = workflow.compile()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Self-Correcting/Auditing Multi-Agent Environment")
    print("="*60 + "\n")
    
    query = input("Enter Query: ")
    print(f"User: {query}\n")
    
    # Reset the supervisor iteration counter for each new run
    _supervisor_stats["call_count"] = 0
    
    final_state = master_system.invoke({"messages": [HumanMessage(content=query)]})
    
    print("\n" + "="*20 + " FINAL DELIVERED REPORT " + "="*20 + "\n")
    print(final_state["messages"][-1].content)
    print("\n" + "="*64)