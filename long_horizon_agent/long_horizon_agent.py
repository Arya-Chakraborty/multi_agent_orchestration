import os
import operator
import re
from typing import Annotated, List, Tuple, TypedDict

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from tools import (
    calculate_advanced_math,
    compare_stock_price_changes,
    get_financial_news,
    get_financial_statements,
    get_stock_price,
    get_stock_risk_metrics,
    web_search,
)


COMPANY_TO_TICKER = {
    "microsoft": "MSFT",
    "apple": "AAPL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "alphabet": "GOOG",
    "google": "GOOG",
    "meta": "META",
    "tesla": "TSLA",
}

planner_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite-preview", temperature=0)
executor_llm = ChatOllama(model="llama3.1", temperature=0)

executor_tools = [
    calculate_advanced_math,
    web_search,
    get_stock_risk_metrics,
    get_financial_news,
    compare_stock_price_changes,
]

executor_agent = create_react_agent(executor_llm, tools=executor_tools)


class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str
    cycle_count: int


class Plan(BaseModel):
    objective_summary: str = Field(description="Single sentence objective interpretation.")
    steps: List[str] = Field(description="Strict step-by-step plan to solve the objective.")


class ReplannerOutput(BaseModel):
    is_finished: bool = Field(description="True ONLY if the objective is fully solved.")
    final_response: str = Field(description="Comprehensive final response if finished, else empty.")
    new_plan: List[str] = Field(description="Remaining actionable tasks if not finished.")
    rationale: str = Field(description="One concise reason for this decision.")


def _build_fallback_plan(user_input: str) -> List[str]:
    lowered = user_input.lower()
    steps: List[str] = []

    tickers = _extract_tickers(user_input)
    ticker_list = ", ".join(tickers) if tickers else "provided tickers"

    if any(word in lowered for word in ["risk", "sharpe", "volatility", "compare"]):
        steps.append(f"Get trailing 1-year risk metrics for {ticker_list}")

    if any(word in lowered for word in ["value", "price", "current"]):
        steps.append(f"Get latest stock prices for {ticker_list}")

    if "news" in lowered:
        steps.append(f"Fetch latest financial news for {ticker_list}")

    added_generic_price_step = any("latest stock prices" in step.lower() for step in steps)
    if not added_generic_price_step:
        if "msft" in lowered:
            steps.append("Get current price of MSFT")
        if "aapl" in lowered:
            steps.append("Get current price of AAPL")

    if "interest rate" in lowered or "federal" in lowered or "fed" in lowered:
        steps.append("Get current US policy rate")

    if "100 shares" in lowered or "50 shares" in lowered or "cost" in lowered or "total" in lowered:
        steps.append("Calculate total cost using retrieved prices and share counts")

    if "above 5%" in lowered or "greater than 5%" in lowered:
        steps.append("Determine whether the policy rate is above 5%")

    if any(word in lowered for word in ["safer", "safe", "risk-adjusted"]):
        steps.append("Analyze all evidence and determine the safer stock")
    else:
        steps.append("Synthesize final response using all retrieved results")
    deduped = []
    seen = set()
    for step in steps:
        key = step.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(step)
    return deduped


def _sanitize_plan_steps(raw_steps: List[str], user_input: str) -> List[str]:
    cleaned = [step.strip() for step in raw_steps if step and step.strip()]
    bad_prefixes = ("title_", "step_", "task_")
    unsupported_markers = [
        "alpha vantage",
        "google news api",
        "store the data in a pandas dataframe",
        "python code",
        "nlp",
        "sentiment analysis",
        "api key",
    ]

    def _is_bad(step: str) -> bool:
        normalized = step.lower().strip()
        if normalized.startswith(bad_prefixes):
            return True
        if normalized in {"title", "step", "task"}:
            return True
        if any(marker in normalized for marker in unsupported_markers):
            return True
        return False

    if not cleaned or any(_is_bad(step) for step in cleaned):
        return _build_fallback_plan(user_input)
    return cleaned


def _extract_tickers(text: str) -> List[str]:
    discovered = []
    uppercase_matches = re.findall(r"\b[A-Z]{1,5}\b", text)
    for match in uppercase_matches:
        if match not in {"AND", "THE", "FOR", "WITH", "FROM", "THEN"} and match not in discovered:
            discovered.append(match)

    lowered = text.lower()
    for company_name, ticker in COMPANY_TO_TICKER.items():
        if company_name in lowered and ticker not in discovered:
            discovered.append(ticker)

    return discovered


def _extract_prices_from_ledger(past_steps: List[Tuple[str, str]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    pattern = re.compile(r"Ticker:\s*([A-Z\-\.]+).*?Latest Close:\s*([0-9]+(?:\.[0-9]+)?)", re.DOTALL)
    for _, result in past_steps:
        if not result:
            continue
        for ticker, price in pattern.findall(result):
            try:
                prices[ticker.upper()] = float(price)
            except ValueError:
                continue
    return prices


def _extract_policy_rate_from_ledger(past_steps: List[Tuple[str, str]]) -> float | None:
    pct_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
    for _, result in reversed(past_steps):
        if not result:
            continue
        lower = result.lower()
        if "policy rate" in lower or "federal funds" in lower or "interest rate" in lower:
            matches = pct_pattern.findall(result)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    return None
    return None


def _extract_share_requests(objective: str) -> List[Tuple[str, int]]:
    matches = re.findall(r"(\d+)\s+shares?\s+of\s+([A-Za-z\.\-]+)", objective, flags=re.IGNORECASE)
    parsed: List[Tuple[str, int]] = []
    for count_text, symbol_or_name in matches:
        ticker = symbol_or_name.upper()
        if ticker not in COMPANY_TO_TICKER.values():
            ticker = COMPANY_TO_TICKER.get(symbol_or_name.lower(), ticker)
        try:
            parsed.append((ticker, int(count_text)))
        except ValueError:
            continue
    return parsed


def _execute_task_deterministically(task: str, objective: str, past_steps: List[Tuple[str, str]]) -> str | None:
    lower_task = task.lower()
    task_tickers = _extract_tickers(task)
    objective_tickers = _extract_tickers(objective)
    tickers = task_tickers if task_tickers else objective_tickers
    if not tickers:
        tickers = ["MSFT", "AAPL"]

    if "risk" in lower_task or "sharpe" in lower_task or "volatility" in lower_task:
        period = "1y"
        period_match = re.search(r"\b(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)\b", lower_task)
        if period_match and period_match.group(1) in SUPPORTED_PERIODS:
            period = period_match.group(1)
        ticker_string = ", ".join(tickers)
        print("   [Deterministic Executor] Calling get_stock_risk_metrics")
        print(f"      -> Tool: get_stock_risk_metrics | Args: {{'tickers_string': '{ticker_string}', 'period': '{period}'}}")
        observation = get_stock_risk_metrics.invoke({"tickers_string": ticker_string, "period": period})
        print("   [Tool Observation]")
        print(f"      {_trim_text(str(observation), 700)}")
        return str(observation)

    if "price" in lower_task or "value" in lower_task:
        outputs = []
        for ticker in tickers:
            print(f"   [Deterministic Executor] Calling get_stock_price for {ticker}")
            print(f"      -> Tool: get_stock_price | Args: {{'ticker': '{ticker}'}}")
            observation = get_stock_price.invoke({"ticker": ticker})
            print("   [Tool Observation]")
            print(f"      {_trim_text(str(observation), 700)}")
            outputs.append(str(observation))
        return "\n-------------------\n".join(outputs)

    if "news" in lower_task:
        outputs = []
        for ticker in tickers:
            news_query = f"{ticker} stock news"
            print(f"   [Deterministic Executor] Calling get_financial_news for {ticker}")
            print(f"      -> Tool: get_financial_news | Args: {{'query': '{news_query}', 'timelimit': 'w', 'max_results': 3}}")
            observation = get_financial_news.invoke({"query": news_query, "timelimit": "w", "max_results": 3})
            print("   [Tool Observation]")
            print(f"      {_trim_text(str(observation), 700)}")
            outputs.append(str(observation))
        return "\n-------------------\n".join(outputs)

    if "policy rate" in lower_task or "interest rate" in lower_task or "federal" in lower_task:
        query = "current US federal funds rate target range percentage"
        print("   [Deterministic Executor] Calling web_search for US policy rate")
        print(f"      -> Tool: web_search | Args: {{'query': '{query}', 'max_results': 5}}")
        observation = web_search.invoke({"query": query, "max_results": 5})
        print("   [Tool Observation]")
        print(f"      {_trim_text(str(observation), 700)}")
        return str(observation)

    if "calculate total cost" in lower_task or "shares" in lower_task:
        share_reqs = _extract_share_requests(objective)
        if not share_reqs:
            return "Could not find share-count instructions in objective."
        prices = _extract_prices_from_ledger(past_steps)
        terms = []
        missing = []
        for ticker, count in share_reqs:
            if ticker not in prices:
                missing.append(ticker)
            else:
                terms.append(f"({count}*{prices[ticker]})")
        if missing:
            return f"Missing prices for: {', '.join(missing)}. Fetch prices first."
        expression = " + ".join(terms)
        print("   [Deterministic Executor] Calling calculate_advanced_math for portfolio total")
        print(f"      -> Tool: calculate_advanced_math | Args: {{'expression': '{expression}'}}")
        observation = calculate_advanced_math.invoke({"expression": expression})
        print("   [Tool Observation]")
        print(f"      {_trim_text(str(observation), 700)}")
        return str(observation)

    if "above 5%" in lower_task or "greater than 5%" in lower_task or "determine whether" in lower_task:
        rate = _extract_policy_rate_from_ledger(past_steps)
        if rate is None:
            return "Could not determine policy rate from prior results."
        return f"Current inferred policy rate: {rate:.2f}%. Above 5%: {'Yes' if rate > 5 else 'No'}."

    if "safer" in lower_task or "analyze" in lower_task or "synthesize" in lower_task:
        return "Synthesis step reserved for final report generation."

    return None


def _build_final_response(state: PlanExecuteState) -> str:
    objective = state.get("input", "")
    ledger = state.get("past_steps", [])
    prices = _extract_prices_from_ledger(ledger)

    metrics_pattern = re.compile(
        r"Risk Metrics for\s+([A-Z\-\.]+).*?Expected Annual Return:\s*([^\n]+).*?Annualized Volatility:\s*([^\n]+).*?Sharpe Ratio:\s*([^\n]+)",
        re.DOTALL,
    )
    metrics: dict[str, dict[str, str]] = {}
    for _, result in ledger:
        for ticker, annual_return, volatility, sharpe in metrics_pattern.findall(result or ""):
            metrics[ticker] = {
                "annual_return": annual_return.strip(),
                "volatility": volatility.strip(),
                "sharpe": sharpe.strip(),
            }

    safer_ticker = None
    best_sharpe = float("-inf")
    for ticker, vals in metrics.items():
        try:
            sharpe_val = float(vals["sharpe"])
            if sharpe_val > best_sharpe:
                best_sharpe = sharpe_val
                safer_ticker = ticker
        except ValueError:
            continue

    lines = [f"Objective: {objective}", "", "Summary of gathered evidence:"]
    if prices:
        lines.append("- Latest prices:")
        for ticker, price in prices.items():
            lines.append(f"  - {ticker}: ${price:.2f}")
    if metrics:
        lines.append("- Risk metrics:")
        for ticker, vals in metrics.items():
            lines.append(
                f"  - {ticker}: Return {vals['annual_return']}, Volatility {vals['volatility']}, Sharpe {vals['sharpe']}"
            )

    news_mentions = []
    for _, result in ledger:
        if isinstance(result, str) and "Top finance news for" in result:
            news_mentions.append(_trim_text(result, 350))
    if news_mentions:
        lines.append("- News snippets were collected for the requested tickers.")

    if safer_ticker:
        lines.append("")
        lines.append(
            f"Safer choice (based on highest observed Sharpe ratio in current run): {safer_ticker}."
        )
    else:
        lines.append("")
        lines.append("Safer choice could not be determined confidently from the collected metrics.")

    return "\n".join(lines)


def _trim_text(text: str, limit: int = 900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def planner_node(state: PlanExecuteState):
    print("\n" + "=" * 56)
    print("[Planner] Drafting master execution plan")
    print("=" * 56)

    sys_prompt = (
        "You are an elite financial strategist. Build a precise, minimal execution plan. "
        "Each step must be one isolated action and should be solvable via one tool interaction or one synthesis action. "
        "Keep steps sequential, observable, and directly tied to the user objective. "
        "Never output placeholder labels like title_1, step_1, task_1."
    )

    planner = planner_llm.with_structured_output(Plan)
    plan = planner.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=state["input"])])

    cleaned_steps = _sanitize_plan_steps(plan.steps, state["input"])
    if not cleaned_steps:
        cleaned_steps = ["Synthesize available evidence and provide best-effort answer."]

    print(f"   Objective Summary: {plan.objective_summary}")
    print("   Plan:")
    for index, step in enumerate(cleaned_steps, start=1):
        print(f"      {index}. {step}")

    return {"plan": cleaned_steps, "past_steps": [], "cycle_count": 0}


def _run_executor_with_trace(task: str) -> str:
    print(f"\n[Executor] Assigned Task: {task}")

    sys_prompt = (
        "You are the execution specialist. Complete ONLY the assigned task. "
        "Use tools when needed, then return a concise factual result. "
        "Do not produce a multi-step plan."
    )

    final_answer = ""
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=task)]

    for event in executor_agent.stream({"messages": messages}, stream_mode="values"):
        message = event["messages"][-1]

        if isinstance(message, AIMessage):
            if message.tool_calls:
                print("   [Executor Thinking] Preparing tool call(s):")
                for tool_call in message.tool_calls:
                    print(f"      -> Tool: {tool_call['name']} | Args: {tool_call['args']}")
            elif message.content:
                preview = _trim_text(str(message.content), 500)
                print(f"   [Executor Synthesis] {preview}")
                final_answer = str(message.content)

        elif isinstance(message, ToolMessage):
            observation = _trim_text(str(message.content), 700)
            print("   [Tool Observation]")
            print(f"      {observation}")

    if not final_answer:
        final_answer = "Executor produced no final synthesis; relying on tool observations captured above."

    print(f"   [Executor Final] {_trim_text(final_answer, 600)}")
    return final_answer


def executor_node(state: PlanExecuteState):
    if not state.get("plan"):
        print("\n[Executor] No remaining tasks. Skipping execution.")
        return {"past_steps": []}

    current_task = state["plan"][0]
    print(f"\n[Executor] Assigned Task: {current_task}")
    execution_result = _execute_task_deterministically(current_task, state.get("input", ""), state.get("past_steps", []))
    if execution_result is None:
        print("   [Executor] No deterministic route matched. Falling back to ReAct executor.")
        execution_result = _run_executor_with_trace(current_task)
    return {"past_steps": [(current_task, execution_result)]}


def replanner_node(state: PlanExecuteState):
    print("\n[Replanner] Evaluating progress and updating plan")

    completed_steps = state.get("past_steps", [])
    cycle_count = state.get("cycle_count", 0) + 1

    if len(completed_steps) >= 3:
        last_three = completed_steps[-3:]
        repeated_task = len({task.lower() for task, _ in last_three}) == 1
        if repeated_task:
            print("   -> ⚠️ Repeated same task detected for 3 cycles. Forcing best-effort completion.")
            return {
                "response": _build_final_response(state),
                "plan": [],
                "cycle_count": cycle_count,
            }

    if cycle_count >= MAX_CYCLES:
        print(f"   -> ⚠️ Max cycle limit ({MAX_CYCLES}) reached. Forcing completion response.")
        return {"response": _build_final_response(state), "plan": [], "cycle_count": cycle_count}

    current_plan = state.get("plan", [])
    remaining_steps = current_plan[1:] if len(current_plan) > 1 else []

    if not remaining_steps:
        print("   Rationale: All queued tasks executed. Finalizing response.")
        return {
            "response": _build_final_response(state),
            "plan": [],
            "cycle_count": cycle_count,
        }

    print("   Rationale: Current step completed; moving to next queued task(s).")

    print("   -> Remaining plan:")
    for index, step in enumerate(remaining_steps, start=1):
        print(f"      {index}. {step}")

    return {"plan": remaining_steps, "cycle_count": cycle_count}


def route_replanner(state: PlanExecuteState):
    if state.get("response"):
        return "FINISH"
    if not state.get("plan"):
        return "FINISH"
    return "Executor"


workflow = StateGraph(PlanExecuteState)
workflow.add_node("Planner", planner_node)
workflow.add_node("Executor", executor_node)
workflow.add_node("Replanner", replanner_node)

workflow.add_edge(START, "Planner")
workflow.add_edge("Planner", "Executor")
workflow.add_edge("Executor", "Replanner")
workflow.add_conditional_edges(
    "Replanner",
    route_replanner,
    {"Executor": "Executor", "FINISH": END},
)

long_horizon_system = workflow.compile()


if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("Autonomous Long-Horizon Agent Online")
    print("=" * 64 + "\n")

    default_query = (
        "Find the current price of MSFT and AAPL. Then, get the current US policy rate. "
        "Finally, calculate what 100 shares of MSFT and 50 shares of AAPL would cost in total, "
        "and tell me if the interest rate is above 5%."
    )

    query = input("Enter objective (press Enter to use default): ").strip()
    if not query:
        query = default_query

    print(f"\nUser Request: {query}")
    final_state = long_horizon_system.invoke(
        {
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": "",
            "cycle_count": 0,
        }
    )

    print("\n" + "=" * 24 + " FINAL DELIVERED REPORT " + "=" * 24 + "\n")
    print(final_state.get("response", "No response generated."))
    print("\n" + "=" * 80)
