import os
import operator
import time
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send 
from langgraph.prebuilt import create_react_agent

# IMPORT YOUR CUSTOM TOOLS
from tools import get_stock_price, get_financial_statements

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

# ==========================================
# 1. STATE DEFINITIONS
# ==========================================
class OverallState(TypedDict):
    tickers: list[str]
    analyses: Annotated[list[str], operator.add]
    final_report: str

class WorkerState(TypedDict):
    ticker: str

# ==========================================
# 2. THE NODES
# ==========================================
def map_tickers(state: OverallState):
    print(f"\n🔀 [Router] Fanning out to {len(state['tickers'])} parallel autonomous agents...")
    return [Send("analyze_stock", {"ticker": t}) for t in state["tickers"]]

def analyze_stock(state: WorkerState):
    ticker = state["ticker"]
    print(f"   ⚙️ [Agent Boot] Spinning up researcher for {ticker}...")
    
    worker_tools = [get_stock_price, get_financial_statements]
    
    # We create a ReAct agent ON THE FLY for this specific thread
    agent = create_react_agent(llm, tools=worker_tools)
    
    # FIX 2: We must format this as a HumanMessage, NOT a SystemMessage, or Gemini crashes.
    task_prompt = (
        f"You are a quantitative researcher analyzing {ticker}. "
        f"1. Call get_stock_price to find the current price and PE. "
        f"2. Call get_financial_statements to find the latest revenue and net income. "
        f"3. Output a clean, 2-sentence summary containing all the numbers you found."
    )
    
    # Execute the autonomous loop with a HumanMessage
    result = agent.invoke({"messages": [HumanMessage(content=task_prompt)]})
    final_answer = result["messages"][-1].content
    
    print(f"   ✅ [Agent Complete] {ticker} data secured!")
    
    # Append the result to the master list
    return {"analyses": [f"[{ticker}]\n{final_answer}"]}

def reducer_node(state: OverallState):
    print("\n📥 [Reducer] All agents returned successfully. Compiling Master Report...")
    
    combined_analyses = "\n\n".join(state["analyses"])
    
    sys_prompt = (
        "You are a Senior Portfolio Manager. You have received several isolated stock reports from your autonomous agents. "
        "Synthesize them into a single, cohesive 1-paragraph market summary covering all the assets and their fundamental data."
    )
    
    response = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Agent Reports:\n{combined_analyses}")
    ])
    
    return {"final_report": response.content}

# ==========================================
# 3. GRAPH COMPILATION
# ==========================================
workflow = StateGraph(OverallState)

workflow.add_node("analyze_stock", analyze_stock)
workflow.add_node("reducer_node", reducer_node)

workflow.add_conditional_edges(START, map_tickers)
workflow.add_edge("analyze_stock", "reducer_node")
workflow.add_edge("reducer_node", END)

parallel_tool_system = workflow.compile()

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("⚡ LangGraph Parallel Multi-Agent Swarm Online")
    print("="*60 + "\n")
    
    # Our targets (Using your actual yfinance tools now!)
    portfolio = ["AAPL", "MSFT", "NVDA", "CAT"]
    
    start_time = time.time()
    
    final_state = parallel_tool_system.invoke({"tickers": portfolio})
    
    end_time = time.time()
    
    print("\n" + "="*20 + " MASTER PORTFOLIO REPORT " + "="*20)
    print(final_state["final_report"])
    print("="*64)
    print(f"⏱️ Total Execution Time for {len(portfolio)} agents: {round(end_time - start_time, 2)} seconds")