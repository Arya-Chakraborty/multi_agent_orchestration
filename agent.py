import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import get_stock_data, get_financial_news, get_risk_metrics, get_portfolio_correlation, get_stocks_by_sector
from langgraph.checkpoint.sqlite import SqliteSaver


# 2. Initialize the Brain (LLM) using Gemini
# Setting temperature to 0 makes the model more deterministic and analytical
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

# 3. Define the tool registry
tools = [get_stock_data, get_financial_news, get_risk_metrics, get_portfolio_correlation, get_stocks_by_sector]

system_instruction = """You are a Senior Quantitative Research Analyst. 
Your primary job is to synthesize market data and news into clear, actionable intelligence.

Adhere to these strict rules:
1. Never guess or hallucinate financial numbers. If you cannot find the data using your tools, state explicitly that the data is unavailable.
2. Be incredibly concise. Executives do not have time for fluff.
3. When comparing multiple assets, ALWAYS format your final comparison as a Markdown table.
4. End every analysis with a one-sentence summary of the prevailing market sentiment based on the news."""

if __name__ == "__main__":
    print("\n" + "="*50)
    print("📈 Quantitative Research Agent Initialized (Bulletproof Loop)")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")
    
    # We use a brand new thread ID to ensure no old database ghosts haunt us
    config = {"configurable": {"thread_id": "research_session_4"}}

    with SqliteSaver.from_conn_string("agent_memory.db") as memory:
        
        # THE FIX: We initialize the agent barebones. No fragile kwargs.
        agent_executor = create_react_agent(
            model=llm, 
            tools=tools, 
            checkpointer=memory
        )

        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("Shutting down agent...")
                break
                
            print("\n=================== EXECUTION TRACE ===================")
            
            # THE FIX: We bundle the SystemMessage directly with your input
            input_messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=user_input)
            ]
            
            # Stream the values natively
            for event in agent_executor.stream(
                {"messages": input_messages},
                config=config,
                stream_mode="values"
            ):
                message = event["messages"][-1]

                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        print("\nTHINKING: I need to fetch external data.")
                        for tc in message.tool_calls:
                            print(f"   -> Action: Calling '{tc['name']}' with args: {tc['args']}")
                    
                    # We use 'if' instead of 'elif' so we never skip the text output
                    if message.content:
                        print("\nSYNTHESIZING FINAL ANSWER:")
                        print(f"   -> {message.content[0]['text']}")

                elif isinstance(message, ToolMessage):
                    obs_text = str(message.content)
                    display_text = obs_text[:400] + "... [truncated]" if len(obs_text) > 400 else obs_text
                    print(f"\nOBSERVATION (Tool Output):\n{display_text}")
                    print("-" * 55)

            print("\n======================= END TRACE =======================")