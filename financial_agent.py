from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
# Load environment variables (including your GROQ_API_KEY)
load_dotenv()
phi.api.groq_api_key = os.getenv("GROQ_API_KEY")

# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role='Search the web for information',
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True
)

# Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    role="You are a financial expert who can provide information on stocks, bonds, and mutual funds",
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True
)

# Multi-agent team
multi_ai_agent = Agent(
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    team=[
        web_search_agent,
        finance_agent
    ],
    instructions=[
        "Always include sources",
        "Use tables to display the data"
    ],
    show_tools_calls=True,
    markdown=True
)

# Run the multi-agent with a sample query
multi_ai_agent.print_response(
    "Summarize analyst recommendation and share the latest news for NVDA",
    stream=True
)