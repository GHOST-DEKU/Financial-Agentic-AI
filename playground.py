import os
import phi
import phi.api
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv

load_dotenv()
# Set Grok API key
phi.api.groq_api_key = os.getenv("GROQ_API_KEY")
# Set phidata API key
phi.api.api_key = os.getenv("PHI_API_KEY")

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    stream=False,                     # ← disable streaming at the Agent level
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools()],
    stream=False,                     # ← disable streaming here as well
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(
    agents=[
        finance_agent,
        web_search_agent
    ]
).get_app()

if __name__ == "__main__":
    serve_playground_app(
        "playground:app",
        reload=True
    )