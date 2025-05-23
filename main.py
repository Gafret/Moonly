import datetime

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool(return_direct=True)
def get_current_time() -> dict:
    """Return the current UTC time in ISO‑8601 format.
    Example → {"utc": "2025‑05‑21T06:42:00Z"}"""

    return {"utc": str(datetime.datetime.now())}


tools = [get_current_time]

graph = create_react_agent(
    model="google_genai:gemini-2.0-flash",
    tools=tools,
)
