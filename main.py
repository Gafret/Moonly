import datetime

from dotenv import load_dotenv
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from zoneinfo import ZoneInfo

load_dotenv()


def delete_excessive_messages(state):
    messages = state["messages"]
    if len(messages) > 5:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:5]]}


@tool(parse_docstring=True)
def get_current_time() -> dict:
    """Return both UTC and Europe/Amsterdam full time and just hours."""

    local = datetime.datetime.now(tz=ZoneInfo("Europe/Amsterdam"))
    utc = datetime.datetime.now(datetime.timezone.utc)

    return {
        "utc_time": utc,
        "utc_hour": utc.hour,
        "local": local,
        "local_hour": local.hour,
    }


@tool(parse_docstring=True)
def recommend_refreshment(hour: int) -> dict:
    """Based on current local hour gotten from get_current_time decides what user will eat and drink.

    Args:
        hour: local hour.
    """

    if hour < 17:
        return {"drink": "coffee", "snack": "cookies"}
    else:
        return {"drink": "kefir", "snack": "cookies"}


tools = [get_current_time, recommend_refreshment]

graph = create_react_agent(
    model="google_genai:gemini-2.0-flash",
    tools=tools,
    post_model_hook=delete_excessive_messages,
)
