import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import RemoveMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


@tool(parse_docstring=True)
def get_current_time() -> dict:
    """Returns 'utc_time', 'local_time', 'utc_hour', 'local_hour'."""

    local = datetime.datetime.now(tz=ZoneInfo("Europe/Amsterdam"))
    utc = datetime.datetime.now(datetime.timezone.utc)

    return {
        "utc_time": str(utc),
        "utc_hour": utc.hour,
        "local": str(local),
        "local_hour": local.hour,
    }


@tool(parse_docstring=True)
def recommend_refreshment(local_hour: int) -> dict:
    """Given 'local_hour' suggests what user should eat and drink.

    Args:
        local_hour: local hour.
    """

    if local_hour < 17:
        return {"drink": "coffee", "snack": "cookies"}
    else:
        return {"drink": "kefir", "snack": "cookies"}


AGENT_PROMPT = """
    You are a helpful assistant, 
    if you are asked about a refreshment, snack or drink 
    first call 'get_current_time' tool to define time
"""


class CustomState(MessagesState):
    summary: str


model = init_chat_model(model="google_genai:gemini-2.0-flash")
model_with_tools = model.bind_tools([get_current_time, recommend_refreshment])


def summarize_conversation(state: CustomState):
    messages = state["messages"]

    # not really elegant to check condition in node itself,
    # but it makes more sense than adding conditional edge that leads to
    # summarizer and agent, because we would need to check message history anyway in summarizer
    # node so we do not summarize just one message
    if len(messages) > 5:
        summary = state.get("summary")

        if summary is not None:
            summary_message = (
                f"This is a summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )

        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}


def call_model(state: CustomState):
    summary = state.get("summary")

    if summary is not None:

        summary_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=summary_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = model_with_tools.invoke([SystemMessage(content=AGENT_PROMPT)] + messages)
    return {"messages": response}


tools = ToolNode(tools=[get_current_time, recommend_refreshment])
builder = StateGraph(CustomState)

builder.add_node("agent", call_model)
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_node("tools", tools)

builder.add_edge(START, "summarize_conversation")
builder.add_edge("summarize_conversation", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "summarize_conversation")

graph = builder.compile()
