from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import RemoveMessage, HumanMessage, SystemMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import get_current_time, recommend_refreshment

load_dotenv()

AGENT_PROMPT = """
    You are a helpful assistant, 
    if you are asked about a refreshment, snack or drink 
    first call 'get_current_time' tool to define time
"""


class CustomState(MessagesState):
    summary: str
    turns: int


model = init_chat_model(model="google_genai:gemini-2.0-flash")
model_with_tools = model.bind_tools([get_current_time, recommend_refreshment])


def summarize_conversation(state: CustomState):
    turns = state.get("turns", 0)

    if turns >= 5:
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

        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
        return {"summary": response.content, "messages": delete_messages, "turns": 0}


def call_model(state: CustomState):
    summary = state.get("summary")

    if summary is not None:

        summary_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=summary_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = model_with_tools.invoke([SystemMessage(content=AGENT_PROMPT)] + messages)
    return {"messages": response}


def turn_counter(state: CustomState):
    turns = state.get("turns", 0) + 1

    return {"turns": turns}


tools = ToolNode(tools=[get_current_time, recommend_refreshment])
builder = StateGraph(CustomState)

builder.add_node("agent", call_model)
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_node("tools", tools)
builder.add_node("turn_counter", turn_counter)

builder.add_edge(START, "turn_counter")
builder.add_edge("turn_counter", "summarize_conversation")
builder.add_edge("summarize_conversation", "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "summarize_conversation")

graph = builder.compile()
