# -----------------------------
# IMPORTS
# -----------------------------

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

import os
import sqlite3
import math
import requests


# Load API keys from the .env file
load_dotenv()


# -----------------------------
# INITIALIZE THE LLM
# -----------------------------

llm = ChatOpenAI(
    model="auto",
    base_url="http://127.0.0.1:31415/v1",
    api_key=os.getenv("FREELLMAPI_API_KEY"),
    temperature=0
)


# -----------------------------
# TOOLS
# -----------------------------

# Tavily searches the internet and returns relevant results
search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    topic="general",
    search_depth="advanced"
)


# A simple calculator for mathematical expressions
@tool
def calculator(expression: str) -> str:
    """
    Calculates mathematical expressions.
    Examples: '2 + 2', 'sqrt(16)', '25 * 10'
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len
        }

        result = eval(
            expression,
            {"__builtins__": None},
            allowed
        )

        return str(result)

    except Exception as e:
        return f"Error evaluating expression: {e}"


# Gets the latest stock information using Alpha Vantage
@tool
def get_stock_price(symbol: str) -> dict:
    """Fetches stock information for a given stock symbol."""

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:
        return {"error": "Alpha Vantage API key is missing."}

    url = "https://www.alphavantage.co/query"

    try:
        response = requests.get(
            url,
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key
            },
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        return {"error": f"Stock API request failed: {e}"}


# Gets the current weather for a city
@tool
def get_weather(city: str) -> dict:
    """Fetches current weather information for a given city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return {"error": "OpenWeather API key is missing."}

    url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        response = requests.get(
            url,
            params={
                "q": city,
                "appid": api_key,
                "units": "metric"
            },
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        return {"error": f"Weather API request failed: {e}"}


# Keep all available tools in one list
tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_weather
]


# Give the LLM access to our tools
llm_with_tools = llm.bind_tools(tools)


# -----------------------------
# CHATBOT STATE
# -----------------------------

class Chatbot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# CHATBOT NODE
# -----------------------------

def chat_node(state: Chatbot):

    # Send the conversation history to the LLM
    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    # Add the LLM response to the conversation
    return {"messages": [response]}


# ToolNode automatically executes the tools requested by the LLM
tools_node = ToolNode(tools)


# -----------------------------
# SQLITE DATABASE
# -----------------------------

# Keep the database in the same folder as this Python file
DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "DotAi_DB.db"
)

conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

# LangGraph uses this to save and retrieve conversations
checkpointer = SqliteSaver(conn)

# Create the required database tables
checkpointer.setup()


# -----------------------------
# CREATE LANGGRAPH WORKFLOW
# -----------------------------

graph = StateGraph(Chatbot)

# Register the nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tools_node)

# Start every conversation with the chatbot
graph.add_edge(START, "chat_node")

# Decide whether to execute a tool or finish the conversation
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

# After executing a tool, send its result back to the LLM
graph.add_edge("tools", "chat_node")


# Compile the graph and connect SQLite memory
workflow = graph.compile(
    checkpointer=checkpointer
)


# -----------------------------
# GET ALL SAVED CHAT THREADS
# -----------------------------

def get_all_threads():

    all_threads = set()

    # Each checkpoint contains information about a saved conversation
    for ckpt in checkpointer.list(None):

        thread_id = ckpt.config["configurable"]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)