# -----------------------------
# IMPORTS
# -----------------------------

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

import sqlite3
import math
import requests
import os


# Load environment variables from .env
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
# Initialize LLM
# llm = ChatOpenAI(
#     model="auto",
#     base_url="http://127.0.0.1:31415/v1",
#     api_key=os.getenv("FREELLMAPI_API_KEY"),
#     temperature=0
# )



# -----------------------------
# TOOLS
# -----------------------------

# Tavily searches the internet
search_tool = TavilySearch(
    max_results=1,
    topic="general",
    search_depth="basic",
    include_raw_content=False,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


# Calculator tool
@tool
def calculator(expression: str) -> str:
    
    """Evaluate a math expression and return the numeric result.

    Use for ANY arithmetic, even simple ones. LLMs get big numbers wrong.
    Input must be a single valid Python expression.
    - Use ** for powers, NOT ^ (in Python, ^ means XOR and gives wrong answers).
    - Use math.sqrt(), math.sin(), etc. for functions.
    - Do not use variables or words, only numbers and operators.

    Examples: 98 + 78237 * 9813**2, math.sqrt(144), (15 / 100) * 2400
    """
        

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# Stock price tool
@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock quote for a ticker symbol.

        Use only when the user asks for a stock's price or daily change.
        Input: the ticker symbol only (e.g. AAPL, TSLA, MSFT), not a company name.
        If the user gives a company name, convert it to its ticker first.

        Returns raw quote data. Report the price ('05. price'), the change
        ('09. change') and the percent change ('10. change percent').
        If the result is empty or contains an error or rate-limit message,
        tell the user. Do not retry.
        """

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


# Weather tool
@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Weather API key is missing."

    try:

        # Step 1: Convert location into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10
        )

        geo_response.raise_for_status()

        locations = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]

        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")


        # Step 2: Get current weather
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()


        # Extract weather information
        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]

        description = weather_data["weather"][0]["description"]

        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")

        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )


        # Prepare the location name
        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)


        # Return formatted weather information
        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )


    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:

        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"


# -----------------------------
# MAKE TOOL LIST
# -----------------------------

tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather
]


# Give the LLM access to the tools
llm_with_tools = llm.bind_tools(tools)


# -----------------------------
# CHATBOT STATE
# -----------------------------

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# CHAT NODE
# -----------------------------

def chat_node(state: ChatState):

    # Instructions for the chatbot
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"

            "- Use search_tool for current events and recent information.\n"

            "- Use calculator for mathematical calculations.\n"

            "- Use get_stock_price when the user asks for stock prices.\n"

            "- Use get_current_weather when the user asks about current weather.\n\n"

            "Answer general questions directly when no tool is required. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    # Send messages to the LLM
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# -----------------------------
# TOOL NODE
# -----------------------------

tool_node = ToolNode(tools)


# -----------------------------
# SQLITE CHECKPOINTER
# -----------------------------

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpoint = SqliteSaver(conn)

# Create the required database tables
checkpoint.setup()


# -----------------------------
# CREATE LANGGRAPH WORKFLOW
# -----------------------------

graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


# Add edges
graph.add_edge(START, "chat_node")

# Decide whether to call a tool or finish
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

# Send tool results back to the chatbot
graph.add_edge("tools", "chat_node")


# Compile the chatbot
chatbot = graph.compile(
    checkpointer=checkpoint
)


# -----------------------------
# GET ALL SAVED THREADS
# -----------------------------

def get_all_threads():

    all_threads = set()

    for ckpt in checkpoint.list(None):

        thread_id = ckpt.config["configurable"]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)