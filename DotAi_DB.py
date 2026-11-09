from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


# Initialize LLM
llm = ChatOpenAI(
    model="auto",
    base_url="http://127.0.0.1:31415/v1",
    api_key=os.getenv("FREELLMAPI_API_KEY"),
    temperature=0
)


# Define chatbot state
class Chatbot(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define chatbot node
def chat_node(state: Chatbot):

    # Get messages from state
    messages = state['messages']

    # Send messages to LLM
    response = llm.invoke(messages)

    # Return AI response
    return {'messages': [response]}


conn=sqlite3.connect("DotAi_DB.db",check_same_thread=False)
# Create memory
checkpointer = SqliteSaver(conn)


# Create graph
graph = StateGraph(Chatbot)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


# Compile workflow
workflow = graph.compile(checkpointer=checkpointer)
def get_all_threads():
    all_threads = set()

    for ckpt in checkpointer.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)