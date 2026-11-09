from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()  # take environment variables from .env.

llm = ChatOpenAI(
    model="auto",
    base_url="http://127.0.0.1:31415/v1",
    api_key=os.getenv("FREELLMAPI_API_KEY"),
    temperature=0
)
class Jokestate(TypedDict):
    topic:str
    joke:str
    explanation:str
    
def gen_joke(state:Jokestate):
    prompt=f"Generate a joke about {state['topic']}."
    response=llm.invoke(prompt)
    return {'joke':response.content}
def exp_joke(state:Jokestate):
    prompt=f"Explain the joke: {state['joke']}."
    response=llm.invoke(prompt)
    return {'explanation':response.content}

#now making a graph
graph=StateGraph(Jokestate)
graph.add_node('gen_joke',gen_joke)
graph.add_node('exp_joke', exp_joke)

#now lets add edges
graph.add_edge(START,'gen_joke')
graph.add_edge('gen_joke', 'exp_joke')
graph.add_edge('exp_joke',END)

CheckpointSaver=InMemorySaver()


workflow=graph.compile(checkpointer=CheckpointSaver)
config={"configurable":{ "thread_id":"1"}}
initial_state={
    "topic":"football"
}
result=workflow.invoke(initial_state,config=config)
print(result)