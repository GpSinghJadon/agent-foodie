from app import get_restaurants_tool
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain.agents import AgentType
import json

# Import relevant functionality
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Create the agent
memory = MemorySaver()
# llm = Ollama(
#     base_url="http://192.168.1.3:11434", model="foodie_v0.1:latest"
# )  # Replace with your Ollama setup

from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://192.168.1.3:11433",
    model="agent_foodie_v1:latest",
    # model="llama3.2.1:latest",
    temperature=0,
    num_gpu=1,
    # other params...
)

tools = [
    Tool(
        name="RestaurantSearch",
        func=get_restaurants_tool,
        description="Useful for finding restaurants. Input should be a valid city and optionally cuisine, budget or dish. Example: 'city=New York cuisine=Italian budget=50 dish=pizza'",
    ),
]
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream({"messages": [HumanMessage(content="hi")]}, config):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="I want to find a restaurant in New York")]},
    config,
):
    print(chunk)
    print("----")
