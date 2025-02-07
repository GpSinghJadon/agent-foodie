from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import StructuredTool, Tool
from langchain_community.llms import Ollama
from langchain.agents import AgentType
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


# Define input schema for the restaurant search
class RestaurantSearchInput(BaseModel):
    # city: str = Field(description="Name of the city to search restaurants.")
    cuisine: Optional[str] = Field(None, description="Type of cuisine")
    budget: Optional[float] = Field(None, description="Budget for two people")
    dish: Optional[str] = Field(None, description="Specific dish name")


# Restaurant data
restaurants_data = [
    {
        "restaurant_id": 1,
        "name": "Pizza on the MARS Place",
        "city": "New York",
        "rating": 4.5,
        "cuisines": ["Italian"],
        "budget_for_2": 30,
        "dishes": ["pizza", "pasta"],
    },
    # ... other restaurant data ...
]


def get_restaurants_tool(params: RestaurantSearchInput) -> str:
    """
    Fetches restaurants based on criteria.

    Args:
        params (RestaurantSearchInput): Search parameters including city, cuisine, budget, and dish

    Returns:
        str: JSON containing the restaurants data
    """
    try:
        if getattr(params, "city"):
            params.city = "New York"
        base_url = "http://localhost:8000/app/v1/restaurants"
        query_params = {"city": "New York"}
        if params.cuisine:
            query_params["cuisine"] = params.cuisine
        if params.budget:
            query_params["budget"] = params.budget
        if params.dish:
            query_params["dish"] = params.dish

        import requests
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching restaurants: {e}"


def websocket_endpoint():
    print("checkpoint 1")
    memory = MemorySaver()

    agent_prompt = """
# Restaurant Search Assistant

You are a friendly AI assistant specialized in helping users find restaurants by using RestaurantSearch tool. Your role is to have natural conversations about restaurant preferences while strictly adhering to the following guidelines:

## Core Behavior

1. Always begin interactions with a warm greeting
2. Ask only ONE question at a time - never request multiple pieces of information simultaneously
3. Never provide restaurant information from memory - rely exclusively on the RestaurantSearch tool
4. Maintain a conversational tone while progressively collecting required information:
   - City (mandatory)
   - Budget (mandatory)
   - Cuisine and/or specific dish (optional)

## Question Sequence

1. If starting fresh, always ask for city first
2. After getting city, ask for budget
3. If neither cuisine nor dish is mentioned, ask for food preferences
4. IMPORTANT: Only proceed to the next question after receiving a clear answer to the current one

## Tool Usage Rules

1. ONLY invoke RestaurantSearch when you have both mandatory parameters:
   - city
   - budget
   
2. Tool Input Format:
   ```
   RestaurantSearch "city={city} budget={budget} cuisine={cuisine} dish={dish}"
   ```
   Example: "city=New York budget=50 cuisine=Italian dish=pizza"

3. If tool returns no results or encounters an error, respond with:
   "Not able to find any matching place right now"

## Conversation Flow Examples

✅ Correct (Single Question):
User: "I want Italian food"
Assistant: "I'd be happy to help you find some Italian restaurants! Which city are you looking to dine in?"

User: "Chicago"
Assistant: "What's your budget per person for the meal?"

❌ Incorrect (Multiple Questions):
Assistant: "I'd be happy to help! Which city are you in and what's your budget? Also, do you have any cuisine preferences?"

## Response Guidelines

1. For successful searches:
   ```
   Here are some restaurants I found:
   1. [Restaurant Name 1]
   2. [Restaurant Name 2]
   3. [Restaurant Name 3]
   ```

2. For no results:
   "I do not have any restaurants matching your criteria."

3. For missing information:
   Ask ONE specific question like:
   - "What's your preferred budget range?"
   OR
   - "Which city are you looking to dine in?"
   OR
   - "Do you have any specific cuisine preferences?"

## Sample Progressive Conversation

User: "I want to find a restaurant"
Assistant: "Hello! I'll help you find a great restaurant. Which city are you looking to dine in?"

User: "San Francisco"
Assistant: "What's your budget per person for the meal?"

User: "$40"
Assistant: "Do you have any specific cuisine preferences?"

User: "Italian"
Assistant: [Uses RestaurantSearch with "city=San Francisco budget=40 cuisine=Italian"]

Remember: Focus on one question at a time, maintain a natural conversation flow, and only use the RestaurantSearch tool with proper parameters.
"""

    llm = ChatOllama(
        base_url="http://192.168.1.6:11433",
        model="agent_foodie_v3_mistral:latest",
        temperature=0.3,
        # num_gpu=1,
        num_predict=10000,
    )

    tools = [
        StructuredTool.from_function(
            func=get_restaurants_tool,
            name="RestaurantSearch",
            description="Useful for finding restaurants based on city, cuisine, budget and dish.",
            args_schema=RestaurantSearchInput,
        )
    ]

    agent_executor = create_react_agent(
        llm, tools, checkpointer=memory, state_modifier=agent_prompt
    )

    try:
        while True:
            data = input("Enter message: ")
            try:
                user_input = data
                if not user_input:
                    raise ValueError(
                        "Invalid message format. Expected {'message': 'user input'}"
                    )
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in message")
            except ValueError as e:
                print(e)
                continue

            try:
                config = {"configurable": {"thread_id": "abc123"}}
                response = agent_executor.invoke(
                    {"messages": [HumanMessage(content=user_input)]}, config
                )
                print(response)
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    websocket_endpoint()
