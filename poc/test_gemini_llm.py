import os
import requests
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
from langchain_anthropic import ChatAnthropic


# Define input schema for the restaurant search
class RestaurantSearchInput(BaseModel):
    budget: float = Field(description="Budget for two people")
    dish_name: str = Field(description="Specific dish name")
    dish_type: str = Field(description="veg, non-veg")


def restaurants_search_tool(
    budget: float,
    dish_name: str,
    dish_type: str,
) -> str:
    """
    Search Restaurants API. Search parameters: budget, dish_name, dish_type

    Args:
        budget float: Budget for two people.
        dish_name str: Specific dish name.
        dish_type str:  Type of dish (e.g., "veg", "non-veg").

    Returns:
        str: JSON containing the restaurants data.

    Example Tool Call:
        get_restaurants_tool(budget=500, dish_name="Pasta", dish_type="veg")
    """
    try:

        base_url = "http://localhost:8000/app/v1/restaurants"
        params = {}
        if budget:
            params["budget"] = budget
        if dish_name:
            params["dish_name"] = dish_name
        if dish_type:
            params["dish_type"] = dish_type

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching restaurants: {e}"


def food_agent_macro():
    memory = MemorySaver()

    agent_prompt = """
You are Marco, a food assistant specializing in helping users discover meals based on their preferences.
* Your primary role is to gather key information about the user's needs, including the dish name, dish type (vegetarian or non-vegetarian), budget (in rupees only), and the number of people it should serve.
* You must ask one question at a time, keeping queries precise and to the point.
* Once all the information is collected, you will use the integrated API to provide meal options based on the user's preferences.
* Greet users with a warm and friendly welcome message to start each interaction.
* Your tone should be approachable yet concise, ensuring the conversation remains efficient.
* The currency used for ordering will always be rupees, and no other currency context should be considered.
* To get/search the list of available restaurants use the RestaurantSearch tool.
* If tool returns no results or encounters an error, respond with:
   "Not able to find any matching place right now"
"""

    os.environ["GOOGLE_API_KEY"] = "AIzaSyDopL-Juz5vS6KAQmtNSVV2iiY2AmLgj8E"

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    tools = [
        StructuredTool.from_function(
            func=restaurants_search_tool,
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
                [print(msg) for msg in response["messages"]]
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    food_agent_macro()
