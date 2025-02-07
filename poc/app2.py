from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import StructuredTool, Tool
from langchain_community.llms import Ollama
from langchain.agents import AgentType
import json
from langchain_ollama import ChatOllama

# Import relevant functionality
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS for WebSocket connections
origins = [
    "http://localhost",  # Allow requests from your local frontend
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# class RestaurantSearchInput(BaseModel):
#     city: str = Field(default="New York", description="City to search in.")
#     cuisine: Optional[str] = Field(description="Cuisine type (e.g., Italian, Mexican).")
#     budget: float = Field(description="Maximum budget for two people.")
#     dish: Optional[str] = Field(
#         description="Specific dish to search for (e.g., pizza, pasta)."
#     )


# Dummy restaurant data (replace with your actual data source)
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
    {
        "restaurant_id": 2,
        "name": "Curry House on the MOON Land Place",
        "city": "New York",
        "rating": 4.2,
        "cuisines": ["Indian"],
        "budget_for_2": 25,
        "dishes": ["curry", "rice"],
    },
    {
        "restaurant_id": 3,
        "name": "Burger Joint",
        "city": "Los Angeles",
        "rating": 4.0,
        "cuisines": ["American"],
        "budget_for_2": 20,
        "dishes": ["burger", "fries"],
    },
]


@app.get("/app/v1/restaurants")
async def get_restaurants(
    city: str, cuisine: str = None, budget: float = None, dish: str = None
):
    filtered_restaurants = restaurants_data
    if city:
        filtered_restaurants = [r for r in filtered_restaurants if r["city"] == city]
    if cuisine:
        filtered_restaurants = [
            r for r in filtered_restaurants if cuisine in r["cuisines"]
        ]
    if budget:
        filtered_restaurants = [
            r for r in filtered_restaurants if r["budget_for_2"] <= budget
        ]
    if dish:
        filtered_restaurants = [r for r in filtered_restaurants if dish in r["dishes"]]
    return filtered_restaurants


def get_restaurants_tool(
    city: str, cuisine: str = None, budget: float = None, dish: str = None
) -> str:
    """
    Fetches restaurants based on criteria.

    Args:
        city (str): Name of city
        cuisine (str, optional): Cuisine type. Defaults to None.
        budget (float): Budget for two people. Defaults to None.
        dish (str, optional): Name of any specific Dish. Defaults to None.

    Returns:
        str: JSON containing the restaurants data
    """
    try:
        # Construct the URL with query parameters
        base_url = (
            "http://localhost:8000/app/v1/restaurants"  # Replace with your actual URL
        )
        params = {"city": city}
        if cuisine:
            params["cuisine"] = cuisine
        if budget:
            params["budget"] = budget
        if dish:
            params["dish"] = dish
        import requests

        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching restaurants: {e}"


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    memory = MemorySaver()

    agent_prompt = """
Your primary function is to have a casual conversation with the user about the user preference like city, cuisine, budget.
Always start the conversation with a greeting to the user.
Once you get the user city, budget, and cuisine/dish preferences you will use the provided `RestaurantSearch` tool to find restaurants based on user preferences.
You should not give any restaurants suggestions from your memory. Always use the tool to find restaurants.
If there is no response from the tool then say "I do not have any restaurants matching your criteria."

**Points to Remember:**
1.  **Tool Usage:** You should call the tool `RestaurantSearch` only when you have the user city, budget and cuisine/dish preferences data from the User.
You MUST use the `RestaurantSearch` tool to find restaurants. 
Do NOT attempt to generate restaurant names, menus, or other details on your own. 
Rely solely on the information provided by the tool.

2.  **Clarification:** If a user's request is ambiguous or lacks necessary information, ask clarifying questions. 
Do not make assumptions about what the user wants. 

    """

    llm = ChatOllama(
        base_url="http://192.168.1.3:11433",
        # model="agent_foodie_v1:latest",
        model="mistral:latest",
        temperature=0.3,
        num_gpu=1,
        num_predict=10000,
        # other params...
    )

    tools = [
        # StructuredTool.from_function(
        #     func=get_restaurants_tool,
        #     name="RestaurantSearch",
        #     description="Useful for finding restaurants.",
        #     args_schema=RestaurantSearchInput,
        #     return_direct=True,
        # )
        Tool(
            name="RestaurantSearch",
            func=get_restaurants_tool,
            description="Useful for finding restaurants based on city, cuisine, budget and dish.",
        ),
    ]

    agent_executor = create_react_agent(
        llm, tools, checkpointer=memory, state_modifier=agent_prompt
    )

    try:
        while True:
            data = await websocket.receive_text()
            try:
                user_input = json.loads(data).get("message")
                if not user_input:
                    raise ValueError(
                        "Invalid message format. Expected {'message': 'user input'}"
                    )
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in message")
            except ValueError as e:
                await websocket.send_text(str(e))
                continue

            try:
                # agent_response = agent.run(user_input)
                config = {"configurable": {"thread_id": "abc123"}}
                response = agent_executor.invoke(
                    {"messages": [HumanMessage(content=user_input)]}, config
                )
                await websocket.send_text(
                    json.dumps({"response": response["messages"][-1].content})
                )
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
