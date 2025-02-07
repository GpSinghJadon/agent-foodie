from typing import Any, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import StructuredTool
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver


app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS for WebSocket connections
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy restaurant data (replace with your actual data source)
restaurants_data = [
    {
        "restaurant_id": 1,
        "name": "Pizza on the MARS Place",
        "rating": 4.5,
        "dish_type": "veg",
        "cuisines": ["Italian"],
        "budget": 30,
        "dishes": ["pizza", "pasta"],
    },
    {
        "dish_type": "veg",
        "restaurant_id": 2,
        "name": "Curry House on the MOON Land Place",
        "rating": 4.2,
        "cuisines": ["Indian"],
        "budget": 25,
        "dishes": ["curry", "rice"],
    },
    {
        "dish_type": "non-veg",
        "restaurant_id": 3,
        "name": "Burger Joint",
        "rating": 4.0,
        "cuisines": ["American"],
        "budget": 20,
        "dishes": ["burger", "fries"],
    },
]


@app.get("/app/v1/restaurants")
async def get_restaurants(
    dish_type: str, dish_name: str, budget: float
) -> list[dict[str, Any]]:
    return [
        r
        for r in restaurants_data
        if r["dish_type"] == dish_type
        and dish_name in r["dishes"]
        and r["budget"] <= budget
    ]


def restaurants_search_tool(
    budget: float,
    dish_name: str,
    dish_type: str,
) -> str:
    """
    Search Restaurants API. Search parameters: budget, dish_name, dish_type.

    Args:
        dish_name (str): Specific dish name.
        dish_type (str): Type of dish (e.g., "veg", "non-veg").
        budget (float): Budget for two people.

    Returns:
        str: JSON containing the restaurant data or an error message.

    Example Tool Call:
        get_restaurants_tool(budget=500, dish_name="Pasta", dish_type="veg")

    """
    try:
        print(f"\n\nThe tool is called: {budget}, {dish_name}, {dish_type} \n\n")
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


# Define input schema for the restaurant search
class RestaurantSearchInput(BaseModel):
    dish_name: Optional[str] = Field(None, description="Specific dish name")
    dish_type: Optional[str] = Field(None, description="veg, non-veg")
    budget: Optional[float] = Field(None, description="Budget")


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    memory = MemorySaver()
    agent_prompt = """
You are Marco, a food assistant specializing in helping users discover meals based on their preferences.
* Your primary role is to gather key information about the user's needs, including the dish name, dish type (vegetarian or non-vegetarian), budget (in rupees only), and the number of people it should serve.
* Always ask only one specific question at a time, keeping queries precise and focused.
* Once all the information is collected, you will use the integrated API to provide meal options based on the user's preferences.
* You must use the RestaurantSearch tool exclusively for searching restaurants. Never generate restaurant suggestions from your memory.
* If the tool returns no results or encounters an error, respond with: "Not able to find any matching place right now."
* Do not attempt to guess or fabricate restaurant names or details.

Example tool call after gathering user details:
    RestaurantSearch(budget=700, dish_name="Biryani", dish_type="non-veg")
"""

    llm = ChatOllama(
        base_url="http://192.168.1.55:11433",
        # model="deepseek-r1:1.5b",
        model="mistral:latest",
        # model="llama3.2:3b",
        # model="wizardlm2:latest",
        # model="qwen2.5:0.5b",
        temperature=0,
        num_predict=8196,  # Support for extended token context
    )

    tools = [
        StructuredTool.from_function(
            func=restaurants_search_tool,
            name="RestaurantSearch",
            description="Useful for finding restaurants based on budget, dish name, and dish type.",
            args_schema=RestaurantSearchInput,
        )
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
                config = {"configurable": {"thread_id": "abc123"}}
                response = agent_executor.invoke(
                    {"messages": [HumanMessage(content=user_input)]}, config
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "response": getattr(
                                response["messages"][-1],
                                "content",
                                response["messages"][-1],
                            )
                        }
                    )
                )
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
