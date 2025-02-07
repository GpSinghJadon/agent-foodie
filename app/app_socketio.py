import logging
import os
import csv
from datetime import datetime

from typing import Any, Optional

import requests
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import StructuredTool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from socketio import AsyncServer, ASGIApp


# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration via .env and pydantic_settings
class Settings(BaseSettings):
    ollama_base_url: str
    ollama_model: str
    restaurants_api_url: str
    audit_file: str = "audit.csv"
    cors_origins: list[str] = ["*"]


config = Settings()


# Initialize Audit Log
def init_audit_log():
    if not os.path.exists(config.audit_file):
        with open(config.audit_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Event", "Details"])


init_audit_log()


def audit_event(event: str, details: str):
    timestamp = datetime.now().isoformat()
    with open(config.audit_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, event, details])


app = FastAPI(
    title="Food Agent API",
    description="A food agent powered by Langchain and Ollama",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,  # Loaded from .env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO
sio = AsyncServer(async_mode="asgi", cors_allowed_origins=config.cors_origins)
socket_app = ASGIApp(sio, socketio_path="socket.io")


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

        params = {}
        if budget:
            params["budget"] = budget
        if dish_name:
            params["dish_name"] = dish_name
        if dish_type:
            params["dish_type"] = dish_type

        response = requests.get(
            config.restaurants_api_url, params=params, timeout=10
        )  # Timeout added
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        audit_event("Restaurant Search", f"Successful search: {params}")
        return response.text

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching restaurants: {e}")
        audit_event("Restaurant Search", f"Error: {e}")
        return f"Error fetching restaurants: {e}"


# Define input schema for the restaurant search
class RestaurantSearchInput(BaseModel):
    dish_name: Optional[str] = Field(None, description="Specific dish name")
    dish_type: Optional[str] = Field(None, description="veg, non-veg")
    budget: Optional[float] = Field(None, description="Budget")


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

# Initialize Socket.IO
sio = AsyncServer(async_mode="asgi", cors_allowed_origins=[])

# Mount Socket.IO as an ASGI app
socket_app = ASGIApp(sio, socketio_path="socket.io")


@sio.event
async def connect(sid, environ):
    audit_event("Client Connect", f"Client {sid} connected.")
    memory = MemorySaver()  # Initialize memory per connection

    llm = ChatOllama(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        temperature=0,
        num_predict=8196,
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
    await sio.enter_room(sid, sid)  # Create a room for each client
    await sio.save_session(sid, {"agent_executor": agent_executor, "memory": memory})


@sio.event
async def disconnect(sid):
    audit_event("Client Disconnect", f"Client {sid} disconnected.")
    await sio.leave_room(sid, sid)
    await sio.disconnect(sid)


@sio.on("chat_message")
async def handle_chat_message(sid, data):
    session = await sio.get_session(sid)
    agent_executor = session["agent_executor"]

    try:
        print(f"event recieved")
        user_input = data.get("message")
        if not user_input:
            raise ValueError(
                "Invalid message format. Expected {'message': 'user input'}"
            )

        config = {"configurable": {"thread_id": "abc123"}}
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config
        )

        await sio.emit(
            "chat_response",
            {
                "response": getattr(
                    response["messages"][-1],
                    "content",
                    response["messages"][-1],
                )
            },
            room=sid,  # Send only to client's room
        )
    except Exception as e:
        await sio.emit("chat_response", {"error": str(e)}, room=sid)


@sio.on("simple_message")
async def handle_simple_message(sid, data):
    print(f"Message from client: {data}")
    await sio.emit("simple_response", {"response": f"Received: {data}"}, room=sid)


# Include the Socket.IO app
app.mount("/", socket_app)
