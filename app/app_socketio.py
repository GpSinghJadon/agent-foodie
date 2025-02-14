from fastapi import Path, Query
import json
import uuid
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import logging
import os
import csv
from datetime import datetime
from faker import Faker

from typing import Any, List, Optional
from app.schemas import (
    ChatMessageSchema,
    LLMSchema,
    ChatSessionSchema,
    ChatSchema,
    RestaurantSchema,
)

import requests
from fastapi import Depends, FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import StructuredTool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings

settings = get_settings()

from socketio import AsyncServer, ASGIApp

from app.db_manager import DatabaseManager, ChatSession, Chat, LLM, Restaurant


from app.logger import setup_logger

logger = setup_logger(__name__)


# Initialize Audit Log
def init_audit_log():
    if not os.path.exists(settings.audit_file):
        with open(settings.audit_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Event", "Details"])


init_audit_log()
# Initialize database manager
db_manager = DatabaseManager()


def audit_event(event: str, details: str):
    timestamp = datetime.now().isoformat()
    with open(settings.audit_file, "a", newline="") as csvfile:
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
    allow_origins=settings.cors_origins,  # Loaded from .env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO
sio = AsyncServer(async_mode="asgi", cors_allowed_origins=settings.cors_origins)
socket_app = ASGIApp(sio, socketio_path="socket.io")


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    llm_id: Optional[int] = None

class LLMMetadata(BaseModel):
    name: str
    base_url: str
    model_name: str


class RestaurantPydantic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    rating: float
    dish_type: str
    cuisines: str
    budget: float
    dishes: str


from app.database import db

def get_db():
    with db.get_db() as session:
        yield session

from app.repositories.chat_repository import ChatRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.llm_repository import LLMRepository
from app.repositories.restaurant_repository import RestaurantRepository

chat_repo = ChatRepository()
session_repo = ChatSessionRepository()
llm_repo = LLMRepository()
restaurant_repo = RestaurantRepository()

@app.post("/chat")
async def chat_endpoint(data: ChatMessage, db: Session = Depends(get_db)):
    if not data.session_id:
        data.session_id = str(uuid.uuid4())
        session_schema = ChatSessionSchema(
            id=data.session_id, created_at=datetime.now().isoformat()
        )
        session_repo.create(db, session_schema)
        logger.info(f"New session created: {data.session_id}")

    if data.llm_id:
        llm = llm_repo.get_by_id(db, data.llm_id)
        if not llm:
            raise HTTPException(status_code=404, detail="LLM not found")

    response_message = f"Echo: {data.message}"
    timestamp = datetime.now().isoformat()

    user_chat_schema = ChatSchema(
        session_id=data.session_id,
        message=data.message,
        role="user",
        timestamp=timestamp,
    )
    agent_chat_schema = ChatSchema(
        session_id=data.session_id,
        message=response_message,
        role="agent",
        timestamp=timestamp,
    )

    chat_repo.create(db, user_chat_schema)
    chat_repo.create(db, agent_chat_schema)

    return {"session_id": data.session_id, "response": response_message}


@app.get("/sessions")
async def list_sessions(db: Session = Depends(get_db)):
    return session_repo.get_all(db)

@app.get("/session/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    chats = chat_repo.get_by_session_id(db, session_id)
    if not chats:
        raise HTTPException(status_code=404, detail="Session not found")
    return [ChatSchema.model_validate(chat) for chat in chats]


@app.post("/llms")
async def add_llm(llm: LLMMetadata, db: Session = Depends(get_db)):
    try:
        llm_schema = LLMSchema(
            name=llm.name, base_url=llm.base_url, model_name=llm.model_name
        )
        llm_repo.create(db, llm_schema)
    except IntegrityError:
        raise HTTPException(status_code=400, detail="LLM with this name already exists")
    return {"message": "LLM added successfully"}

@app.get("/llms")
async def list_llms(db: Session = Depends(get_db)):
    llms = llm_repo.get_all(db)
    return [{"id": llm.id, "name": llm.name, "base_url": llm.base_url, "model_name": llm.model_name} for llm in llms]


@app.put("/llms/{llm_id}")
async def update_llm(llm_id: int, llm: LLMMetadata, db: Session = Depends(get_db)):
    llm_schema = LLMSchema(
        name=llm.name, base_url=llm.base_url, model_name=llm.model_name
    )
    updated_llm = llm_repo.update(db, llm_id, llm_schema)
    if not updated_llm:
        raise HTTPException(status_code=404, detail="LLM not found")
    return {"message": "LLM updated successfully"}


@app.put("/llms/{llm_id}")
async def update_llm(llm_id: int, llm: LLMMetadata, db: Session = Depends(get_db)):
    llm_schema = LLMSchema(
        name=llm.name, base_url=llm.base_url, model_name=llm.model_name
    )
    updated_llm = llm_repo.update(db, llm_id, llm_schema)
    if not updated_llm:
        raise HTTPException(status_code=404, detail="LLM not found")
    return {"message": "LLM updated successfully"}

    return {"message": "LLM updated successfully"}


@app.get("/app/v1/restaurants", response_model=dict[str, List[RestaurantSchema]])
async def get_restaurants(
    dish_type: str = Query(default="veg", description="veg/non-veg"),
    dish_name: str = Query(
        default="Pizza", description="Dish name which has to served in restaurant"
    ),
    budget: float = Query(default=10, description="Total budget of the meal"),
    skip: int = Query(default=0, description="Skip N records"),
    limit: int = Query(default=10, description="Limit N records"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    restaurants = restaurant_repo.search(db, dish_type, dish_name, budget, skip, limit)
    return {"data": restaurants}


@app.post("/populate_fake_restaurants")
async def populate_restaurants(db: Session = Depends(get_db)):
    faker = Faker()
    restaurants: list[RestaurantSchema] = []
    for _ in range(10):
        restaurants.append(
            RestaurantSchema(
                name=faker.company(),
                rating=round(
                    faker.pyfloat(min_value=1, max_value=5, right_digits=1), 1
                ),
                dish_type=faker.random_element(elements=["veg", "non-veg"]),
                cuisines=faker.random_element(
                    elements=[
                        "Indian",
                        "Italian",
                        "Chinese",
                        "Mexican",
                        "Thai",
                        "Japanese",
                        "French",
                        "Spanish",
                        "American",
                        "Greek",
                        "Vietnamese",
                        "Lebanese",
                        "Moroccan",
                        "Ethiopian",
                        "Brazilian",
                        "Peruvian",
                        "Korean",
                        "Turkish",
                        "German",
                        "Russian",
                    ]
                ),
                budget=faker.pyfloat(min_value=100, max_value=1000),
                dishes=",".join(
                    faker.random_elements(
                        unique=True,
                        elements=[
                            "Butter Chicken",
                            "Chicken Tikka Masala",
                            "Samosa",
                            "Naan",
                            "Biryani",
                            "Dal Makhani",
                            "Palak Paneer",
                            "Rogan Josh",
                            "Vindaloo",
                            "Tandoori Chicken",
                            "Pizza Margherita",
                            "Spaghetti Carbonara",
                            "Lasagna",
                            "Risotto",
                            "Gnocchi",
                            "Kung Pao Chicken",
                            "Sweet and Sour Pork",
                            "Mapo Tofu",
                            "Peking Duck",
                            "Dim Sum",
                            "Tacos al Pastor",
                            "Enchiladas",
                            "Guacamole",
                            "Burritos",
                            "Chiles Rellenos",
                            "Pad Thai",
                            "Green Curry",
                            "Tom Yum Soup",
                            "Mango Sticky Rice",
                            "Massaman Curry",
                            "Sushi",
                            "Ramen",
                            "Tempura",
                            "Teriyaki Chicken",
                            "Miso Soup",
                            "Coq au Vin",
                            "Boeuf Bourguignon",
                            "Crêpes",
                            "Onion Soup",
                            "Ratatouille",
                            "Paella",
                            "Tapas",
                            "Gazpacho",
                            "Tortilla Española",
                            "Churros",
                            "Hamburger",
                            "Hot Dog",
                            "Mac and Cheese",
                            "Apple Pie",
                            "Barbecue Ribs",
                            "Souvlaki",
                            "Moussaka",
                            "Spanakopita",
                            "Baklava",
                            "Gyros",
                            "Pho",
                            "Banh Mi",
                            "Goi Cuon",
                            "Bun Cha",
                            "Ca Phe Sua Da",
                            "Hummus",
                            "Falafel",
                            "Shawarma",
                            "Tabbouleh",
                            "Baba Ghanoush",
                            "Tagine",
                            "Couscous",
                            "Pastilla",
                            "Harira",
                            "Mint Tea",
                            "Injera",
                            "Doro Wat",
                            "Kitfo",
                            "Gomen",
                            "Tibs",
                            "Feijoada",
                            "Moqueca",
                            "Coxinha",
                            "Pão de Queijo",
                            "Brigadeiro",
                            "Ceviche",
                            "Lomo Saltado",
                            "Aji de Gallina",
                            "Pachamanca",
                            "Suspiro Limeño",
                            "Kimchi",
                            "Bibimbap",
                            "Bulgogi",
                            "Tteokbokki",
                            "Japchae",
                            "Kebab",
                            "Baklava",
                            "Dolma",
                            "Manti",
                            "Pide",
                            "Schnitzel",
                            "Sausage",
                            "Sauerkraut",
                            "Black Forest Cake",
                            "Spätzle",
                            "Borscht",
                            "Pelmeni",
                            "Beef Stroganoff",
                            "Blini",
                            "Kvass",
                        ],
                    )
                ),
            )
        )
    restaurant_repo.create_batch(db, restaurants)
    return {"message": "Fake restaurant data populated successfully"}


@app.post("/load_restaurants")
async def load_restaurants(db: Session = Depends(get_db)):
    restaurants: list[RestaurantSchema] = []
    for row in json.load(open("restaurants.json", "r")):
        restaurants.append(RestaurantSchema(**row))
    restaurant_repo.create_batch(db, restaurants)
    return {
        "message": f"{len(restaurants)} Logical restaurants data populated successfully"
    }


@app.get("/restaurants/{restaurant_id}")
async def get_restaurant(restaurant_id: int = Path(..., description="The ID of the restaurant"), db: Session = Depends(get_db)):
    restaurant = restaurant_repo.get_by_id(db, restaurant_id)
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    return restaurant

@app.get("/restaurants")
async def list_restaurants(
    skip: int = Query(default=0, description="Skip N records"),
    limit: int = Query(default=10, description="Limit N records"),
    db: Session = Depends(get_db)
):
    return restaurant_repo.get_all(db, skip=skip, limit=limit)


@app.post("/restaurants")
async def create_restaurant(
    name: str,
    rating: float,
    dish_type: str,
    cuisines: str,
    budget: float,
    dishes: str,
    db: Session = Depends(get_db)
):
    restaurant_schema = RestaurantSchema(
        name=name,
        rating=rating,
        dish_type=dish_type,
        cuisines=cuisines,
        budget=budget,
        dishes=dishes,
    )
    return restaurant_repo.create(db, restaurant_schema)

@app.put("/restaurants/{restaurant_id}")
async def update_restaurant(
    restaurant_id: int,
    name: str = None,
    rating: float = None,
    dish_type: str = None,
    cuisines: str = None,
    budget: float = None,
    dishes: str = None,
    db: Session = Depends(get_db)
):
    restaurant_schema = RestaurantSchema(
        name=name,
        rating=rating,
        dish_type=dish_type,
        cuisines=cuisines,
        budget=budget,
        dishes=dishes,
    )
    updated_restaurant = restaurant_repo.update(db, restaurant_id, restaurant_schema)
    if not updated_restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    return updated_restaurant

@app.delete("/restaurants/{restaurant_id}")
async def delete_restaurant(restaurant_id: int, db: Session = Depends(get_db)):
    if not restaurant_repo.delete(db, restaurant_id):
        raise HTTPException(status_code=404, detail="Restaurant not found")
    return {"message": f"Restaurant {restaurant_id} deleted successfully"}

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
            settings.restaurants_api_url, params=params, timeout=10
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
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
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
    session_id: str = data.get("session_id")
    llm_id = data.get("llm_id")
    message = data.get("message")
    # socket_session = await sio.get_session(sid)
    # agent_executor = socket_session.get("agent_executor")
    # memory = socket_session.get("memory")

    if not session_id or not message:
        await sio.emit("chat_response", {"error": "Invalid message format"}, room=sid)
        return

    # Load the DB session
    db = Session()
    try:
        with db_manager.get_db() as db:  # Use the context manager here

            # Check if session exists, create if not
            session = session_repo.get_by_id(db, session_id)
            if not session:
                session = ChatSessionSchema(
                    id=session_id, created_at=datetime.now().isoformat()
                )
                session = session_repo.create(db, session)
                logger.info(f"New session created: {session.id}")

            # Load the selected LLM from the database
            llm_data = db.query(LLM).filter(LLM.id == llm_id).first()
            llm_data = llm_repo.get_by_id(db, llm_id)
            if not llm_data:
                await sio.emit("chat_response", {"error": "LLM not found"}, room=sid)
                return

            # Create the LLM instance with loaded config
            llm = ChatOllama(
                base_url=llm_data.base_url,
                model=llm_data.model_name,
                temperature=0,  # Adjust as needed
                num_predict=8196,  # Adjust as needed
            )

            # Fetch chat history for the session
            chat_history = chat_repo.get_by_session_id(db, session_id)
            history_messages = []
            for chat in chat_history:
                if chat.role == "user":
                    history_messages.append(HumanMessage(content=chat.message))
                else:
                    history_messages.append(AIMessage(content=chat.message))

            # Recreate the agent executor with the specific LLM and memory for the session
            memory = (
                MemorySaver()
            )  # this will create new memory for every chat request. In a real application scenario, this memory should be loaded from some storage (like Redis). In the same way when the conversation ends, this memory should be saved to some place, in this case I'm just throwing it away.
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

            # Invoke the agent with the chat history
            config = {
                "configurable": {"thread_id": "abc123"}
            }  # this needs to be improved.
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=message)] + history_messages}, config
            )  # Append history messages

            response_message = getattr(
                response["messages"][-1],
                "content",
                response["messages"][-1],
            )
            # Save the chat response
            timestamp = datetime.now()
            user_message = ChatSchema(
                session_id=session_id,
                message=message,
                role="user",
                timestamp=timestamp.isoformat(),
            )
            user_message = chat_repo.create(db, user_message)

            agent_message = ChatSchema(
                session_id=session_id,
                message=response_message,
                role="agent",
                timestamp=timestamp.isoformat(),
            )
            db.add(agent_message)
            agent_message = chat_repo.create(db, agent_message)

        # Emit the response
        await sio.emit(
            "chat_response",
            {
                "response": response_message,
                "session_id": session_id,
                "user_message_id": user_message.id,
                "agent_message_id": agent_message.id,
                "model_id": llm_id,
            },
            room=sid,  # Send only to client's room
        )

    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        await sio.emit("chat_response", {"error": str(e)}, room=sid)


@sio.on("simple_message")
async def handle_simple_message(sid, data):
    print(f"Message from client: {data}")
    await sio.emit("simple_response", {"response": f"Received: {data}"}, room=sid)


# Include the Socket.IO app
app.mount("/", socket_app)
