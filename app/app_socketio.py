import uuid
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import logging
import os
import csv
from datetime import datetime
from faker import Faker

from typing import Any, Optional

import requests
from fastapi import Depends, FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import StructuredTool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from socketio import AsyncServer, ASGIApp

from .db_manager import DatabaseManager, ChatSession, Chat, LLM, Restaurant


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
# Initialize database manager
db_manager = DatabaseManager()


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


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    llm_id: Optional[int] = None

class LLMMetadata(BaseModel):
    name: str
    base_url: str
    model_name: str

def get_db():
    with db_manager.get_db() as db:
        yield db



@app.post("/chat")
async def chat_endpoint(data: ChatMessage, db: Session = Depends(get_db)):
    if not data.session_id:
        data.session_id = str(uuid.uuid4())
        new_session = ChatSession(id=data.session_id, created_at=datetime.now().isoformat())
        db.add(new_session)
        logger.info(f"New session created: {data.session_id}")

    if data.llm_id:
        llm = db.query(LLM).filter(LLM.id == data.llm_id).first()
        if not llm:
            raise HTTPException(status_code=404, detail="LLM not found")

    response_message = f"Echo: {data.message}"
    timestamp = datetime.now().isoformat()

    user_message = Chat(session_id=data.session_id, message=data.message, role="user", timestamp=timestamp)
    agent_message = Chat(session_id=data.session_id, message=response_message, role="agent", timestamp=timestamp)
    
    db.add(user_message)
    db.add(agent_message)
    db.commit()

    return {"session_id": data.session_id, "response": response_message}

@app.get("/sessions")
async def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).all()
    return [{"session_id": session.id, "created_at": session.created_at} for session in sessions]

@app.get("/session/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    chats = db.query(Chat).filter(Chat.session_id == session_id).all()
    if not chats:
        raise HTTPException(status_code=404, detail="Session not found")
    return [{"message": chat.message, "role": chat.role, "timestamp": chat.timestamp} for chat in chats]

@app.post("/llms")
async def add_llm(llm: LLMMetadata, db: Session = Depends(get_db)):
    try:
        new_llm = LLM(name=llm.name, base_url=llm.base_url, model_name=llm.model_name)
        db.add(new_llm)
        db.commit()
    except IntegrityError:
        raise HTTPException(status_code=400, detail="LLM with this name already exists")
    return {"message": "LLM added successfully"}

@app.get("/llms")
async def list_llms(db: Session = Depends(get_db)):
    llms = db.query(LLM).all()
    return [{"id": llm.id, "name": llm.name, "base_url": llm.base_url, "model_name": llm.model_name} for llm in llms]

@app.put("/llms/{llm_id}")
async def update_llm(llm_id: int, llm: LLMMetadata, db: Session = Depends(get_db)):
    db_llm = db.query(LLM).filter(LLM.id == llm_id).first()
    if not db_llm:
        raise HTTPException(status_code=404, detail="LLM not found")
    
    db_llm.name = llm.name
    db_llm.base_url = llm.base_url
    db_llm.model_name = llm.model_name
    db.commit()
    
    return {"message": "LLM updated successfully"}


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


@app.post("/populate_restaurants")
async def populate_restaurants(db: Session = Depends(get_db)):
    faker = Faker()
    for _ in range(10):
        restaurant = Restaurant(
            name=faker.company(),
            rating=round(faker.pyfloat(min_value=1, max_value=5, right_digits=1), 1),
            dish_type=faker.random_element(elements=["veg", "non-veg"]),
            cuisines=faker.random_element(elements=['Indian','Italian','Chinese','Mexican','Thai','Japanese','French','Spanish','American','Greek','Vietnamese','Lebanese','Moroccan','Ethiopian','Brazilian','Peruvian','Korean','Turkish','German','Russian']),
            budget=faker.pyfloat(min_value=100, max_value=1000),
            
            dishes=','.join(faker.random_elements(unique=True, elements=['Butter Chicken','Chicken Tikka Masala','Samosa','Naan','Biryani','Dal Makhani','Palak Paneer','Rogan Josh','Vindaloo','Tandoori Chicken','Pizza Margherita','Spaghetti Carbonara','Lasagna','Risotto','Gnocchi','Kung Pao Chicken','Sweet and Sour Pork','Mapo Tofu','Peking Duck','Dim Sum','Tacos al Pastor','Enchiladas','Guacamole','Burritos','Chiles Rellenos','Pad Thai','Green Curry','Tom Yum Soup','Mango Sticky Rice','Massaman Curry','Sushi','Ramen','Tempura','Teriyaki Chicken','Miso Soup','Coq au Vin','Boeuf Bourguignon','Crêpes','Onion Soup','Ratatouille','Paella','Tapas','Gazpacho','Tortilla Española','Churros','Hamburger','Hot Dog','Mac and Cheese','Apple Pie','Barbecue Ribs','Souvlaki','Moussaka','Spanakopita','Baklava','Gyros','Pho','Banh Mi','Goi Cuon','Bun Cha','Ca Phe Sua Da','Hummus','Falafel','Shawarma','Tabbouleh','Baba Ghanoush','Tagine','Couscous','Pastilla','Harira','Mint Tea','Injera','Doro Wat','Kitfo','Gomen','Tibs','Feijoada','Moqueca','Coxinha','Pão de Queijo','Brigadeiro','Ceviche','Lomo Saltado','Aji de Gallina','Pachamanca','Suspiro Limeño','Kimchi','Bibimbap','Bulgogi','Tteokbokki','Japchae','Kebab','Baklava','Dolma','Manti','Pide','Schnitzel','Sausage','Sauerkraut','Black Forest Cake','Spätzle','Borscht','Pelmeni','Beef Stroganoff','Blini','Kvass'])),
        )
        db.add(restaurant)
    db.commit()
    return {"message": "Fake restaurant data populated successfully"}


from fastapi import Path, Query

@app.get("/restaurants/{restaurant_id}")
async def get_restaurant(restaurant_id: int = Path(..., description="The ID of the restaurant"), db: Session = Depends(get_db)):
    restaurant = db.query(Restaurant).filter(Restaurant.id == restaurant_id).first()
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    return restaurant

@app.get("/restaurants")
async def list_restaurants(
    skip: int = Query(default=0, description="Skip N records"),
    limit: int = Query(default=10, description="Limit N records"),
    db: Session = Depends(get_db)
):
    restaurants = db.query(Restaurant).offset(skip).limit(limit).all()
    return restaurants

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
    restaurant = Restaurant(
        name=name,
        rating=rating,
        dish_type=dish_type,
        cuisines=cuisines,
        budget=budget,
        dishes=dishes
    )
    db.add(restaurant)
    db.commit()
    db.refresh(restaurant)
    return restaurant

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
    restaurant = db.query(Restaurant).filter(Restaurant.id == restaurant_id).first()
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    
    if name: restaurant.name = name
    if rating: restaurant.rating = rating
    if dish_type: restaurant.dish_type = dish_type
    if cuisines: restaurant.cuisines = cuisines
    if budget: restaurant.budget = budget
    if dishes: restaurant.dishes = dishes
    
    db.commit()
    db.refresh(restaurant)
    return restaurant

@app.delete("/restaurants/{restaurant_id}")
async def delete_restaurant(restaurant_id: int, db: Session = Depends(get_db)):
    restaurant = db.query(Restaurant).filter(Restaurant.id == restaurant_id).first()
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    
    db.delete(restaurant)
    db.commit()
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
    session_id = data.get("session_id")
    llm_id = data.get("llm_id")
    message = data.get("message")

    if not session_id or not message:
        await sio.emit("chat_response", {"error": "Invalid message format"}, room=sid)
        return

    # Load the DB session
    db = SessionLocal()
    try:
        # Fetch chat history for the session
        chat_history = db.query(Chat).filter(Chat.session_id == session_id).all()
        history_messages = [
            {"role": chat.role, "message": chat.message} for chat in chat_history
        ]

        # Fetch the selected LLM
        llm = db.query(LLM).filter(LLM.id == llm_id).first()
        if not llm:
            await sio.emit("chat_response", {"error": "LLM not found"}, room=sid)
            return

        # Simulate LLM interaction
        response_message = f"LLM {llm.name} Response: Echo of '{message}'"
        try:
            config = {"configurable": {"thread_id": "abc123"}}
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config
            )
            response_message = getattr(
                        response["messages"][-1],
                        "content",
                        response["messages"][-1],
                    )
            await sio.emit(
                "chat_response",
                {
                    "response": response_message
                },
                room=sid,  # Send only to client's room
            )
        except Exception as e:
            await sio.emit("chat_response", {"error": str(e)}, room=sid)

        # Save the chat response
        timestamp = datetime.now()
        db.add(Chat(session_id=session_id, message=message, role="user", timestamp=timestamp))
        db.add(Chat(session_id=session_id, message=response_message, role="agent", timestamp=timestamp))
        db.commit()

        # Emit the response
        await sio.emit("chat_response", {"response": response_message}, room=sid)

    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        await sio.emit("chat_response", {"error": str(e)}, room=sid)

    finally:
        db.close()


@sio.on("simple_message")
async def handle_simple_message(sid, data):
    print(f"Message from client: {data}")
    await sio.emit("simple_response", {"response": f"Received: {data}"}, room=sid)


# Include the Socket.IO app
app.mount("/", socket_app)
