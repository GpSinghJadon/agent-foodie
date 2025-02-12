from pydantic import BaseModel, ConfigDict
from typing import Optional, List


class ChatMessageSchema(BaseModel):
    message: str
    session_id: Optional[str] = None
    llm_id: Optional[int] = None


class LLMSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    name: str
    base_url: str
    model_name: str


class ChatSessionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: str


class ChatSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    session_id: str
    message: str
    role: str
    timestamp: str


class RestaurantSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    name: str
    rating: float
    dish_type: str
    cuisines: str
    budget: float
    dishes: str
