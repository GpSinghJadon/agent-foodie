from sqlalchemy import Float, Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True)
    created_at = Column(String)
    chats = relationship("Chat", back_populates="session")


class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    message = Column(String)
    role = Column(String)
    timestamp = Column(String)
    session = relationship("ChatSession", back_populates="chats")


class LLM(Base):
    __tablename__ = "llms"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True)
    base_url = Column(String)
    model_name = Column(String)


class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    rating = Column(Float)
    dish_type = Column(String)
    cuisines = Column(String)
    budget = Column(Float)
    dishes = Column(String)
