from sqlalchemy import Float, create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from contextlib import contextmanager

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(String, primary_key=True)
    created_at = Column(String)
    chats = relationship("Chat", back_populates="session")

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('chat_sessions.id'))
    message = Column(String)
    role = Column(String)
    timestamp = Column(String)
    session = relationship("ChatSession", back_populates="chats")

class LLM(Base):
    __tablename__ = 'llms'
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


class DatabaseManager:
    """
    The `DatabaseManager` class is responsible for managing the database connection and session for the application. It creates an SQLAlchemy engine and session factory, and provides a context manager to easily obtain a database session.
    The `__init__` method initializes the database engine and creates the necessary database tables. The `get_db` method is a context manager that returns a database session, ensuring it is properly closed after use.
    """        
    def __init__(self, db_url="sqlite:///chat_agent.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
