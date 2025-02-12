from typing import List, Optional
from sqlalchemy.orm import Session
from app.models import Chat
from app.schemas import ChatSchema
from app.repositories.base import BaseRepository
from app.logger import setup_logger

logger = setup_logger(__name__)


class ChatRepository(BaseRepository[Chat, ChatSchema]):
    def create(self, db: Session, schema: ChatSchema) -> Chat:
        logger.info(f"Creating new chat message for session: {schema.session_id}")
        db_chat = Chat(**schema.model_dump(exclude={"id"}))
        db.add(db_chat)
        db.commit()
        db.refresh(db_chat)
        return db_chat

    def get_by_id(self, db: Session, id: int) -> Optional[Chat]:
        logger.debug(f"Fetching chat message with id: {id}")
        return db.query(Chat).filter(Chat.id == id).first()

    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[Chat]:
        logger.debug(f"Fetching all chat messages with skip: {skip}, limit: {limit}")
        return db.query(Chat).offset(skip).limit(limit).all()

    def get_by_session_id(self, db: Session, session_id: str) -> List[Chat]:
        logger.debug(f"Fetching all chat messages for session: {session_id}")
        return db.query(Chat).filter(Chat.session_id == session_id).all()

    def update(self, db: Session, id: int, schema: ChatSchema) -> Optional[Chat]:
        logger.info(f"Updating chat message with id: {id}")
        db_chat = self.get_by_id(db, id)
        if db_chat:
            for key, value in schema.model_dump(exclude={"id"}).items():
                setattr(db_chat, key, value)
            db.commit()
            db.refresh(db_chat)
        return db_chat

    def delete(self, db: Session, id: int) -> bool:
        logger.info(f"Deleting chat message with id: {id}")
        db_chat = self.get_by_id(db, id)
        if db_chat:
            db.delete(db_chat)
            db.commit()
            return True
        return False
