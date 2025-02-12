from typing import List, Optional
from sqlalchemy.orm import Session
from app.models import ChatSession
from app.schemas import ChatSessionSchema
from app.repositories.base import BaseRepository
from app.logger import setup_logger

logger = setup_logger(__name__)


class ChatSessionRepository(BaseRepository[ChatSession, ChatSessionSchema]):
    def create(self, db: Session, schema: ChatSessionSchema) -> ChatSession:
        logger.info(f"Creating new chat session: {schema.id}")
        db_session = ChatSession(**schema.model_dump())
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session

    def get_by_id(self, db: Session, id: str) -> Optional[ChatSession]:
        logger.debug(f"Fetching chat session with id: {id}")
        return db.query(ChatSession).filter(ChatSession.id == id).first()

    def get_all(
        self, db: Session, skip: int = 0, limit: int = 100
    ) -> List[ChatSession]:
        logger.debug(f"Fetching all chat sessions with skip: {skip}, limit: {limit}")
        return db.query(ChatSession).offset(skip).limit(limit).all()

    def update(
        self, db: Session, id: str, schema: ChatSessionSchema
    ) -> Optional[ChatSession]:
        logger.info(f"Updating chat session with id: {id}")
        db_session = self.get_by_id(db, id)
        if db_session:
            for key, value in schema.model_dump().items():
                setattr(db_session, key, value)
            db.commit()
            db.refresh(db_session)
        return db_session

    def delete(self, db: Session, id: str) -> bool:
        logger.info(f"Deleting chat session with id: {id}")
        db_session = self.get_by_id(db, id)
        if db_session:
            db.delete(db_session)
            db.commit()
            return True
        return False
