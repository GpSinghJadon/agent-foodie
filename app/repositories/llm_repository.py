from typing import List, Optional
from sqlalchemy.orm import Session
from app.models import LLM
from app.schemas import LLMSchema
from app.repositories.base import BaseRepository
from app.logger import setup_logger

logger = setup_logger(__name__)


class LLMRepository(BaseRepository[LLM, LLMSchema]):
    def create(self, db: Session, schema: LLMSchema) -> LLM:
        logger.info(f"Creating new LLM: {schema.name}")
        db_llm = LLM(**schema.model_dump(exclude={"id"}))
        db.add(db_llm)
        db.commit()
        db.refresh(db_llm)
        return db_llm

    def get_by_id(self, db: Session, id: int) -> Optional[LLM]:
        logger.debug(f"Fetching LLM with id: {id}")
        return db.query(LLM).filter(LLM.id == id).first()

    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[LLM]:
        logger.debug(f"Fetching all LLMs with skip: {skip}, limit: {limit}")
        return db.query(LLM).offset(skip).limit(limit).all()

    def update(self, db: Session, id: int, schema: LLMSchema) -> Optional[LLM]:
        logger.info(f"Updating LLM with id: {id}")
        db_llm = self.get_by_id(db, id)
        if db_llm:
            for key, value in schema.model_dump(exclude={"id"}).items():
                setattr(db_llm, key, value)
            db.commit()
            db.refresh(db_llm)
        return db_llm

    def delete(self, db: Session, id: int) -> bool:
        logger.info(f"Deleting LLM with id: {id}")
        db_llm = self.get_by_id(db, id)
        if db_llm:
            db.delete(db_llm)
            db.commit()
            return True
        return False
