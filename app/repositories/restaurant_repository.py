from typing import List, Optional
from sqlalchemy.orm import Session
from app.models import Restaurant
from app.schemas import RestaurantSchema
from app.repositories.base import BaseRepository
from app.logger import setup_logger

logger = setup_logger(__name__)


class RestaurantRepository(BaseRepository[Restaurant, RestaurantSchema]):
    def create(self, db: Session, schema: RestaurantSchema) -> Restaurant:
        logger.info(f"Creating new restaurant: {schema.name}")
        db_restaurant = Restaurant(**schema.model_dump(exclude={"id"}))
        db.add(db_restaurant)
        db.commit()
        db.refresh(db_restaurant)
        return db_restaurant

    def get_by_id(self, db: Session, id: int) -> Optional[Restaurant]:
        logger.debug(f"Fetching restaurant with id: {id}")
        return db.query(Restaurant).filter(Restaurant.id == id).first()

    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[Restaurant]:
        logger.debug(f"Fetching all restaurants with skip: {skip}, limit: {limit}")
        return db.query(Restaurant).offset(skip).limit(limit).all()

    def search(
        self,
        db: Session,
        dish_type: Optional[str] = None,
        dish_name: Optional[str] = None,
        budget: Optional[float] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Restaurant]:
        logger.info(
            f"Searching restaurants with criteria - dish_type: {dish_type}, dish_name: {dish_name}, budget: {budget}"
        )
        query = db.query(Restaurant)

        if dish_type:
            query = query.filter(Restaurant.dish_type == dish_type)
        if dish_name:
            query = query.filter(Restaurant.dishes.contains(dish_name))
        if budget:
            query = query.filter(Restaurant.budget <= budget)

        return query.offset(skip).limit(limit).all()

    def update(
        self, db: Session, id: int, schema: RestaurantSchema
    ) -> Optional[Restaurant]:
        logger.info(f"Updating restaurant with id: {id}")
        db_restaurant = self.get_by_id(db, id)
        if db_restaurant:
            for key, value in schema.model_dump(exclude={"id"}).items():
                setattr(db_restaurant, key, value)
            db.commit()
            db.refresh(db_restaurant)
        return db_restaurant

    def delete(self, db: Session, id: int) -> bool:
        logger.info(f"Deleting restaurant with id: {id}")
        db_restaurant = self.get_by_id(db, id)
        if db_restaurant:
            db.delete(db_restaurant)
            db.commit()
            return True
        return False
