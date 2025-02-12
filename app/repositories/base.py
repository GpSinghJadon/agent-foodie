from typing import Protocol, TypeVar, Generic, List, Optional
from sqlalchemy.orm import Session

T = TypeVar("T")
S = TypeVar("S")


class BaseRepository(Protocol[T, S]):
    def create(self, db: Session, schema: S) -> T: ...

    def get_by_id(self, db: Session, id: int | str) -> Optional[T]: ...

    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[T]: ...

    def update(self, db: Session, id: int | str, schema: S) -> Optional[T]: ...

    def delete(self, db: Session, id: int | str) -> bool: ...
