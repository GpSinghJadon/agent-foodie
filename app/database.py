from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from app.config import get_settings
from app.logger import setup_logger

logger = setup_logger(__name__)


class Database:
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(
            self.settings.database_url,
            pool_pre_ping=True,  # Enable connection health checks
            pool_size=5,  # Set connection pool size
            max_overflow=10,  # Maximum number of connections to allow above pool_size
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    @contextmanager
    def get_db(self):
        db = self.SessionLocal()
        try:
            logger.debug("Database session started")
            yield db
        except Exception as e:
            logger.error(f"Database session error: {e}")
            db.rollback()
            raise
        finally:
            logger.debug("Database session closed")
            db.close()


# Create database instance
db = Database()
