import logging
import sys
from app.config import get_settings

settings = get_settings()


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(settings.log_format))
        logger.addHandler(handler)

        # Add file handler for audit logging
        file_handler = logging.FileHandler(settings.audit_file)
        file_handler.setFormatter(logging.Formatter(settings.log_format))
        logger.addHandler(file_handler)

    return logger
