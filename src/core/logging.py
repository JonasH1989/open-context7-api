"""Application logging configuration."""

import logging
import sys
from pathlib import Path
from types import FrameType

from loguru import logger

from src.core.config import settings


class InterceptHandler(logging.Handler):
    """Handler to redirect standard Python logging to Loguru.

    Captures log messages from libraries using standard logging
    and forwards them to Loguru for consistent formatting.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to Loguru.

        Args:
            record: Log record to emit.
        """
        # Convert log level to Loguru format
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the original caller of the log message
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Set up application logging system.

    Configures Loguru with:
    - Console output (development only)
    - File logging with rotation
    - Standard library log interception
    """
    # Set up handler to capture standard Python logging
    intercept_handler = InterceptHandler()
    logging.basicConfig(handlers=[intercept_handler], level=logging.NOTSET)

    # Remove existing handlers from web server loggers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith(("uvicorn.", "gunicorn.")):
            logging.getLogger(logger_name).handlers = []

    # Connect web server loggers to our intercept handler
    logging.getLogger("uvicorn").handlers = [intercept_handler]
    logging.getLogger("uvicorn.access").handlers = [intercept_handler]
    logging.getLogger("uvicorn.error").handlers = [intercept_handler]

    # Gunicorn logger handlers
    logging.getLogger("gunicorn").handlers = [intercept_handler]
    logging.getLogger("gunicorn.access").handlers = [intercept_handler]
    logging.getLogger("gunicorn.error").handlers = [intercept_handler]

    # Remove default Loguru handler to avoid duplicates
    logger.remove()

    # Create log directory if it doesn't exist
    log_dir = Path(settings.LOG_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler (only in development for visibility)
    if not settings.is_production:
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            level=settings.LOG_LEVEL,
            colorize=True,
            diagnose=False,
        )

    # File handler for application logs with rotation
    logger.add(
        log_dir / settings.LOG_FILENAME,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="gz",
        enqueue=True,  # Non-blocking logging
    )

    logger.info(
        f"Logging configured: environment={settings.ENVIRONMENT}, "
        f"level={settings.LOG_LEVEL}, path={log_dir}"
    )
