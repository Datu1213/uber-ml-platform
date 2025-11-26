# Common module initialization
from common.config import settings
from common.logging import setup_logging, get_logger

__version__ = "1.0.0"

# Initialize logging on import
setup_logging(
    level=settings.debug and "DEBUG" or "INFO",
    format_json=settings.is_production()
)

__all__ = ["settings", "get_logger"]
