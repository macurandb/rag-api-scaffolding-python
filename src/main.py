"""Main application entry point."""

from .adapters.input.api import create_app
from .adapters.output.logger import configure_logging
from .config import settings

# Configure logging
configure_logging(settings.log_level)

# Create application
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
