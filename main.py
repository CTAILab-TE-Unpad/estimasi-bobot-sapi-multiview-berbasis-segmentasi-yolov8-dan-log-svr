"""Application entry point.

Run directly::

    python main.py

Or with uvicorn directly (recommended for production)::

    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Note: Use ``--workers 1`` because YOLO models are loaded into a single
ModelRegistry instance stored in ``app.state``. Multiple workers would each
load their own model copy, multiplying memory usage. For horizontal scaling,
use multiple containers behind a load balancer instead.
"""
import uvicorn

from cattle_weight.api.app import create_app
from cattle_weight.settings import get_settings

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=1,
        log_level=settings.log_level.lower(),
    )
