"""FastAPI dependency providers.

Using ``Depends`` injection instead of importing ``app.state`` directly keeps
route handlers decoupled from the application lifecycle and makes them easy to
test by overriding dependencies via ``app.dependency_overrides``.
"""
from __future__ import annotations

from fastapi import Request

from cattle_weight.infrastructure.model_registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    """Retrieve the :class:`ModelRegistry` stored in ``app.state`` at startup.

    This dependency is injected into route handlers that need access to the
    loaded ML models.

    Args:
        request: The current FastAPI ``Request`` object.

    Returns:
        The application-wide :class:`ModelRegistry` instance.
    """
    return request.app.state.registry
