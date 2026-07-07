"""Cattle Weight Estimation package.

This package provides a production-grade FastAPI microservice for estimating
cattle live weight from side and back images using YOLOv8 segmentation and
a Log-space SVR regression model.

Submodules
----------
- ``cattle_weight.core``       : ML pipeline (segmentation, morphometry, prediction)
- ``cattle_weight.api``        : FastAPI application and HTTP layer
- ``cattle_weight.infrastructure`` : Model registry and logging setup
- ``cattle_weight.settings``   : Runtime configuration (environment-based)
- ``cattle_weight.config``     : Domain constants (algorithm-level, not deployment)
- ``cattle_weight.exceptions`` : Custom domain exception hierarchy
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
