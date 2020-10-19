# routes.py
from aiohttp import web
import numpy as np 

from .cost_estimate_linear_regression import cost_estimate_linear_regression


__all__ = [
    "fit"
]


async def fit(request: web.Request) -> web.Response:
    """
    Handle `/fit` API request.
    Expects to find cost and test samples in `"data"` field of POSTed JSON data.

    Linear regression weights are in `"weights"` field of JSON response.

    Args:
        request:
    Returns:

    """
    data = await request.json()

    weights = cost_estimate_linear_regression(np.ndarray(data["data"]))

    return web.json_response({"weights": result.tolist()})
