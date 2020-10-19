# routes.py
import typing

from aiohttp import web
import numpy as np

from .types import ArrayType


__all__ = [
    "fit_handler_factory"
]


def fit_handler_factory(compute_weights_fn: typing.Callable[[ArrayType, ArrayType], ArrayType]):
    """
    Return a handler for the `/fit` API POST request.

    Args:
        compute_weights_fn: Function that will be used by returned handler to compute linear regression weights.

    Returns:
        callable: to be used when constructing aiohttp server. 
    """

    async def fit_handler(request: web.Request) -> web.Response:
        """
        Handle `/fit` API request.
        Expects to find cost and test samples in `"data"` field of POSTed JSON data.

        Linear regression weights are in `"weights"` field of JSON response.

        Args:
            request:
        Returns:

        """
        data = await request.json()

        weights = compute_weights_fn(np.ndarray(data["data"]))

        return web.json_response({"weights": weights.tolist()})

    return fit_handler
