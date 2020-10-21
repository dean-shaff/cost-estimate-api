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
        Handle request to perform linear regression.
        Expects to find cost and test samples in `"data"` field of POSTed JSON data.
        Target cost should be in the first column of the POSTed array
        with the input samples are in additional columns.

        Args:
            request: Request object whose JSON data has data for linear regression
        Returns:
            JSON Response whose `"weights"` field contains linear regression weights
        """
        json_data = await request.json()

        data = np.asarray(json_data["data"])

        kwargs = {}
        for name in ["learning_rate", "epochs"]:
            if name in json_data:
                kwargs[name] = json_data[name]

        x, y = data[:, 1:], data[:, 0]

        weights = compute_weights_fn(x, y, **kwargs)

        return web.json_response({"weights": weights.tolist()})

    return fit_handler
