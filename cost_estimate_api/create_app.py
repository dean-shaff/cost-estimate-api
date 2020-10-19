# create_app.py
from aiohttp import web

from .routes import fit_handler_factory
from .linear_regression import compute_weights_tf

__all__ = [
    "create_app"
]


def create_app() -> web.Application:
    app = web.Application()
    app.add_routes([web.post('/fit', fit_handler_factory(compute_weights_tf))])
    return app
