# create_app.py
from aiohttp import web

from .routes import fit

__all__ = [
    "create_app"
]


def create_app() -> web.Application:
    app = web.Application()
    app.add_routes([web.post('/fit', fit)])
    return app
