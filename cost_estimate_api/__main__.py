import argparse
import logging

from aiohttp import web

from .create_app import create_app


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cost Estimate API")
    parser.add_argument("-p", "--port", dest="port", type=int, default=8080,
                        help="Specify port on which to run API (default %(default)s)")

    return parser


if __name__ == "__main__":
    parsed = create_parser().parse_args()

    app = create_app()
    web.run_app(app, port=parsed.port)
