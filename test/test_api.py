from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp import web
import numpy as np

from cost_estimate_api import create_app


class TestAPI(AioHTTPTestCase):

    async def get_application(self):
        """
        Defining this method is required for using aiohttp with unittest
        """
        return create_app()

    @unittest_run_loop
    async def test_fit(self):
        data = np.random.randn(10, 4)
        resp = await self.client.request("POST", "/fit", json={'data': data.tolist()})
        self.assertTrue(resp.status == 200)
        result = await resp.json()
        self.assertTrue(len(result["weights"]) == 4)
