import asyncio

import aiohttp


request_data = {
    "data": [
        [100000, 1000, 3, False],
        [200000, 1500, 4, False],
        [300000, 1500, 4.5, True],
        [257000, 1200, 3, False]
    ]
}


async def main():
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8080/fit"
        async with session.post(url, json=request_data) as resp:
            weights = (await resp.json())["weights"]
            print(weights)


if __name__ == "__main__":
    asyncio.run(main())
