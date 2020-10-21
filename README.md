## Cost Estimate API

A simple, one route API for getting linear regression solutions:

```
y = a0*x0 + a1*x1 + ... + b
```

Uses `tensorflow` to train a neural network to get linear regressions weights.

### Installation

With [poetry](https://python-poetry.org/) installed

```
me@local:/path/to/cost-estimate-api$ poetry install
```

Without poetry:

```
me@local:/path/to/cost-estimate-api$ python -m pip -r requirements.txt
```

### Usage

Start API server locally:

```
me@local:/path/to/cost-estimate-api$ poetry run python -m cost_estimate_api
======== Running on http://0.0.0.0:8080 ========
```

This will run the server on port 8080.

Pass `-p`/`--port` command line argument to run on a different port:

```
me@local:/path/to/cost-estimate-api$ poetry run python -m cost_estimate_api -p 5000
======== Running on http://0.0.0.0:5000 ========
```

To disable `tensorflow` logging, set the `TF_CPP_MIN_LOG_LEVEL` environment variable:

```
me@local:/path/to/cost-estimate-api$ TF_CPP_MIN_LOG_LEVEL=3 poetry run python -m cost_estimate_api -p 5000
======== Running on http://0.0.0.0:5000 ========
```

With the server running, we can make a request (taken from `examples/request_example.py`):

```python
import asyncio

import aiohttp

# the first column is the cost, and the remaining columns correspond to Square Footage, Bedrooms, and the presence of a Swimming pool, respectively .

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
```

(This example assumes you have the server running on port 8080)


### Routes

##### POST `/fit`:

Request body:

```
{
  "data": [
    [cost_0, input_0_0, input_0_1, ....]
    [cost_1, input_1_0, input_1_1, ....]
  ]
}
```

The expected output should be in the first column of the data. The input values occupy the rest of the columns.

Response body:

```
{
  "weights": [fitted1, fitted2, ..., bias]
}
```

The number of returned values is equal to the number of input parameters plus one.
