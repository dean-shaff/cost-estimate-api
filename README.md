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

With the server running, we can make a request to fit some data (taken from `examples/request_example.py`):

Here we'll be training on some data pertaining to house values. The first value in each row is a house's market price and the remaining values correspond to Square Footage, Bedrooms, and the presence of a Swimming pool, respectively. In other words, the first column of our training data is the value we're attempting to predict, and the remaining columns are input parameters to the linear regression model.

```python
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
```

(This example assumes you have the server running on port 8080)

Sometimes the computed weights may not be satisfactory, so we can modify some of the training parameters:

```python
request_data = {
    "data": [
        [100000, 1000, 3, False],
        [200000, 1500, 4, False],
        [300000, 1500, 4.5, True],
        [257000, 1200, 3, False]
    ],
    "learning_rate": 0.0001,
    "epochs": 500
}
```

### Testing

```
me@local:/path/to/cost-estimate-api$ poetry run python -m unittest
```


### Routes

##### POST `/fit`:

Request body:

```
{
  "data": [
    [cost_0, input_0_0, input_0_1, ....]
    [cost_1, input_1_0, input_1_1, ....]
  ],
  "learning_rate": 0.001,
  "epochs": 400
}
```

The `learning_rate` and `epochs` fields are optional.

Each row in the `data` field is a single training sample. The first value in each row is the expected output value. The remaining values correspond to input parameters.

Response body:

```
{
  "weights": [weight_0, weight_1, ..., bias]
}
```

The number of returned values is equal to the number of input parameters plus one.
