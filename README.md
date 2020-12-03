# DeepLog Trainer

## Introduction

TODO

## Development

The first step is to create a virtual environment. You can do this with PyCharm or from the terminal as follows:

```sh
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

After this you should configure your IDE to use this environment. On Pycharm it can be done on `Settings > Project > Python interpreter`.

## Docker containers


```sh
docker build --tag docker.devo.internal/dev/mlx/experiments/deeplog-trainer:latest .
```

```sh
docker run --detach \
    --network host \
    --name deeplog-trainer \
    docker.devo.internal/dev/mlx/experiments/deeplog-trainer:latest
```
