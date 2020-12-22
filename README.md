# DeepLog Trainer

## Introduction

TODO

## Dev environment

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
## Drain parameters
Drain is configured using [configparser](https://docs.python.org/3.4/library/configparser.html). Config filename is `drain3.ini` in working directory.
## Run Drain
Run the following code from terminal. The arguments --input and --output are respectively the name of the file to be parsed and the name of the folder where the results will be saved 
```
python3 -m path.to.script.run.run_drain --input 'name_file' --output 'name_directory'
```
## Tests
Run tests with Pytest: from the root folder of the project run the following code:
```sh
pytest 'test_*.py'
```
Coverage:
```
coverage erase && \
coverage run --include='./deeplog_trainer/*' -m pytest && \
coverage report --include='./deeplog_trainer/*' -m && \
coverage html --include='./deeplog_trainer/*' -d './reports/coverage'

```
