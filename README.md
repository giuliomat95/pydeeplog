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
pytest 
```
Coverage:
```
coverage erase && \
coverage run --include='./deeplog_trainer/*' -m pytest && \
coverage report --include='./deeplog_trainer/*' -m && \
coverage html --include='./deeplog_trainer/*' -d './reports/coverage'

```
## Run Model
To run the `run_model.py` file, set the following parameters in the command line:
+ filepath: file path of the json dataset to parse.
+ window_size: length of chunks, input of the LSTM neural network. Default value set to 10.
+ min_length: the minimum length of a sequence to be parsed. Default value set to 4.
+ output: filepath of the output model file.
+ LSTM_units: number of units in each LSTM layer. Default value set to 64.
+ train_ratio: it defines the train set. Default value set to 0.7.
+ val_ratio: it defines the validation set. Default value set to 0.85.
+ batch_size: number of samples that will be propagated through the network. Default value set to 512.
+ early_stop: number of epochs with no improvement after which training will be stopped. Default value set to 7.

Execute the command `python3 -m run.run_model.py -h` to display of arguments.
Example of execution:
```
python3 -m path.to.script.run.run_model.py --filepath run/batrasio_result/data.json --output model_result/model --window_size 12 
--n_epochs 100 --train_ratio 0.5 --val_ratio 0.75
```
## Tensorboard
To visualize the evolution of the loss/accuracy trend of the train/validation process, run the following code from the root folder:
```
tensorboard --logdir logdir
```
