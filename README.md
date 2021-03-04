# DeepLog Trainer

## Introduction

In this project, we provide a Deep Learning model, called **DeepLog**, that aims to detect anomalies occurred in a 
system logs. In particular, we propose an LSTM (Long Short Term Memory)-based model, a Recurrent Neural Network (RNN) 
capable of memorizing long-term dependencies over time.\
The architecture of Deeplog is composed by three main components: the `log key anomaly detection model`, the `parameter 
value anomaly detection model`, and the `workflow model` to diagnose detected anomalies.   
For more information read the following paper: https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf.

## Log Parser stage: Drain

As a first step, the free-text log entries are parsed into a structured representation, what we call the 
`parser stage`. The goal of this operation is to extract the constant message, called `log key` that recurs few times 
and store the variable part, known as the `parameter`, for each entry log. \
The state-of-the-art log parsing method is represented by Drain, a Depth-Tree based online log parsing method. 
For further details: https://pinjiahe.github.io/papers/ICWS17.pdf. \
Once that logs are parsed, each of them is encoded with the respective log key id and grouped in different "sessions". 
In particular, a new group is created every time the content of the log message is "TCP source connection created", 
and it contains all the logs until a new message with the same starting delimiter come in. 

## Log key anomaly detection model

Once the numerical representation of each session is ready, DeepLog treats these sequences as a Multi-class Time Series
Classification,  where each distinct log key defines a class. (`log key anomaly detection model`).\
At every time steps, given an  history of recent log keys in input, the LSTM model outputs a probability distribution 
over the n log key classes. Lastly, fixed the integer parameter `g`, DeepLog detects the testing key as an anomaly if 
it's not among the "g" keys with the greatest probability value of being the next key (the top g candidates), and as
normal otherwise. Consequently, the parameter g works likewise a threshold for the number of anomalies detected.

## Workflows

At a later stage, the log key sequences are used to create a workflows model as a deterministic finite state machine 
(https://en.wikipedia.org/wiki/Finite-state_machine). In particular, we try to detect the divergence points caused by 
*concurrency*, i.e. when when several operations are present in the same task, or the paths that can be reduced to a 
loop.\
In this way, the workflow model is very useful towards enabling users to diagnose what had gone wrong in the execution 
of a task when an anomaly has been detected.

## Parameter value anomaly detection model

The last part of Deeplog's architecture, as said before, is the `parameter value anomaly detection model`. In this stage 
Deeplog analyses the variable part of the message, what we have called the `paramaters`, to detect the anomalies that 
have not been shown as a deviation from a normal execution path in the first part, but as an irregular parameter value.\
For each key, the parameters are stored along with the elapsed time between one incoming log and another, in order to 
create a multivariate time series. Then, an LSTM forecasting model is used to detect the anomalies. \
With a similar approach to the first stage, the LSTM model takes in input, at each time step, an history sequence of 
value vectors, and try to predict the next one. To evaluate the model, the data is split in three sets: train, 
validation and test. For each vector in the validation set, we calculate the MSE between him and its prediction 
and finally modeled as Gaussian distribution. \
At deployment, if the error between a prediction and an observed value vector is within a high-level of confidence 
interval of the above Gaussian distribution, the parameter value vector of the incoming log entry is considered normal,
and is abnormal otherwise. 

In this tutorial we will evaluate DeepLog on Batrasio, a real time data set provided by the Devo platform. 
In Batrasio, every message containing the text "*TCP source SSL error*" or "*TCP source socket error*" is labeled as 
abnormal.

## Dev environment

The first step is to create a virtual environment. You can do this with PyCharm or from the terminal as follows:

```sh
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

After this you should configure your IDE to use this environment.
On Pycharm it can be done on `Settings > Project > Python interpreter`.

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

Drain is configured using [configparser](https://docs.python.org/3.4/library/configparser.html).
Config filename is `drain3.ini` in working directory.

## Run Drain

Run the following code from terminal. The arguments --input and --output are respectively the filepath of the data to 
be parsed and the name of the folder where the results will be saved 
```sh
python3 -m path.to.script.run.run_drain --input_file data/sample_batrasio.log --output_path batrasio_result
```

## Run Log key anomaly detection Model

To run the `run_model.py` file, set the following parameters in the command line:
+ `input_file`: path of the input json dataset to parse.
+ `window_size`: length of chunks, input of the LSTM neural network. Default value set to 10.
+ `min_length`: the minimum length of a sequence to be parsed. Default value set to 4.
+ `output_path`: path of the directory where to save the trained model, as well as the config values.
+ `output_file`: name of the output model file.
+ `LSTM_units`: number of units in each LSTM layer. Default value set to 64.
+ `train_ratio`: it defines the train set. Default value set to 0.7.
+ `val_ratio`: it defines the validation set. Default value set to 0.85.
+ `batch_size`: number of samples that will be propagated through the network. Default value set to 512.
+ `early_stop`: number of epochs with no improvement after which training will be stopped. Default value set to 7.
+ `max_epochs`: maximum number of epochs if the process is not stopped before by the early_stop. Default value set to 50.
+ `out_tensorboard_path`: name of the folder where to save the tensorboard results. If empty any board is stored. 
  Default value set to `None`.

The parameters without default values are mandatory to run the file.  
Execute the command `python3 -m run.run_model.py -h` to display the arguments.
Example of execution:
```sh
python3 -m path.to.script.run.run_model.py --input_file run/data/data.json --output_path model_result  
--output_file model.h5 --window_size 12 --max_epochs 100 --train_ratio 0.5 --val_ratio 0.75 --out_tensorboard_path logdir
```

## Run parameter value anomaly detection model

To run the `run_parameter_detection.py` file, set the following parameters in the command line:
+ `input_file`: path of the input dataset to parse, with all the parameters of a specific log key message.
+ `window_size`: length of chunks, input of the LSTM neural network. Default value set to 5.
+ `LSTM_units`: number of units in each LSTM layer. Default value set to 64.
+ `max_epochs`: maximum number of epochs if the process is not stopped before by the early_stop. Default value set to 100.
+ `train_ratio`: it defines the train set. Default value set to 0.5.
+ `val_ratio`: it defines the validation set. Default value set to 0.75.
+ `early_stop`: number of epochs with no improvement after which training will be stopped. Default value set to 7.
+ `batch_size`: number of samples that will be propagated through the network. Default value set to 16.
+ `out_tensorboard_path`: name of the folder where to save the tensorboard results. If empty any board is stored. 
  Default value set to `None`.
+ `alpha`: confidence level of the confidence interval. Default value se to 0.95.

Execute the command `python3 -m run.run_parameter_detection.py -h` to display the arguments.
Example of execution:
```sh
python3 -m path.to.script.run.run_model.py --input_file run/data/dataset.json --output_path model_result  
--output_file model.h5 --window_size 12 --max_epochs 100 --train_ratio 0.5 --val_ratio 0.75 --out_tensorboard_path logdir
```
## Tensorboard

To visualize the evolution of the loss/accuracy trend of the train/validation process, run the following code from the 
root folder:
```sh
tensorboard --logdir logdir
```
## Tests

Run tests with Pytest: from the root folder of the project run the following code:
```sh
pytest 
```
For a specific file test it is also possible add to the previous command the file path you want to test. Example:
```sh
pytest tests/model/test_data_preprocess.py
```
Coverage:
```sh
coverage erase && \
coverage run --include='./deeplog_trainer/*' -m pytest && \
coverage report --include='./deeplog_trainer/*' -m && \
coverage html --include='./deeplog_trainer/*' -d './reports/coverage'

```
