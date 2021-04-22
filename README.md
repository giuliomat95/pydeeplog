# DeepLog Trainer

## Introduction

In this project, we provide a Deep Learning model, called **DeepLog**, that aims 
to detect anomalies occurred in a system logs. In particular, we propose an LSTM 
(Long Short Term Memory)-based model, a Recurrent Neural Network (RNN) 
capable of memorizing long-term dependencies over time.\
The architecture of Deeplog is composed by three main components: the 
`log key anomaly detection model`, the `parameter value anomaly detection 
model`, and the `workflow model` to diagnose detected anomalies.   
For more information read the following paper: 
https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf.

## Set up

### Dev environment

The first step is to create a virtual environment. You can do this with PyCharm 
or from the terminal as follows:

```sh
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

After this you should configure your IDE to use this environment.
On Pycharm it can be done on `Settings > Project > Python interpreter`.

### Docker containers

```sh
docker build --tag docker.devo.internal/dev/mlx/experiments/deeplog-trainer: \
latest .
```

```sh
docker run --detach \
    --network host \
    --name deeplog-trainer \
    docker.devo.internal/dev/mlx/experiments/deeplog-trainer:latest
```
## Implementation details

### Log Parser stage: Drain
As a first step, the free-text log entries are parsed into a structured 
representation, what we call the `parser stage`. The goal of this operation is 
to extract the constant message, called `log key` that recurs few times 
and store the variable part, known as the `parameter`, for each entry log. \
The state-of-the-art log parsing method is represented by Drain, a Depth-Tree 
based online log parsing method. For further details: 
https://pinjiahe.github.io/papers/ICWS17.pdf. \
Once that logs are parsed, each of them is encoded with the respective log key 
id and grouped in different "sessions". \
In this project we have  used the open source implementation 
[Drain3](https://github.com/IBM/Drain3).

### Log key anomaly detection model

Once the numerical representation of each session is ready, DeepLog treats these
sequences as a Multi-class Time Series Classification,  where each distinct log 
key defines a class. (`log key anomaly detection model`).\
At every time steps, given an  history of recent log keys in input, the LSTM 
model outputs a probability distribution over the n log key classes. Lastly, 
fixed the integer parameter `g`, DeepLog detects the testing key as an anomaly 
if it's not among the "g" keys with the greatest probability value of being the 
next key (the top g candidates), and as normal otherwise. Consequently, the 
parameter g works likewise a threshold for the number of anomalies detected.

### Workflows

At a later stage, the log key sequences are used to create a workflows model as 
a deterministic finite state machine 
(https://en.wikipedia.org/wiki/Finite-state_machine). In particular, we try to 
detect the divergence points caused by *concurrency*, i.e. when when several 
operations are present in the same task, or the paths that can be reduced to a 
loop.\
In this way, the workflow model is very useful towards enabling users to 
diagnose what had gone wrong in the execution of a task when an anomaly has been
 detected.

### Parameter value anomaly detection model

The last part of Deeplog's architecture, as said before, is the `parameter value
anomaly detection model`. In this stage Deeplog analyses the variable part of 
the message, what we have called the `paramaters`, to detect the anomalies that 
have not been shown as a deviation from a normal execution path in the first 
part, but as an irregular parameter value.\
For each key, the parameters are stored along with the elapsed time between one 
incoming log and another, in order to create a multivariate time series. Then, 
an LSTM forecasting model is used to detect the anomalies. \
With a similar approach to the first stage, the LSTM model takes in input, at 
each time step, an history sequence of value vectors, and try to predict the 
next one. To evaluate the model, the data is split in three sets: train, 
validation and test. For each vector in the validation set, we calculate the MSE
between him and its prediction and finally modeled as Gaussian distribution. \
At deployment, if the error between a prediction and an observed value vector is 
within a high-level of confidence interval of the above Gaussian distribution, 
the parameter value vector of the incoming log entry is considered normal,
and is abnormal otherwise. 

## Commands

### Run Drain
Before running Drain, set the parameters in the config file `drain3.ini` in the
working directory. \
Available parameters are:

- `[DRAIN]/sim_th` - similarity threshold (default 0.4)
- `[DRAIN]/depth` - depth of all leaf nodes (default 4)
- `[DRAIN]/max_children` - max number of children of an internal node 
    (default 100)
- `[DRAIN]/extra_delimiters` - delimiters to apply when splitting log message 
    into words (in addition to whitespace) (default none). 
    Format is a Python list e.g. `['_', ':']`.
- `[MASKING]/masking` - parameters masking - in json format (default "")
- `[SNAPSHOT]/snapshot_interval_minutes` - time interval for new snapshots 
    (default 1)
- `[SNAPSHOT]/compress_state` - whether to compress the state before saving it.
    This can be useful when using Kafka persistence. 
- `[ADAPTER_PARAMS]/adapter_type` - name of the method to be used to group logs
    in different sessions. Available ones: ['only_delimiter', 'delimiter+regex', 
    'only_identifier', 'interval_time']. 
- `[ADAPTER_PARAMS]/delimiter` - string sentence that calls a new session.
- `[ADAPTER_PARAMS]/regex` - identifier in format “r-string”. For example,
    it could be the id of a particular process.
- `[ADAPTER_PARAMS]/anomaly_labels` - list with the strings leading to an 
    anomaly. If any list is provided, the output file will contain a binary 
    flag that indicates whether it is an anomalous sequence (because a message 
    contains a string in the list) or not.
    Note: the current DeepLog implementation is unsupervised, thus it doesn't 
    use this flag. This flag can be set if other supervised algorithms are to be
    applied.
- `[ADAPTER_PARAMS]/time_format` - format of time. Example: '%H:%M:%S.%f'.
- `[ADAPTER_PARAMS]/delta` - dictionary indicating the time that must elapse 
    to create a new session. Example: {'minutes'=1, 'seconds'=30}
- `[ADAPTER_PARAMS]/logformat` - Format of the entry log. Example: 
    '\<Pid> \<Content>'. It must contain the word 'Content'.

Run the following code from terminal. The arguments `--input_file` and 
`--output_path` are respectively the filepath of the data to be parsed and the 
name of the folder where the results will be saved. The default output path is
`artifacts/drain_result`. The argument `--config_path`, instead, is the filepath
 of the config file. The default configuration file is `drain.ini` located in 
 the root folder.
```sh
python3 -m run.run_drain --input_file data/{filename}.log \
--config_file 
```

### Run Log key anomaly detection Model
To run the `run_log_key_detection_model.py` file, set the following parameters 
in the command line:
+ `input_file`: path of the input json dataset to parse. Default path: 
`artifacts/drain_result/data.json`.
+ `window_size`: length of chunks, input of the LSTM neural network. Default 
value set to 10.
+ `min_length`: the minimum length of a sequence to be parsed. Default value set
 to 4.
+ `output_path`: path of the directory where to save the trained model, as well 
as the config values. Default path: `artifacts/log_key_model_result`.
+ `LSTM_units`: number of units in each LSTM layer. Default value set to 64.
+ `train_ratio`: it defines the train set. Default value set to 0.7.
+ `val_ratio`: it defines the validation set. Default value set to 0.85.
+ `batch_size`: number of samples that will be propagated through the network. 
Default value set to 512.
+ `early_stop`: number of epochs with no improvement after which training will 
be stopped. Default value set to 7.
+ `max_epochs`: maximum number of epochs if the process is not stopped before by
 the early_stop. Default value set to 50.
+ `out_tensorboard_path`: name of the folder where to save the tensorboard 
results. If empty any board is stored. Default value set to `None`.

The model is saved in `h5` format with the name `log_key_model.h5` in the 
directory provided.
The parameters without default values are mandatory to run the file.  
Execute the command `python3 -m run.run_log_key_detection_model -h` to 
display the arguments.
Example of execution:
```sh
python3 -m run.run_log_key_detection_model --window_size 12 \
--max_epochs 100 --train_ratio 0.5 \
--val_ratio 0.75 --out_tensorboard_path logdir
```

### Run workflows model
To run the `run.workflow.py` file, set the following parameter in the command 
line:
+ `input_file`: path of the input json dataset to parse. Default path: 
`artifacts/drain_result/data.json`.
+ `output_path`: path of the directory where to save the workflows in a pickle 
 file. Default path: `artifacts/workflows`
+ `min_length`: the minimum length of a sequence to be parsed. Default value set
 to 4.
+ `train_ratio`: it defines the train set. Default value set to 0.7.
+ `val_ratio`: it defines the validation set. Default value set to 0.85.
+ `threshold`: threshold to calculate similarity between two sequences. 
 Default value: 0.8.
+ `back_steps`: number of step backwards to research similar workflows. Default 
 value: 1.

Example of execution:
```sh
python3 -m run.run_workflow --train_ratio 0.5 --val_ratio 0.75
```

### Run parameter value anomaly detection model
In order to evaluate the parameter value anomaly detection model, due to the 
absence of a dataset with log messages whose parameter values are mainly 
numerical, we used a synthetic data. In general, it should be provided a 
different matrix with all the parameters for each log key derived from the log 
parser stage.
 
To run the `run_parameter_detection_model.py` file, set the following parameters 
in the command line:
+ `input_file`: path of the input dataset to parse, with all the parameters of a
 specific log key message.
+ `output_path`: path of the directory where to save the trained model.
 Default path: `artifacts/log_par_model_result`.
+ `window_size`: length of chunks, input of the LSTM neural network. Default 
value set to 5.
+ `LSTM_units`: number of units in each LSTM layer. Default value set to 64.
+ `max_epochs`: maximum number of epochs if the process is not stopped before by
the early_stop. Default value set to 100.
+ `train_ratio`: it defines the train set. Default value set to 0.5.
+ `val_ratio`: it defines the validation set. Default value set to 0.75.
+ `early_stop`: number of epochs with no improvement after which training will 
be stopped. Default value set to 7.
+ `batch_size`: number of samples that will be propagated through the network. 
Default value set to 16.
+ `out_tensorboard_path`: name of the folder where to save the tensorboard 
results. If empty any board is stored. 
  Default value set to `None`.
+ `alpha`: confidence level of the confidence interval. Default value se to 
0.95.

The model is saved in `h5` format with the name `log_par_model.h5` in the 
directory provided.
Execute the command `python3 -m run.run_parameter_detection_model -h` to 
display the arguments.
Example of execution:
```sh
python3 -m run.run_parameter_detection_model --input_file data/dataset.json \
--output_path model_result \
--window_size 12 --max_epochs 100 --train_ratio 0.5 \
--val_ratio 0.75 --out_tensorboard_path logdir
```
### Tensorboard

To visualize the evolution of the loss/accuracy trend of the train/validation 
process, run the following code from the root folder:
```sh
tensorboard --logdir logdir
```
### Tests

Run tests with Pytest: from the root folder of the project run the following 
code:
```sh
pytest 
```
For a specific file test it is also possible add to the previous command the 
file path you want to test. Example:
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

## Quick Example
Let's see, for instance, how to apply, Deeplog on Batrasio system logs, a real 
time data set provided by the Devo platform. \
In Batrasio dataset, each session
starts with the delimiter “*TCP source connection created*”.  Every time the 
delimiter is detected in the content, a new session is created and it contains 
all the following messages until the delimiter is shown again. About the 
anomalies, instead, all the messages containing the text 
"*TCP source SSL error*" or "*TCP source socket error*" are labeled as abnormal.
We stored a sample of the dataset in the `data` folder, called 
`sample_batrasio.log`.

### Commands:
+ Drain: 
```sh 
python3 -m run.run_drain --input_file data/sample_batrasio.log 
```
+ Log Key anomaly detection:
```sh
python3 -m run.run_log_key_detection_model \
--window_size 10 --max_epochs 100 --train_ratio 0.5 \
--val_ratio 0.75 --out_tensorboard_path logdir
```
+ Workflow:
```sh
python3 -m run.run_workflow --train_ratio 0.5 --val_ratio 0.75
```
+ Parameter value anomaly detection:
```sh
python3 -m run.run_parameter_detection_model --input_file data/dataset.json \
--window_size 12 --max_epochs 100 --train_ratio 0.5 \
--val_ratio 0.75 --out_tensorboard_path logdir
```
