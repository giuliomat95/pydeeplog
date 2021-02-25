import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
import logging as logger

from deeplog_trainer.parameter_detection.data_preprocess import DataPreprocess
from deeplog_trainer.model.training import ModelTrainer
from deeplog_trainer.model.model_manager import ModelManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logger.basicConfig(stream=sys.stdout, level=logger.INFO, format='%(message)s')

# generate synthetic dataset to evaluate parameter anomaly detection model
# noise: The standard deviation of the gaussian noise applied to the output.
x, y = make_regression(n_samples=100, n_features=1, noise=1)
dataset = np.stack((x[:, 0], y), axis=1)
window_size = 10
num_params = np.shape(dataset)[1]
data_preprocess = DataPreprocess()
train_idx, val_idx, test_idx = \
        data_preprocess.split_idx(len(dataset), train_ratio=0.7,
                                  val_ratio=0.85)
train_dataset = dataset[train_idx]
val_dataset = dataset[val_idx]
test_dataset = dataset[test_idx]
sc_X = StandardScaler()
train_dataset = sc_X.fit_transform(train_dataset)
val_dataset = sc_X.transform(val_dataset)
test_dataset = sc_X.transform(test_dataset)
logger.info('Datasets sizes: {}, {}, {}'.format(len(train_idx), len(val_idx),
                                                len(test_idx)))
model_manager = ModelManager()
model = model_manager.build(ModelManager.MODEL_TYPE_LOG_PARAMS,
                            input_size=window_size,
                            num_params=num_params)
X_train, y_train = data_preprocess.generate(train_dataset, window_size)
X_val, y_val = data_preprocess.generate(val_dataset, window_size)
X_test, y_test = data_preprocess.generate(test_dataset, window_size)
print("x_train.shape: {}".format(X_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("x_val.shape: {}".format(X_val.shape))
print("y_val.shape: {}".format(y_val.shape))
print("x_test.shape: {}".format(X_test.shape))
print("y_test.shape: {}".format(y_test.shape))
print('=======================')
model_trainer = ModelTrainer(logger, epochs=200, early_stop=100, batch_size=512)
# Run training and validation to fit the model
history = model_trainer.train(model, [X_train, y_train], [X_val, y_val],
                              metric_index='mse')
model.summary()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
pred = model.predict(X_val, verbose=1).reshape(-1)
plt.scatter(train_dataset[:, 0], train_dataset[:, 1])
plt.scatter(val_dataset[window_size:, 0], pred, c="red")
plt.show()
print(pred)
print(y_test)
