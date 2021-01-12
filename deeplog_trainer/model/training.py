import tensorflow as tf
import numpy as np
import os


class ValLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, logger, loss_index='loss', metric_index='accuracy'):
        """
        Attributes
        :param logger: logger function from logging module
        :param loss_index: loss function name
        :param metric_index: metric index
        """
        self.logger = logger
        self.loss_index = loss_index
        self.metric_index = metric_index
        # build dictionary to save the best loss and metric values during the training and the validation process
        self.best_loss = {'train': np.Inf, 'val': np.Inf}
        self.best_metric = {'train': 0, 'val': 0}
        # Let's declare that the ValLossLogger class inherits from the Callback class in Keras
        super().__init__()

    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.logger.info('Start training')

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch printing out the corresponding loss/accuracy values"""
        if logs['val_' + self.metric_index] > self.best_metric['val']:
            self.best_loss['train'] = logs[self.loss_index]
            self.best_loss['val'] = logs['val_' + self.loss_index]
            self.best_metric['train'] = logs[self.metric_index]
            self.best_metric['val'] = logs['val_' + self.metric_index]

        msg = 'Epoch: %d - Loss: %.4f (acc.: %.4f) - Val. loss:  %.4f (acc.: %.4f)' % (
            epoch, logs[self.loss_index], logs[self.metric_index],
            logs['val_' + self.loss_index], logs['val_' + self.metric_index])
        self.logger.info(msg)

    def on_train_end(self, logs=None):
        """Called at the end of the training process. The loss/accuracy value of the best model are displayed"""
        self.logger.info('Loss: %.4f (acc.: %.4f) - Val. loss:  %.4f (acc.: %.4f)' % (
            self.best_loss['train'], self.best_metric['train'],
            self.best_loss['val'], self.best_metric['val']))
        self.logger.info('Training finished')


class ModelTrainer:

    def __init__(self, logger, epochs=50, batch_size=512, early_stop=7, checkpoints_path='artifacts'):
        """
        Description: Once the numerical representation of each session is ready, DeepLog treats these sequences as a
        Multi-class Time Series Classification, which is a supervised learning problem aiming to predict class labels
        over the time based on past behaviour.
        :param logger: logger function from logging module
        :param epochs: Number of times the entire dataset is passed forward and backward by the neural network
        :param batch_size: Number of samples that will be propagated through the network.
        :param early_stop: Number of epochs with no improvement after which training will be stopped.
        :param checkpoints_path: filepath where to store the checkpoints
        """
        self.logger = logger
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.checkpoints_path = checkpoints_path

    def train(self, model, train_dataset, val_dataset):
        """
        Given the model, the train and validation set as arguments, the method train process is trigger.
        The model checkpoints allows to save the model according to the 'monitor' index.
        """
        train_logger = ValLossLogger(self.logger)
        early_stop = tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            patience=self.early_stop,
            verbose=0)
        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)
        filepath = self.checkpoints_path + '/checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5'
        # Periodically save the model.
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=0,
                                                        save_best_only=True, save_weights_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.checkpoints_path, histogram_freq=0,
                                                              embeddings_freq=0)

        history = model.fit(
            train_dataset[0],
            train_dataset[1],
            validation_data=(val_dataset[0], val_dataset[1]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[train_logger, model_ckpt, early_stop, tensorboard_callback],
            shuffle=False,
            verbose=1
        )
