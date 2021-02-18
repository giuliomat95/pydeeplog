import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
import os


class ModelManager:
    def __init__(self, input_size, num_tokens, lstm_units):
        """
        Attributes:
        :param input_size: Length of network input object
        :param num_tokens: Number of unique keys in the training file
        :param lstm_units: Number of LSTM units in each layer of the network
        """
        self.input_size = input_size
        self.lstm_units = lstm_units
        self.num_tokens = num_tokens
    # This is the factory method
    @staticmethod
    def build(self, model_type: str):
        if model_type == 'log_keys':
            return self.log_keys_model()
        elif model_type == 'log_params':
            return self.log_params_model()

    def log_keys_model(self):
        # Consider using an embedding if there are too many different input
        # classes
        x = Input(shape=(self.input_size, self.num_tokens))
        x_input = x

        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = LSTM(self.lstm_units, return_sequences=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tfa.layers.WeightNormalization(Dense(256, activation='relu'))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tfa.layers.WeightNormalization(Dense(128, activation='relu'))(x)
        x = Dense(self.num_tokens, activation='softmax')(x)

        model = Model(inputs=x_input, outputs=x)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            # Adam algorithm set as optimizer gave the best results in terms of
            # accuracy
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=['accuracy']
        )
        return model

    def log_params_model(self):
        x = Input(shape=(self.input_size, self.num_tokens))
        x_input = x
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tfa.layers.WeightNormalization(Dense(256, activation='relu'))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tfa.layers.WeightNormalization(Dense(128, activation='relu'))(x)
        x = Dense(self.num_tokens, activation='linear')(x)
        model = Model(inputs=x_input, outputs=x)

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            # Adam algorithm set as optimizer gave the best results in terms of
            # accuracy
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=['mse']
        )
        return model

    def save(self, model, output_path, output_file):
        model.save(os.path.join(output_path, output_file), save_format='h5')

    def load(self, filepath):
        model = load_model(filepath)
        return model
