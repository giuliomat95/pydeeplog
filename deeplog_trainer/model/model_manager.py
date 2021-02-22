import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization
import os


class ModelManager:
    MODEL_TYPE_LOG_KEYS = 'log_keys'
    MODEL_TYPE_LOG_PARAMS = 'log_params'

    def __init__(self, input_size, lstm_units):
        """
        Attributes:
        :param input_size: Length of network input object
        :param num_tokens: Number of unique keys in the training file
        :param lstm_units: Number of LSTM units in each layer of the network
        """
        self.input_size = input_size
        self.lstm_units = lstm_units

    # This is the factory method
    def build(self, model_type: str, **kwargs):
        if model_type == ModelManager.MODEL_TYPE_LOG_KEYS:
            if not ('num_tokens' in kwargs and isinstance(kwargs['num_tokens'],
                                                          int)):
                raise Exception('Provide right params')
            return self._build_log_keys_model(kwargs['num_tokens'])
        elif model_type == ModelManager.MODEL_TYPE_LOG_PARAMS:
            if not ('num_params' in kwargs and isinstance(kwargs['params'],
                                                          int)):
                raise Exception('Provide right params')
            return self._build_log_params_model(kwargs['num_params'])
        else:
            raise Exception('Model type unknown')

    def _build_log_keys_model(self, num_tokens):
        # Consider using an embedding if there are too many different input
        # classes
        x = Input(shape=(self.input_size, num_tokens))
        x_input = x

        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = LSTM(self.lstm_units, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = WeightNormalization(Dense(256, activation='relu'))(x)
        x = BatchNormalization()(x)
        x = WeightNormalization(Dense(128, activation='relu'))(x)
        x = Dense(num_tokens, activation='softmax')(x)

        model = Model(inputs=x_input, outputs=x)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            # Adam algorithm set as optimizer gave the best results in terms of
            # accuracy
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=['accuracy']
        )
        return model

    def _build_log_params_model(self, num_params):
        x = Input(shape=(self.input_size, num_params))
        x_input = x

        x = LSTM(self.lstm_units, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = WeightNormalization(Dense(256, activation='relu'))(x)
        x = BatchNormalization()(x)
        x = WeightNormalization(Dense(128, activation='relu'))(x)
        x = Dense(num_params, activation='linear')(x)
        model = Model(inputs=x_input, outputs=x)

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=['mse']
        )
        return model

    def save(self, model, output_path, output_file):
        model.save(os.path.join(output_path, output_file), save_format='h5')

    def load(self, filepath):
        model = load_model(filepath)
        return model
