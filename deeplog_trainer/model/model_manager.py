import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization
import os


class ModelManager:
    MODEL_TYPE_LOG_KEYS = 'log_keys'
    MODEL_TYPE_LOG_PARAMS = 'log_params'
    DEFAULT_LSTM_UNITS = 64

    # This is the factory method
    def build(self, model_type: str, input_size: int, **kwargs):
        lstm_units = kwargs.get('lstm_units', ModelManager.DEFAULT_LSTM_UNITS)
        if model_type == ModelManager.MODEL_TYPE_LOG_KEYS:
            self._validate_log_keys_kwargs(kwargs)
            return self._build_log_keys_model(input_size=input_size,
                                              lstm_units=lstm_units,
                                              num_tokens=kwargs['num_tokens'])
        elif model_type == ModelManager.MODEL_TYPE_LOG_PARAMS:
            self._validate_log_params_kwargs(kwargs)
            return self._build_log_params_model(input_size=input_size,
                                                lstm_units=lstm_units,
                                                num_params=kwargs['num_params'])
        else:
            raise Exception('Model type unknown')

    def _validate_log_keys_kwargs(self, kwargs):
        if not ('num_tokens' in kwargs and isinstance(kwargs['num_tokens'],
                                                      int)):
            raise Exception('Provide right params')

    def _validate_log_params_kwargs(self, kwargs):
        if not ('num_params' in kwargs and isinstance(kwargs['num_params'],
                                                      int)):
            raise Exception('Provide right params')

    def _build_log_keys_model(self, input_size: int, lstm_units: int,
                              num_tokens: int):
        # Consider using an embedding if there are too many different input
        # classes
        x = Input(shape=(input_size, num_tokens))
        x_input = x

        x = LSTM(lstm_units, return_sequences=True)(x)
        x = LSTM(lstm_units, return_sequences=False)(x)
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

    def _build_log_params_model(self, input_size: int, lstm_units: int,
                                num_params: int):
        x = Input(shape=(input_size, num_params))
        x_input = x

        x = LSTM(lstm_units, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = WeightNormalization(Dense(256))(x)
        x = BatchNormalization()(x)
        x = WeightNormalization(Dense(128))(x)
        x = Dense(num_params, activation='linear')(x)
        model = Model(inputs=x_input, outputs=x)

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=['mse']
        )
        return model

    def save(self, model, output_path: str, output_file: str):
        model.save(os.path.join(output_path, output_file), save_format='h5')

    def load(self, filepath: str):
        model = load_model(filepath)
        return model
