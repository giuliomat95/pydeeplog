import numpy as np
import tensorflow as tf
class MockModel:
    def __init__(self, model_path: str = None):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data, epochs,
            batch_size, callbacks, shuffle, verbose):
        return self



