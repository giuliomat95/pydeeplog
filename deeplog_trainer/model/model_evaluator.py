import numpy as np


class ModelEvaluator:
    def __init__(self, model, X_data, y_data, top_k):
        """
        Description: During detection stage, given an historical sequence of
        integers in input, the LSTM model outputs, thanks to the activation
        function "softmax", a probability distribution over the n log key
        classes. Lastly, fixed the integer parameter top_k, DeepLog detects the
        testing key as an anomaly if it's not among the "top_k" keys with the
        greatest probability value of being the next key, and as normal
        otherwise.
        Attributes:
        :param model: model to be evaluated
        :param top_k: top k candidates parameter
        :param X_data: input array
        :param y_data: test array
        """
        self.model = model
        self.top_k = top_k
        self.X_data = X_data
        self.y_data = y_data
        self.n_items = len(self.X_data)

    def get_anomalies_idx(self):
        """
        Returns the indexes of testing keys labeled as anomalous during
        detection stage
        """
        y_pred = self.predict()
        anomalies_idx = [i for i in range(self.n_items) if
                         np.argmax(self.y_data[i]) not in y_pred[i]]
        return anomalies_idx

    def predict(self):
        """Returns  the top_k indexes with the highest probability value"""
        y_pred = self.model.predict(self.X_data)  # Return a probability value
        # for each index between 1 and the number of tokens of being the next
        # key of the previous sequence
        # assert len(y_pred[1])==num_tokens
        y_pred = y_pred.argsort()[:, -self.top_k:]
        return y_pred

    def compute_scores(self):
        """
        Returns a dictionary with the number of items parsed, the number of keys
        labeled as normal and the accuracy value.
        """
        n_normal = self.n_items - len(self.get_anomalies_idx())
        accuracy = self.n_items and n_normal / self.n_items
        # returns 0 if n_items = 0

        return {
            'n_items': self.n_items,
            'n_normal': n_normal,
            'accuracy': accuracy,
        }
