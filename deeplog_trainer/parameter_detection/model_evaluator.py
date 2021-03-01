from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class ModelEvaluator:
    """
    Description: To evaluate the parameter detection anomaly detection model,
    Deeplog calculates the MSE between each vector in validation set and
    its prediction to generate a Gaussian distribution. At deployment, if the
    error between a prediction and an observed value vector is within a
    high-level of confidence interval of the above Gaussian distribution,
    the parameter value vector of the incoming log entry is considered normal,
    otherwise abnormal.
    """
    def __init__(self, logger):
        """
        :param logger: logger function from logging module
        """
        self.logger = logger

    def plot_time_series(self, scaler, dataset, val_idx: list, test_idx: list,
                         window_size: int, y_val_pred: list, y_test_pred: list):
        """
        :param scaler: Object of the package sklearn used to normalize the
        dataset
        :param dataset: input data of the model to be evaluated
        :param val_idx: indexes of the validation set
        :param test_idx: indexes of the testing set
        :param window_size: length of input array at each time step
        :param y_val_pred: predictions of the outputs of the validation set
        :param y_test_pred: predictions of the outputs of the testing set
        :return: times series graph for each parameter showing the evolution of
        the data and its prediction.
        """
        y_val_pred = scaler.inverse_transform(y_val_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        num_params = dataset.shape[1]
        for i in range(1, num_params):
            plt.figure(figsize=(20, 6))
            plt.plot(np.arange(len(dataset)), dataset[:, i], 'o-', mfc='none',
                     label='Real')
            plt.plot(val_idx[window_size-1:], y_val_pred[:, i], 'o-',
                     mfc='none', label='Valid Prediction')
            plt.plot(test_idx[window_size-1:], y_test_pred[:, i], 'o-',
                     mfc='none', label='Test Prediction', color='r')
            plt.axvspan(val_idx[0], val_idx[-1], color='orange', alpha=0.2)
            plt.axvspan(test_idx[0], test_idx[-1], color='red', alpha=0.2)
            plt.legend()
            plt.show()

    def get_MSEs(self, y_data: list, y_pred: list):
        """
        :param y_data: outputs array to predict
        :param y_pred: predictions of `y_data`
        :return: Mean squared errors array between `y_data` and `y_pred`
        """
        errors = []
        for i in range(len(y_data)):
            errors.append(mean_squared_error(y_data[i], y_pred[i]))
        return errors

    def build_confidence_interval(self, y_val: list, y_val_pred: list,
                                  alpha: float = 0.95):
        """
        :param y_val: outputs array of the validation set
        :param y_val_pred: predictions of the outputs of the validation set
        :param alpha: confidence level of the confidence interval
        :return: the two endpoints of the confidence interval of the Gaussian
        distribution of MSEs from the validation set, as well as a graph showing
        the Gaussian distribution and the related confidence interval.
        """
        MSE_val = self.get_MSEs(y_val, y_val_pred)
        mu = np.mean(MSE_val)
        sd = np.std(MSE_val)
        x = np.linspace(0, max(MSE_val), 1000)
        # Model the MSEs as a Gaussian distribution and build the confidence
        # interval of level alpha
        interval = stats.norm.interval(alpha, mu, sd)
        pdf = stats.norm.pdf(x, mu, sd)
        plt.plot(x, pdf)
        plt.fill_between(x, pdf, where=(x > interval[0]) & (x <= interval[1]),
                         color='blue')
        plt.title('Confidence interval of level {} for validation errors'
                  .format(alpha))
        plt.show()
        return interval

    def get_anomalies_idx(self, y_test: list, y_test_pred: list, interval: list,
                          alpha: float):
        """
        :param y_test: outputs array of the testing set
        :param y_test_pred: predictions of the outputs of the testing set
        :param interval: confidence interval of the Gaussian distribution built
        with the MSEs from the validation set
        :param alpha: confidence level of the confidence interval
        :return: indexes of data labeled as anomalous as well as a graph showing
        the trend of the MSEs.
        """
        MSE_test = self.get_MSEs(y_test, y_test_pred)
        anomalies_idx = [i for i in range(len(MSE_test))
                         if MSE_test[i] > interval[1]
                         or MSE_test[i] < interval[0]]
        plt.hlines(interval[1], 0, len(MSE_test), colors='red', linestyles='--',
                   label='CI={:d}%'.format(int(alpha*100)))
        plt.legend()
        plt.title('Value vector errors')
        plt.ylabel('MSE')
        plt.plot(MSE_test)
        plt.show()
        return anomalies_idx
