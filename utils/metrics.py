__all__ = ["scaling", "DataEval"]

from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
import numpy as np

"""
Scaling and metrics
auth: Methodfunc - Kwak Piljong
date: 2021.08.26
modify date: 2021.08.27
version: 0.2
"""


def scaling(data, method="minmax"):
    """
    select method : minmax, normal, robust, standard
    """

    if method == "minmax":
        sc = MinMaxScaler()

    elif method == "robust":
        sc = RobustScaler()

    elif method == "normal":
        sc = Normalizer()

    elif method == "standard":
        sc = StandardScaler()

    else:
        raise "Not support scale"

    sc_data = sc.fit_transform(data)

    return sc, sc_data


class DataEval:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        self.y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    def get(self):
        mse = self.__mse()
        rmse = self.__rmse()
        mae = self.__mae()
        mape = self.__mape()
        acc = self.__check_acc()

        return mse, rmse, mae, mape, acc

    def __mse(self):
        mse = np.mean(np.square(np.subtract(self.y_true, self.y_pred)))
        return mse

    def __rmse(self):
        rmse = np.sqrt(np.mean(np.square(np.subtract(self.y_true, self.y_pred))))
        return rmse

    def __mae(self):
        mae = np.mean(np.abs(np.subtract(self.y_true, self.y_pred)))
        return mae

    def __mape(self):
        mape = (
            np.mean(np.abs(np.subtract(self.y_true, self.y_pred) / self.y_true)) * 100
        )
        return mape

    def __check_acc(self):
        """
        calc acc
        """
        y_true = self.y_true.ravel()
        y_pred = self.y_pred.ravel()
        error_rate = (np.abs(y_pred - y_true) / y_true) * 100
        calc_acc = np.array([100 - i for i in error_rate if i != np.inf])

        acc = np.zeros(calc_acc.shape)

        for i in range(len(calc_acc)):
            if calc_acc[i] < 0:
                acc[i] = 0
            else:
                acc[i] = calc_acc[i]

        return acc
