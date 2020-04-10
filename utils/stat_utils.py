import numpy as np


class StatUtils:

    @staticmethod
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())