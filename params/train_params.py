from sklearn import preprocessing
import numpy as np

from params.data_params import DataParams


class TrainParams:

    data_params:DataParams = None
    dataX_last:np.ndarray = None
    scaler_close:preprocessing.MinMaxScaler = None

    train_count:int = 0
    test_rmse:np.float = 999999
    test_rmse_list:list = []

    test_predict:np.ndarray

