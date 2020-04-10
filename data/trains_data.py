from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np
from statistics import mean
import pandas as pd

from data.stocks import Stocks
from params.data_params import DataParams
from params.global_params import GlobalParams
from utils.date_utils import DateUtils


class TrainsData:
    """학습을 위한 데이터를 만든다."""

    def __init__(self, params:GlobalParams):
        self.params = params

    @staticmethod
    def to_ndarray(cols_data):
        """matrix 데이터로 변경한다."""
        if isinstance(cols_data, Series):
            return np.reshape(list(cols_data), (-1, 1))
        elif isinstance(cols_data, DataFrame):
            return cols_data.values

    def get_scaled_cols(self, data, column_name, scaler=None):
        """컬럼을 스케일링을 시킨다."""
        scale_data = self.to_ndarray(data[column_name])
        scale_data = scale_data.astype(float)
        if scaler is None:
            scaler = preprocessing.MinMaxScaler()
            #scaler = preprocessing.StandardScaler()
            return scaler.fit_transform(scale_data), scaler
        else:
            return scaler.transform(scale_data), scaler

    def add_mean_line(self, data):
        """데이터를 스케일링 시킨다."""

        closes = []
        lines = [5, 10, 20, 60, 120]
        last_line = lines[len(lines) -1]
        values = []
        for index, item in data.iterrows():
            closes.append(item['close'])
            closes_cnt = len(closes)
            if closes_cnt > last_line:
                closes.pop(0)

            line_means = {}
            for line_cnt in lines:
                start = 0
                if closes_cnt > line_cnt:
                    start = closes_cnt - line_cnt

                line_means['close_' + 'ml' + str(line_cnt)] = mean(closes[start:])
            values.append(line_means)
        df_lines = pd.DataFrame(values)
        return pd.concat([data, df_lines], axis=1)


    def get_scaled_data(self, data, scaler_close=None):
        """데이터를 스케일링 시킨다."""
        scaled_data = data.copy()

        scaled_data['close'], scaler_close = self.get_scaled_cols(scaled_data, 'close', scaler_close)
        for column in scaled_data.columns.values:
            if column == 'volume':
                scaled_data['volume'], _ = self.get_scaled_cols(scaled_data, 'volume')
            elif column not in ['date', 'close']:
                scaled_data[column], _ = self.get_scaled_cols(scaled_data, column, scaler_close)
        return scaled_data, scaler_close

    def get_dataXY(self, data):
        """RNN을 위한 데이터로 만든다. """
        columns = ['close', 'open', 'high', 'low', 'volume']
        for column in data.columns.values:
            if column not in columns and column != 'date':
                columns.append(column)
        x = self.to_ndarray(data[columns])
        y = self.to_ndarray(data['close'])
        y_date = data['date']

        dataX = []
        dataY = []
        seq_length = self.params.seq_length
        y_len = len(y)

        for i in range(0, y_len - seq_length):
            _x = x[i:i + seq_length]
            _y = y[i + seq_length]  # Next close price
            dataX.append(_x)
            dataY.append(_y)

        dataX_last = [x[y_len - seq_length: y_len]]
        return dataX, dataY, dataX_last, y_date

    def get_dataXY_plain(self, data):
        """통계를 위한 데이터로 만든다. """
        x = self.to_ndarray(data[['close', 'open', 'high', 'low', 'volume']])
        y = self.to_ndarray(data['close'])
        x = x[:-1]
        y = y[1:].reshape(-1)
        return x, y

    def get_train_test_plain(self, data):
        """통계를 위한 데이터로 만든다. """
        dp = DataParams()
        invest_count = self.params.invest_count
        x, y = self.get_dataXY_plain(data)

        data_count = len(y)
        train_size = int((data_count-invest_count) * self.params.train_percent / 100)
        train_last = data_count - invest_count
        dp.trainX = x[0:train_size]
        dp.testX = x[train_size:train_last]
        dp.investX = x[train_last:data_count]
        dp.trainY = y[0:train_size]
        dp.testY = y[train_size:train_last]
        dp.investY = y[train_last:data_count]
        return dp

    def split_train_test(self, dataX, dataY, invest_count, test_count=None, y_date=None):
        """train 및 test 데이터로 나눈다."""
        dp = DataParams()

        data_count = len(dataY)
        date_count = len(y_date)
        if test_count is None:
            train_count = int((data_count-invest_count) * self.params.train_percent / 100)
            test_count = data_count - train_count - invest_count
        else:
            train_count = data_count - test_count - invest_count
        train_last = data_count - invest_count

        dp.trainX = np.array(dataX[0:train_count])
        dp.testX = np.array(dataX[train_count:train_last])
        dp.investX = np.array(dataX[train_last:data_count])

        dp.trainY = np.array(dataY[0:train_count])
        dp.testY = np.array(dataY[train_count:train_last])
        dp.investY = np.array(dataY[train_last:data_count])

        dp.trainClose = dp.trainX[:,-1][:,[0]]
        dp.testClose = dp.testX[:, -1][:, [0]]

        dp.trainY_date = y_date[date_count-train_count-invest_count-1:date_count-invest_count-1]
        dp.investY_date = y_date[date_count-invest_count-1:]
        dp.testY_date = y_date[date_count-test_count-invest_count-1:date_count-invest_count-1]
        return dp

    def split_train_test_before(self, dataX, dataY, invest_count, test_count=None, y_date=None):
        """train 및 test 데이터로 나눈다."""
        test_split_count = self.params.test_split_count
        dp = DataParams()

        data_count = len(dataY)
        date_count = len(y_date)

        data_split = int((data_count - invest_count) / (test_split_count+1))

        if test_count is None:
            test_split = int((data_count-invest_count) * (1 - self.params.train_percent/100) / test_split_count )
        else:
            test_split = int((data_count - test_count - invest_count)  / test_split_count)

        invest_start = data_count - invest_count
        trainX = None
        trainY = None
        testX = None
        testY = None
        for i in range(test_split_count+1):
            if i == 0:
                trainX = dataX[:data_split]
                trainY = dataY[:data_split]
            elif i == test_split_count:
                trainX = np.append(trainX, dataX[data_split * i + test_split: invest_start], axis=0)
                trainY = np.append(trainY, dataY[data_split * i + test_split: invest_start], axis=0)
            else:
                trainX = np.append(trainX, dataX[data_split * i + test_split: data_split * (i + 1)], axis=0)
                trainY = np.append(trainY, dataY[data_split * i + test_split: data_split * (i + 1)], axis=0)

            if i != 0:
                if testX is None:
                    testX = dataX[data_split*i : data_split*i + test_split]
                    testY = dataY[data_split*i : data_split*i + test_split]
                else:
                    testX = np.append(testX, dataX[data_split*i : data_split*i + test_split], axis=0)
                    testY = np.append(testY, dataY[data_split*i  :data_split*i + test_split], axis=0)

        dp.trainX = np.array(trainX)
        dp.trainY = np.array(trainY)
        dp.testX = np.array(testX)
        dp.testY = np.array(testY)
        dp.investX = np.array(dataX[invest_start:])
        dp.investY = np.array(dataY[invest_start:])

        dp.investY_date = y_date[date_count - invest_count - 1:]
        return dp


    def get_train_test(self, data, scaler_close=None):
        """train, test 데이터로 만든다."""
        data = data.copy()
        data = data[(data[['close', 'open', 'high', 'low', 'volume']] != 0).all(1)]
        data.index = pd.RangeIndex(len(data.index))
        #data = self.add_mean_line(data)

        if self.params.invest_end_date is not None:
            data =  data.query("date<='{}'".format(self.params.invest_end_date))

        if self.params.invest_start_date is not None:
            invest_data = data.query("date>='{}'".format(self.params.invest_start_date))
            invest_count = len(invest_data.index) -1
            self.params.invest_count = invest_count
            invest_start_date_str = self.params.invest_start_date
        else:
            invest_count = 0
            self.params.invest_count = 0
            invest_start_date_str = data.tail(1)['date'].to_string(index=False)

        invest_start_date = DateUtils.to_date(invest_start_date_str)
        if hasattr(self.params, 'stock_training_period_years'):
            period = self.params.stock_training_period_years
            stock_start_date = DateUtils.add_years(invest_start_date, -period)
            stock_start_date = stock_start_date.strftime("%Y.%m.%d")
            data = data.query("date>='{}'".format(stock_start_date))

        test_count = None
        if hasattr(self.params, 'stock_test_period_years') and self.params.stock_test_period_years is not None:
            period = self.params.stock_test_period_years
            test_start_date = DateUtils.add_years(invest_start_date, -period)
            test_start_date = DateUtils.to_date_str(test_start_date)
            test_data = data.query("date>='{}'".format(test_start_date))
            test_count = len(test_data.index) - invest_count

        scaled_data, scaler_close = self.get_scaled_data(data, scaler_close)
        dataX, dataY, dataX_last, y_date = self.get_dataXY(scaled_data)
        data_params = self.split_train_test(dataX, dataY, invest_count, test_count, y_date)
        return data_params, scaler_close, dataX_last

if __name__ == '__main__':
    params = GlobalParams()
    stock = Stocks(params)
    stock_data = stock.get_stock_data(35720)
    trains_data = TrainsData(params)
    data = trains_data.add_mean_line(stock_data)
    scaled_data, scaler_close = trains_data.get_scaled_data(data)
    print(scaled_data.tail())
    # stock_start_date = DateUtils.to_date('2018.01.01') - relativedelta(years=22)
    # stock_start_date = stock_start_date.strftime("%Y.%m.%d")
    # print(stock_start_date)