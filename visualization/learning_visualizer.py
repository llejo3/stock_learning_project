import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from data.data_utils import DataUtils
from params.data_params import DataParams
from visualization.main_visualizer import MainVisualizer


class LearningVisualizer:
    """학습관련 그래프를 그린다."""

    def __init__(self, params):
        self.params = params
        self.main = MainVisualizer(params)

    def draw_rmses(self, train_rmse, test_rmse, corp_name):
        """RMSE 그래프를 그린다."""
        dir = 'trains'
        dir_chart = os.path.join(self.main.DIR_CHARTS, dir)
        DataUtils.create_dir(dir_chart)
        rmse_data = {'train': train_rmse, 'test': test_rmse}
        #print(len(train_rmse), len(test_rmse))
        df_rmses = pd.DataFrame.from_dict(rmse_data)

        self.draw_rmse_seaborn(df_rmses, dir, corp_name)
        self.save_csv(df_rmses, dir, corp_name)


    def get_file_path(self, dir, corp_name, extension):
        """저장할 파일 경로 """
        return os.path.join(self.main.DIR_CHARTS, dir, corp_name + "." + extension)


    def draw_predictions(self, corp_name, scaler_close, data_params, test_predict, invest_predicts=None):
        """예측 그래프를 그린다."""
        dir = 'predicts'
        dir_chart = os.path.join(self.main.DIR_CHARTS, dir)
        DataUtils.create_dir(dir_chart)

        if invest_predicts is not None and test_predict is not None:
            predicts = np.append(test_predict, invest_predicts)
            #print(len(test_predict), len(invest_predicts))
        elif test_predict is not None:
            predicts = test_predict
        else:
            predicts = invest_predicts
        dataY = np.append(data_params.testY, data_params.investY)

        preds = []
        for pred in predicts:
            #print(pred, scaler_close)
            preds.append(DataUtils.inverse_scaled_data(scaler_close, pred))

        close_values = []
        for y in dataY:
            close_values.append(DataUtils.inverse_scaled_data(scaler_close, y))

        df_data = self.get_predicts_data(data_params, close_values, preds)
        df_data['date'] = pd.to_datetime(df_data['date'], format='%Y.%m.%d')
        df_data_index = df_data.set_index('date')
        self.draw_predictions_seaborn(df_data_index, dir, corp_name)
        self.save_csv(df_data, dir, corp_name)

    def get_predicts_data(self, data_params:DataParams, close_values:list, preds:list) -> pd.DataFrame:
        testY_date = data_params.testY_date
        investY_date = data_params.investY_date
        dataY_date = testY_date.append(investY_date, ignore_index=True)

        preds_len = len(preds)
        close_len =len(close_values)
        if preds_len > close_len:
            preds = preds[preds_len-close_len:]
        elif preds_len < close_len:
            close_values = close_values[close_len-preds_len:]
            dataY_date = dataY_date[close_len-preds_len:]

        predicts_data ={'date': dataY_date[1:], 'close': close_values, 'predict': preds}
        return pd.DataFrame.from_dict(predicts_data)


    def draw_predictions_seaborn(self, pridects_data, dir, corp_name):
        file_path = self.get_file_path(dir, corp_name, 'png')
        plt = self.main.get_plt()
        fig, ax = plt.subplots(figsize=self.main.FIG_SIZE)
        sns.lineplot(data=pridects_data).set_title(corp_name)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))
        #fig.autofmt_xdate()
        ax.set(xlabel='Date', ylabel='Price(원)')
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

        plt.grid(color='k', linestyle='dotted', linewidth=1, alpha=0.4)
        fig.savefig(file_path)
        plt.close()

    def draw_rmse_seaborn(self, df_rmses, dir, corp_name):
        """Seaborn 차트로 그래프를 그린다."""
        file_path = self.get_file_path(dir, corp_name, "png")
        plt = self.main.get_plt()
        fig, ax = plt.subplots(figsize=self.main.FIG_SIZE)
        sns.lineplot(data=df_rmses).set_title(corp_name)
        plt.grid(color='k', linestyle='dotted', linewidth=1, alpha=0.4)
        fig.savefig(file_path)
        plt.close()

    def save_csv(self, df_rmses, dir, corp_name):
        """Seaborn 차트로 그래프를 그린다."""
        file_path = self.get_file_path(dir, corp_name, "csv")
        DataUtils.save_csv(df_rmses, file_path)

