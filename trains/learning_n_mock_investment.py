import pandas as pd
import numpy as np
import sys, traceback
import os
from copy import deepcopy

from sklearn import preprocessing

from data.stocks import Stocks
from data.trains_data import TrainsData
from params.global_params import GlobalParams
from params.invest_params import InvestParams
from params.train_params import TrainParams
from trains.learning import Learning
from trains.mock_investment import MockInvestment
from data.data_utils import DataUtils
from data.corp import Corp
from visualization.invest_visualizer import InvestVisualizer
from visualization.learning_visualizer import LearningVisualizer


class LearningNMockInvestment:
    """학습시키고 모의투자를 실행한다."""

    DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(DIR, '..', 'result')

    # 학습 결과의 컬럼명 정의
    RESULT_COLUMNS = ['no', 'code', 'name', 'rmse', 'invest_result', 'all_invest_result', 'train_cnt', 'invest_date']

    # 예측 결과의 컬럼명 정의
    RESULT_COLUMNS_NEXT =['no', 'last_date', 'code', 'name', 'rmse', 'train_cnt', 'last_close_money',
                                'last_pred_money', 'last_pred_ratio', 'trade_money']

    def __init__(self, params:GlobalParams):
        self.params:GlobalParams = params

        if params.result_type == 'forcast':  # 예측의 경우
            self.result_columns = self.RESULT_COLUMNS_NEXT
        else:
            self.result_columns = self.RESULT_COLUMNS

        #self.logging = LearningLogging()


    def train_n_invest(self, corp_code:str, corp_name:str, no:int, invest_only:bool=False)->(list, list):
        """입력한 회사에 대해서 학습시키고 모의투자를 실행한다."""

        stocks = Stocks(self.params)
        stock_data = stocks.get_stock_data(corp_code)

        ip = InvestParams()
        if self.params.invest_start_date is None:
            tps = self.trains(corp_code, corp_name, stock_data, invest_only)
            ip.last_money = self.params.invest_money
            ip.index_money = ip.last_money
            if self.params.result_type == 'forcast':
                invest = MockInvestment(self.params)
                ip = invest.invest(corp_code, corp_name, tps, invest_only)
        else:
            invest = MockInvestment(self.params)
            tps = self.trains(corp_code, corp_name, stock_data, invest_only)
            ip = invest.invest(corp_code, corp_name, tps, invest_only)

        if invest_only:
            tp = self.get_train_params_4rmse(tps)
            test_predict = None
        else:
            tp = self.select_train_params(tps)
            test_predict = tp.test_predict
        if ip.test_predict is not None:
            test_predict = ip.test_predict

        try:
            if self.params.save_train_graph and (test_predict is not None or ip.predict_list is not None)  == True:
                visual = LearningVisualizer(self.params)
                visual.draw_predictions(corp_name, tp.scaler_close, tp.data_params, test_predict, ip.predict_list)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)

        return self.get_train_invest_result(no, corp_code, corp_name, stock_data, tp, ip), ip.daily_data

    def select_train_params(self, tps:dict) -> TrainParams:
        tps_len = len(tps)

        rmse_avg = 0
        selected_tp:TrainParams = None
        train_cnt = 0
        for train_model, tp in tps.items():
            rmse_avg += tp.test_rmse/tps_len
            train_cnt += tp.train_count
            if selected_tp is None:
                selected_tp = tp
            elif len(selected_tp.test_predict) > len(tp.test_predict):
                selected_tp = tp
        selected_tp.test_rmse = rmse_avg
        selected_tp.train_count = train_cnt
        return selected_tp

    def get_train_params_4rmse(self, tps:dict) -> TrainParams:
        tps_len = len(tps)

        rmse_avg = 0
        for train_model, tp in tps.items():
            rmse_avg += tp.test_rmse / tps_len
        selected_tp = next(iter(tps))
        selected_tp.test_rmse = rmse_avg
        return selected_tp


    def train_n_invest_twins(self, corp_code, corp_name, no)->list:
        """겨별 세션과 통합세션에서 예측한 값의 평균을 예측값으로 한다."""
        stocks = Stocks(self.params)
        trains_data = TrainsData(self.params)
        learning = Learning(self.params)
        params_all = GlobalParams('ALL_CORPS')
        learning_all = Learning(params_all)
        invest = MockInvestment(self.params)

        stock_data = stocks.get_stock_data(corp_code)
        data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data)
        tp = learning.learn(corp_code, corp_name, data_params)
        tp_all = learning_all.learn(corp_code, corp_name, data_params)
        ip =invest.invest_n_all(corp_code, dataX_last, data_params, params_all, scaler_close)
        tp.test_rmse = np.mean([tp.test_rmse, tp_all.test_rmse])
        tp.train_count = tp.train_count + tp_all.train_count
        return self.get_train_invest_result(no, corp_code, corp_name, stock_data, tp, ip)


    def get_train_invest_result(self, no, corp_code, corp_name, stock_data, tp, ip)->list:
        """결과를 보여준다."""
        invest = MockInvestment(self.params)
        if self.params.result_type == 'forcast':
            last_date = stock_data.tail(1)['date'].to_string(index=False).strip()
            last_close_money, last_pred_money, trade_money = invest.get_real_money(tp.data_params, tp.scaler_close, ip.last_predict)
            last_pred_ratio = (last_pred_money - last_close_money) / last_close_money * 100
            last_pred_ratio = "{:.2f}".format(last_pred_ratio) + "%"
            print(no, last_date, corp_code, corp_name, tp.test_rmse, tp.train_count, last_close_money, last_pred_money, last_pred_ratio, trade_money)
            if self.params.debug is True:
                print()
            return [no, last_date, corp_code, corp_name, tp.test_rmse, tp.train_count, last_close_money, last_pred_money, last_pred_ratio, trade_money]
        else:
            invest_date = self.params.invest_start_date + '~' + self.params.invest_end_date
            if self.params.debug is True:
                print(no, corp_code, corp_name, tp.test_rmse, ip.last_money, ip.index_money, tp.train_count, invest_date)
                print()
            return [no, corp_code, corp_name, tp.test_rmse, ip.last_money, ip.index_money, tp.train_count, invest_date]


    def trains(self, corp_code:str, corp_name:str, stock_data:pd.DataFrame, invest_only:bool=False) -> dict:
        """입력한 회사에 대해서 학습시킨다"""
        tps = {}
        model_type = self.params.model_type
        train_model = self.params.train_model
        for tm in train_model.split("_"):
            self.params.__init__(model_type, train_model=tm)
            tp = self.train(corp_code, corp_name, stock_data, None, invest_only)
            tps[tm] = tp
        self.params.__init__(model_type, train_model=train_model)
        return tps

    def train(self, corp_code:str, corp_name:str, stock_data:pd.DataFrame,
              scaler_close:preprocessing.MinMaxScaler=None, invest_only:bool=False) -> TrainParams:
        """입력한 회사에 대해서 학습시킨다"""

        trains_data = TrainsData(self.params)
        learning = Learning(self.params)

        data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data, scaler_close)
        if invest_only:
            tp = learning.get_test_rmse(corp_code, data_params)
        else:
            tp = learning.learn(corp_code, corp_name, data_params)
        tp.scaler_close = scaler_close
        tp.dataX_last = dataX_last
        return tp


    def train_n_invest_one(self, corp_code, corp_name, stock_data):
        """입력한 회사에 대해서 학습시키고 한번의 모의투자를 실행한다."""
        invest = MockInvestment(self.params)

        tp = self.train(corp_code, corp_name, stock_data)
        ip = invest.invest(corp_code, corp_name,  tp)
        return tp.test_rmse, ip.last_money, ip.index_money, tp.train_count

    def train_n_invests(self, corps, start_no=1, invest_only: bool = False) -> pd.DataFrame:
        """입력한 회사들에 대해서 학습시키고 모의투자를 실행한다."""
        comp_rmses = []
        no = 1
        invest_daily_data = []
        date: str = None
        for index, corp_data in corps.iterrows():
            if no < start_no:
                no += 1
                continue
            corp_code = corp_data['종목코드']
            corp_name = corp_data['회사명']
            try :
                result, invest_daily = self.train_n_invest(corp_code, corp_name, no, invest_only)
                date = result[7]
                if invest_daily is not None:
                    invest_daily_data.append(invest_daily)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                no += 1
                continue

            comp_rmses.append(result)
            if no % 10 == 0 and self.params.debug == True:
                df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
                DataUtils.save_csv(df_comp_rmses, self.get_result_file_path(date))
            no += 1

        df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
        DataUtils.save_csv(df_comp_rmses, self.get_result_file_path(date))
        #DataUtils.save_csv(df_comp_rmses, self.get_result_file_path(date))

        if len(invest_daily_data) > 1:
            try :
                visualizer = InvestVisualizer(self.params)
                return visualizer.draw_invest_daily(invest_daily_data, corps)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)

    def train_n_invests_for_all(self, corps):
        """입력한 회사들에 대해서 학습시키고 모의투자를 실행한다."""
        stocks = Stocks(self.params)
        trains_data = TrainsData(self.params)
        learning = Learning(self.params)
        invest = MockInvestment(self.params)
        
        list = []
        for index, corp_data in corps.iterrows():
            corp_code = corp_data['종목코드']
            corp_name = corp_data['회사명']
            try:
                stock_data = stocks.get_stock_data(corp_code)
                data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                continue
            list.append({'corp_code':corp_code, 'corp_name':corp_name, 'scaler_close':scaler_close,
                         'dataX_last':dataX_last, 'data_params':data_params, 'stock_data':stock_data})

        all_data_params = self.gather_data_params(list)
        tp = learning.learn("ALL_CORPS", "ALL_CORPS", all_data_params)

        invest_daily_data = []
        comp_rmses = []
        index = 1
        for item in list:
            corp_code = item['corp_code']
            corp_name = item['corp_data']
            dataX_last = item['dataX_last']
            data_params = item['data_params']
            scaler_close = item['scaler_close']
            stock_data = item['stock_data']
            ip = invest.invest(corp_code, corp_name, tp)

            visualizer = LearningVisualizer(self.params)
            visualizer.draw_predictions(corp_name, scaler_close, data_params, tp.test_predict, ip.predict_list)

            result, invest_daily = self.get_train_invest_result(index, corp_code, corp_name, tp.test_rmse, ip.last_money,
                                                                ip.index_money, tp.train_count, stock_data, data_params,
                                                                scaler_close, ip.last_predict)
            if invest_daily != None:
                invest_daily_data.append(invest_daily)
            comp_rmses.append(result)
            index += 1

        df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
        DataUtils.save_csv(df_comp_rmses, self.get_result_file_path())

        if len(invest_daily_data) > 1:
            try:
                visualizer = InvestVisualizer(self.params)
                return visualizer.draw_invest_daily(invest_daily_data, corps)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)


    def gather_data_params(self, list):
        g_data_params = None
        for item in list:
            data_params = item['data_params']
            if g_data_params is None:
                g_data_params = deepcopy(data_params)
            else:
                g_data_params.trainX = np.append(g_data_params.trainX, data_params.trainX, axis=0)
                g_data_params.trainClose = np.append(g_data_params.trainClose, data_params.trainClose, axis=0)
                g_data_params.trainY = np.append(g_data_params.trainY, data_params.trainY, axis=0)
                g_data_params.testX = np.append(g_data_params.testX, data_params.testX, axis=0)
                g_data_params.testClose = np.append(g_data_params.testClose, data_params.testClose, axis=0)
                g_data_params.testY = np.append(g_data_params.testY, data_params.testY, axis=0)
        return g_data_params

    def train_n_invests_for_name(self, corp_names:list, invest_only:bool=False) -> None:
        """회사이름으로 검색하여 학습시킴 """
        corp = Corp()
        comp_rmses = []
        no = 1
        date = None
        for corp_name in corp_names:
            corp_code = corp.get_corp_code(corp_name)
            try :
                result, invest_daily = self.train_n_invest(corp_code, corp_name, no, invest_only)
                date = result[1]
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                no += 1
                continue

            #result = self.let_train_invest(corp_code, corp_name, no)
            comp_rmses.append(result)
            no += 1
        df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
        DataUtils.save_csv(df_comp_rmses, self.get_result_file_path(date))
        #DataUtils.save_csv(df_comp_rmses, self.get_result_file_path(date))

    def forcasts(self, corps_n_date:list) -> None:
        """ 각 종목의 날짜로 예측을 수행한다. """
        comp_rmses = []
        no = 1
        date = None
        for corp_n_date in corps_n_date:
            corp_code = corp_n_date[0].replace("A", "")
            corp_name = corp_n_date[1]
            forcast_date = corp_n_date[2]

            params = GlobalParams('FORCAST')
            params.forcast_date = forcast_date
            invests = LearningNMockInvestment(params)
            result, invest_daily = invests.train_n_invest(corp_code, corp_name, no, False)
            comp_rmses.append(result)
            date = result[1]
            no += 1

        df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.result_columns)
        DataUtils.save_csv(df_comp_rmses, self.get_result_file_path(date))

    def get_result_file_path(self, date: str = None) -> str:
        """결과를 저장할 경로"""
        if date is None:
            file_name = self.params.result_file_name + '.txt'
        else:
            file_name = self.params.result_file_name + "_" + date + '.txt'
        if "_" in self.params.train_model:
            return  os.path.join(self.RESULT_DIR, self.params.train_model, self.params.ensemble_type, file_name)
        else:
            return os.path.join(self.RESULT_DIR, self.params.train_model, file_name)




