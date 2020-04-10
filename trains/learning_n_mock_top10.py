import pandas as pd

from operator import itemgetter

from data.stocks import Stocks
from trains.learning_n_mock_investment import LearningNMockInvestment
from trains.mock_investment import MockInvestment
from data.data_utils import DataUtils


class LearningNMockTop10:
    """학습시키고 too10을 구입하는 방법으로 모의투자를 실행한다."""

    # 학습 결과의 컬럼명 정의
    RESULT_COLUMNS = ['no', 'code', 'name', 'last_pred_ratio', 'last_close_money', 'last_money', 'stock_count', 'all_invest_result', 'all_stock_count', 'rmse']

    MAX_PERCENT = 30
    MAX_RMSE = 0.03

    def __init__(self, params):
        self.params = params

    def train_n_invests_top10(self, corps):
        """ 상위 10개를 사는 방법으로 모의투자를 실행한다. """
        learning_invest = LearningNMockInvestment(self.params)
        invest_count = self.params.invest_count
        invest_data = []
        for i in range(invest_count):
            j=0
            for index, corp_data in corps.iterrows():
                if i == 0:
                    invest_row = self.train_top10(i, j, corp_data)
                    invest_data.append(invest_row)
                else:
                    invest_row = invest_data[j]
                    self.train_top10(i, j, corp_data, invest_row)
                #print(i, j, invest_row)
                j += 1
            self.invest_top10(i, invest_data)
            print(invest_data)
        self.sell_all_stock(invest_data)
        print("final", invest_data)
        df_invest_data = pd.DataFrame(invest_data, columns=self.RESULT_COLUMNS)
        DataUtils.save_excel(df_invest_data, learning_invest.get_result_file_path())

    def train_top10(self, i, j, corp_data, invest_row=None):
        """ top10 모의투자 방법을 위하여 학습을 시킨다."""
        learning_invest = LearningNMockInvestment(self.params)
        invest = MockInvestment(self.params)

        invest_count = self.params.invest_count
        if invest_row is None:
            corp_code = corp_data['종목코드']
        else:
            corp_code = invest_row[1]
        stocks = Stocks(self.params)
        stock_data = stocks.get_stock_data(corp_code)
        stock_data_now = stock_data[:i - invest_count]
        rmse_val, train_cnt, data_params, dataX_last, scaler_close = learning_invest.train(corp_code, stock_data_now)

        last_money, last_predict, invest_predicts, all_invest_money = invest.invest(corp_code, dataX_last, data_params)
        last_close_money, last_pred_money = invest.get_real_money(data_params, scaler_close, last_predict)
        last_pred_ratio = (last_pred_money - last_close_money) / last_close_money * 100

        if invest_row is None:
            corp_name = corp_data['회사명']
            all_invest_money, all_stock_count = invest.buy_stock(self.params.invest_money, last_close_money, 0)
            invest_row = [j, corp_code, corp_name, last_pred_ratio, last_close_money, 0, 0, all_invest_money, all_stock_count, rmse_val]
        else:
            #print(invest_row)
            invest_row[3] = last_pred_ratio
            invest_row[4] = last_close_money
        return invest_row

    def is_top10(self, last_pred_ratio, top_cnt, rmse):
        return last_pred_ratio < self.MAX_PERCENT and top_cnt < 10 and rmse < self.MAX_RMSE

    def invest_top10(self, i, invest_data):
        """ top10 방법으로 모의투자한다."""
        invest = MockInvestment(self.params)
        invest_data.sort(key=itemgetter(3), reverse=True)
        data_len = len(invest_data)

        # 주식을 판다.
        if i==0:
            selled_cnt = 0
            total_money = self.params.invest_money * 10
        else:
            selled_cnt = 0
            total_money = 0
            top_cnt = 0
            for i in range(data_len):
                invest_row = invest_data[i]
                last_pred_ratio = invest_row[3]
                now_close = invest_row[4]
                last_money = invest_row[5]
                now_stock_cnt = invest_row[6]
                rmse = invest_row[9]

                if self.is_top10(last_pred_ratio, top_cnt, rmse):
                    top_cnt += 1
                    total_money += last_money
                    invest_row[5] = 0
                    if now_stock_cnt > 0:
                        selled_cnt += 1
                else:
                    now_money, now_stock_cnt = invest.sell_stock(last_money, now_close, now_stock_cnt)
                    total_money += now_money
                    invest_row[5] = 0
                    invest_row[6] = now_stock_cnt

        # 주식을 구매한다.
        top_cnt = 0
        allow_money = total_money / (10 - selled_cnt)
        for i in range(data_len):
            invest_row = invest_data[i]
            last_pred_ratio = invest_row[3]
            rmse = invest_row[9]
            if self.is_top10(last_pred_ratio, top_cnt, rmse):
                #print("before", i, invest_row)
                top_cnt += 1
                now_stock_cnt = invest_row[6]
                if now_stock_cnt == 0:
                    now_close = invest_row[4]
                    now_money, now_stock_cnt = invest.buy_stock(allow_money, now_close, now_stock_cnt)
                    invest_row[5] = now_money
                    invest_row[6] = now_stock_cnt
                    #print("after", i, invest_row)

        invest_data.sort(key=itemgetter(0))

    def sell_all_stock(self, invest_data):
        """인덱스를 구하기 위하여 처음에 산 주식을 모두 판다."""
        invest = MockInvestment(self.params)
        data_len = len(invest_data)
        for i in range(data_len):
            invest_row = invest_data[i]
            now_close = invest_row[4]
            last_money = invest_row[5]
            now_stock_cnt = invest_row[6]
            now_money, now_stock_cnt = invest.sell_stock(last_money, now_close, now_stock_cnt)
            invest_row[5] = now_money
            invest_row[6] = now_stock_cnt

            all_invest_money = invest_row[7]
            all_stock_count = invest_row[8]
            all_invest_money, all_stock_count = invest.sell_stock(all_invest_money, now_close, all_stock_count)
            invest_row[7] = all_invest_money
            invest_row[8] = all_stock_count