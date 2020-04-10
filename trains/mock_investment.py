from builtins import dict

import tensorflow.compat.v1 as tf
import math
import numpy as np
import sys, traceback
from sklearn.linear_model import LinearRegression

from data.data_utils import DataUtils
from params.invest_params import InvestParams
from trains.learning import Learning
from models.stacked_rnn import StackedRnn
from visualization.invest_visualizer import InvestVisualizer

tf.disable_v2_behavior()

class MockInvestment:
    """모의투자"""

    def __init__(self, params):
        self.params = params

    def invest_scaled_money(self, invest_scaled_predict, now_scaled_close, scaler_close, now_money, now_stock_cnt):
        """예측 값에 따라 매수 매도를 실행한다."""
        now_close = DataUtils.inverse_scaled_data(scaler_close, now_scaled_close)
        if invest_scaled_predict == -1:
            invest_predict = -1
        else:
            invest_predict = DataUtils.inverse_scaled_data(scaler_close, invest_scaled_predict)
        return self.invest_money(invest_predict, now_close, now_money, now_stock_cnt)

    def invest_money(self, invest_predict, now_close, now_money, now_stock_cnt):
        """예측 값에 따라 매수 매도를 실행한다."""
        invest_min_percent = self.params.invest_min_percent
        if invest_predict == -1:
            ratio = 100
        else:
            ratio = self.get_invest_ratio(invest_predict, now_close)

        if ratio > invest_min_percent and now_stock_cnt == 0:
            now_money, now_stock_cnt = self.buy_stock(now_money, now_close, now_stock_cnt)
            if self.params.debug is True:
                print("bought", ratio, now_money, now_stock_cnt, now_close)

        elif ratio < -invest_min_percent and now_stock_cnt > 0:
            now_money, now_stock_cnt = self.sell_stock(now_money, now_close, now_stock_cnt)
            if self.params.debug is True:
                print("sold", ratio, now_money, now_stock_cnt, now_close)
        return now_money, now_stock_cnt

    def get_high_n_low(self, x, scaler_close):
        x_last = x[:, -1][0]
        x_high = DataUtils.inverse_scaled_data(scaler_close, x_last[2])
        x_low = DataUtils.inverse_scaled_data(scaler_close, x_last[3])
        return x_high, x_low

    def invest_scaled_money_before(self, before_scaled_close, before_scaled_predict, x, now_money, now_stock_cnt,
                                   scaler_close):
        x_high, x_low = self.get_high_n_low(x, scaler_close)

        before_invest_predict = DataUtils.inverse_scaled_data(scaler_close, before_scaled_predict)
        before_close = DataUtils.inverse_scaled_data(scaler_close, before_scaled_close)
        return self.invest_money_before2(before_close, before_invest_predict, x_high, x_low, now_money, now_stock_cnt)

    def invest_money_before(self, before_close, before_invest_predict, x_high, x_low, now_money, now_stock_cnt):
        invest_min_percent = self.params.invest_min_percent
        invest_max_percent = self.params.invest_max_percent

        ratio = self.get_invest_ratio(before_invest_predict, before_close)

        trade_price = before_close + (before_invest_predict - before_close) * (invest_max_percent / 100)
        if ratio > invest_min_percent and x_high > trade_price and now_stock_cnt > 0:
            now_money, now_stock_cnt = self.sell_stock(now_money, trade_price, now_stock_cnt)
            if self.params.debug is True:
                print("before_sold", ratio, now_money, now_stock_cnt, trade_price)

        elif ratio < -invest_min_percent and x_low < trade_price  and now_stock_cnt == 0:
            now_money, now_stock_cnt = self.buy_stock(now_money, trade_price, now_stock_cnt)
            if self.params.debug is True:
                print("before_bought", ratio, now_money, now_stock_cnt, trade_price)

        return now_money, now_stock_cnt

    def invest_money_before2(self, before_close, before_invest_predict, x_high, x_low, now_money, now_stock_cnt):
        """방향은 그대로 사용하고 거는 값의 최소값을 사용한다."""
        trade_min_percent = self.params.trade_min_percent
        invest_max_percent = self.params.invest_max_percent

        ratio = self.get_invest_ratio(before_invest_predict, before_close)

        trade_price = before_close + (before_invest_predict - before_close) * (invest_max_percent / 100)
        if ratio > 0 and now_stock_cnt > 0:
            trade_min_price = before_close * (1.0+trade_min_percent/100)
            if trade_price < trade_min_price:
                trade_price = trade_min_price
            if x_high > trade_price:
                now_money, now_stock_cnt = self.sell_stock(now_money, trade_price, now_stock_cnt)
                if self.params.debug is True:
                    print("before_sold", ratio, now_money, now_stock_cnt, trade_price)
        elif ratio <= 0 and now_stock_cnt == 0:
            trade_max_price = before_close * (1.0-trade_min_percent/100)
            if trade_price > trade_max_price:
                trade_price = trade_max_price
            if x_low < trade_price:
                now_money, now_stock_cnt = self.buy_stock(now_money, trade_price, now_stock_cnt)
                if self.params.debug is True:
                    print("before_bought", ratio, now_money, now_stock_cnt, trade_price)
        return now_money, now_stock_cnt

    def get_invest_ratio(self, invest_predict, now_close):
        """ 예측에 대한 비율을 구한다. """
        return (invest_predict - now_close) / now_close * 100

    def buy_stock(self, now_money, now_close, now_stock_cnt):
        """주식을 산다."""
        fee_percent = self.params.fee_percent
        fee = now_close * fee_percent / 100
        cnt = math.floor(now_money / (now_close + fee))
        if cnt > 0:
            now_money -= (now_close + fee) * cnt
            now_stock_cnt += cnt
        return now_money, now_stock_cnt


    def sell_stock(self, now_money, now_close, now_stock_cnt):
        """주식을 판다."""
        if now_stock_cnt > 0:
            now_money += self.to_money(now_close, now_stock_cnt)
            now_stock_cnt = 0
        return now_money, now_stock_cnt

    def to_money(self, now_stock_cnt, now_close):
        """주식매도를 해서 돈으로 바꾼다."""
        money = 0
        if now_stock_cnt > 0:
            fee_percent = self.params.fee_percent
            tax_percent = self.params.tax_percent

            fee = now_close * fee_percent / 100
            tax = now_close * tax_percent / 100
            money = (now_close - (fee + tax)) * now_stock_cnt
        return money

    def get_real_money(self, data_params, scaler_close, last_predict):
        """실제 가격을 가져온다."""

        invest_max_percent = self.params.invest_max_percent
        trade_min_percent = self.params.trade_min_percent

        close_scaled_money = data_params.testY[-1][0]
        close_money = DataUtils.inverse_scaled_data(scaler_close, close_scaled_money)
        predict_money = DataUtils.inverse_scaled_data(scaler_close, last_predict)
        if predict_money > close_money:
            trade_min_money = close_money * (1+trade_min_percent/100)
            trade_money = close_money + (predict_money - close_money) * (invest_max_percent/100)
            if trade_money < trade_min_money:
                trade_money = trade_min_money
        else:
            trade_max_money = close_money * (1 - trade_min_percent / 100)
            trade_money = close_money - (close_money - predict_money) * (invest_max_percent / 100)
            if trade_money > trade_max_money:
                trade_money = trade_max_money
        return close_money, predict_money, trade_money

    def invest(self, comp_code:str, corp_name:str, tps:dict, invest_only:bool=False)->InvestParams:
        """학습 후 모의 주식 거래를 한다."""

        ip = InvestParams()
        ip.predict_list, ip.last_predict, ip.test_predict = self.ensemble_predicts(comp_code, tps, invest_only)
        #print(ip.predict_list, ip.last_predict, ip.test_predict)

        tp = next(iter(tps.values()))
        investX = tp.data_params.investX
        
        #investY = data_params.investY
        invest_count = len(investX)
        ip.invest_money = self.params.invest_money
        #print('mockInvestment', 'invest', ip.invest_money)

        invest_line_trading = self.params.invest_line_trading
        now_stock_cnt = 0
        if self.params.index_money is not None:
            ip.index_money = self.params.index_money
        else:
            ip.index_money = ip.invest_money
        all_stock_count = now_stock_cnt

        invest_list = []
        index_invest_list = []

        before_scaled_close = None
        before_invest_predict = None
        now_scaled_close = None

        for i in range(invest_count):
            x = investX[i:i + 1]
            invest_predict = ip.predict_list[i][0]
            x_last = x[:, -1][0]
            now_scaled_close = x_last[0]
            if i != 0 and invest_line_trading == True:
                ip.invest_money, now_stock_cnt = self.invest_scaled_money_before(before_scaled_close, before_invest_predict,
                                                                              x, ip.invest_money, now_stock_cnt, tp.scaler_close)

            ip.invest_money, now_stock_cnt = self.invest_scaled_money(invest_predict, now_scaled_close, tp.scaler_close,
                                                                   ip.invest_money, now_stock_cnt)
            if i == 0:
                ip.index_money, all_stock_count = self.invest_scaled_money(-1, now_scaled_close, tp.scaler_close,
                                                                             ip.index_money, all_stock_count)
                index_invest_list.append([now_scaled_close, ip.index_money, all_stock_count])
                #print(ip.index_money, all_stock_count, scaler_close.inverse_transform(now_scaled_close)[0][0])
            before_scaled_close = now_scaled_close
            before_invest_predict = invest_predict
            invest_list.append([now_scaled_close, ip.invest_money, now_stock_cnt])

        if now_scaled_close is not None:
            now_close = DataUtils.inverse_scaled_data(tp.scaler_close, now_scaled_close)
            ip.invest_money += self.to_money(now_stock_cnt, now_close)
            ip.last_money = ip.invest_money
            ip.index_money += self.to_money(all_stock_count, now_close)
            invest_list.append([now_scaled_close, ip.invest_money, 0])
            index_invest_list.append([now_scaled_close, ip.index_money, 0])

            try:
                visualizer = InvestVisualizer(self.params)
                ip.daily_data = visualizer.draw_invests(corp_name, invest_list, index_invest_list, tp.scaler_close, tp.data_params)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        else:
            ip.last_money = ip.invest_money

        return ip

    def get_predicts(self, comp_code, tp):
        learning = Learning(self.params)
        investX = tp.data_params.investX
        #print(investX)

        graph_params = learning.get_train_model()
        X = graph_params.X
        Y_pred = graph_params.Y_pred
        training = graph_params.training
        output_keep_prob = graph_params.output_keep_prob

        saver = tf.train.Saver()
        session_path = learning.get_session_path(comp_code)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, session_path)

            invest_predicts = None
            if len(investX) > 0:
                invest_predicts = sess.run(Y_pred, feed_dict={X: investX, output_keep_prob: 1.0, training: False})
            last_predict = sess.run(Y_pred, feed_dict={X: tp.dataX_last, output_keep_prob: 1.0, training: False})
        return invest_predicts, last_predict[0]

    def ensemble_predicts(self, corp_code:str, tps:dict, invest_only:bool=False):
        if len(tps) == 1:
            tp = next(iter(tps.values()))
            invest_predicts, last_predict =  self.get_predicts(corp_code, tp)
            return invest_predicts, last_predict, None
        else:

            if self.params.ensemble_type == 'rmse_best':
                return self.ensemble_predicts_best(corp_code, tps)
            elif self.params.ensemble_type == 'linear_regression':
                return self.ensemble_predicts_linear_regression(corp_code, tps)

            total_rmse, tp_cnt = self.get_total_rmse(tps)
            model_type = self.params.model_type
            back_train_model = self.params.train_model

            ens_predicts = None
            ens_last_predict = None
            ens_test_predict = None
            for train_model, tp in tps.items():
                self.params.__init__(model_type, train_model=train_model)
                invest_predicts, last_predict = self.get_predicts(corp_code, tp)
                if invest_predicts is not None:
                    ens_predicts = self.get_ensemble_value(total_rmse, tp.test_rmse, invest_predicts, ens_predicts, tp_cnt)
                ens_last_predict = self.get_ensemble_value(total_rmse, tp.test_rmse, last_predict, ens_last_predict, tp_cnt)
                if invest_only == False:
                    ens_test_predict = self.get_ensemble_value(total_rmse, tp.test_rmse, tp.test_predict, ens_test_predict, tp_cnt)

            self.params.__init__(model_type, train_model=back_train_model)
            return ens_predicts, ens_last_predict, ens_test_predict

    def ensemble_predicts_best(self, corp_code, tps):
        model_type = self.params.model_type
        back_train_model = self.params.train_model
        good_test_rmse = None
        good_train_model = None
        ens_test_predict = None
        tp_best = None
        for train_model, tp in tps.items():
            if good_test_rmse is None or tp.test_rmse < good_test_rmse:
                good_test_rmse = tp.test_rmse
                good_train_model = train_model
                ens_test_predict = tp.test_predict
                tp_best = tp

        self.params.__init__(model_type, train_model=good_train_model)
        ens_predicts, ens_last_predict = self.get_predicts(corp_code, tp_best)
        self.params.__init__(model_type, train_model=back_train_model)
        return ens_predicts, ens_last_predict, ens_test_predict


    def ensemble_predicts_linear_regression(self, corp_code:str, tps:dict):

        min_len = 10**10
        for train_model, tp in tps.items():
            test_predict = tp.test_predict
            if  len(test_predict) < min_len:
                min_len = len(test_predict)

        model_type = self.params.model_type
        back_train_model = self.params.train_model
        testX = None
        testY = None
        investX = None
        lastX = None
        for train_model, tp in tps.items():
            test_predict = tp.test_predict
            if len(test_predict) > min_len:
                test_predict = test_predict[len(test_predict)-min_len:]
            elif testY is None:
                testY = tp.data_params.testY

            self.params.__init__(model_type, train_model=train_model)
            invest_predicts, last_predict = self.get_predicts(corp_code, tp)

            last_predict = np.asarray([last_predict])
            if testX is None:
                testX = test_predict
                investX = invest_predicts
                lastX = last_predict
            else:
                testX = np.append(testX, test_predict, axis=1)
                investX = np.append(investX, invest_predicts, axis=1)
                #print(last_predict)
                lastX = np.append(lastX, last_predict, axis=1)

        self.params.__init__(model_type, train_model=back_train_model)

        reg = LinearRegression()
        reg.fit(testX, testY)
        #print(reg.score(testX, testY))
        #print(reg.coef_)
        ens_predicts = reg.predict(investX)
        ens_test_predict = reg.predict(testX)
        ens_last_predict = reg.predict(lastX)
        return ens_predicts, ens_last_predict, ens_test_predict


    def get_total_rmse(self, tps:dict):
        total_rmse = 0
        tp_cnt = 0
        ens_type = self.params.ensemble_type
        for train_model, tp in tps.items():
            if ens_type == 'rmse_square_ratio':
                total_rmse += 1/tp.test_rmse**2
            else:
                total_rmse += 1/tp.test_rmse
            tp_cnt += 1
        return total_rmse, tp_cnt


    def get_ensemble_value(self, total_rmse:np.float, rmse:np.float, value:np.ndarray,
                           prev_value:np.ndarray=None, tp_cnt:int=1)->np.ndarray:
        if prev_value is None:
            value = self.get_ensemble_rule_value(total_rmse, rmse, value, tp_cnt)
        else:
            now_size = len(value)
            prev_size = len(prev_value)
            if now_size > prev_size:
                value = value[now_size-prev_size:]
            elif now_size < prev_size:
                prev_value = prev_value[prev_size-now_size:]

            value =  self.get_ensemble_rule_value(total_rmse, rmse, value, tp_cnt) + prev_value
        return value

    def get_ensemble_rule_value(self, total_rmse:np.float, rmse:np.float,
                                value:np.ndarray, tp_cnt:int) -> np.ndarray:
        ens_type = self.params.ensemble_type
        if ens_type == 'rmse_square_ratio':
            return value * (1/rmse**2) * (1/ total_rmse)
        elif ens_type == 'rmse_ratio':
            return value * (1/rmse) * (1/total_rmse)
        elif ens_type == 'avg':
            return value/tp_cnt
        else:
            return value

    def invest_plain(self, investX, fitted_model):
        """학습 후 모의 주식 거래를 한다."""
        invest_count = self.params.invest_count
        invest_money = self.params.invest_money

        predicts = fitted_model.get_test_rmse(investX)
        before_close = None
        before_invest_predict = None
        now_close = None
        now_stock_cnt = 0
        all_invest_money = invest_money
        all_stock_count = 0
        for i in range(invest_count):
            x = investX[i:i + 1]
            #print(x)
            now_close = x[0][0]
            x_high = x[0][2]
            x_low = x[0][3]
            invest_predict = predicts[i]
            if i != 0:
                invest_money, now_stock_cnt = self.invest_money_before(before_close, before_invest_predict,
                                                                       x_high, x_low, invest_money, now_stock_cnt)

            invest_money, now_stock_cnt = self.invest_money(invest_predict, now_close, invest_money, now_stock_cnt)
            if i == 0:
                all_invest_money, all_stock_count = self.invest_money(-1, now_close, all_invest_money, all_stock_count)
            before_close = now_close
            before_invest_predict = invest_predict

        if now_close != None:
            invest_money += self.to_money(now_stock_cnt, now_close)
            all_invest_money += self.to_money(all_stock_count, now_close)

        return invest_money, all_invest_money

    def invest_n_all(self, comp_code, dataX_last, data_params, params_all, scaler_close=None):
        """학습 후 모의 주식 거래를 한다."""

        ip = InvestParams()
        investX = data_params.investX
        invest_count = len(investX)
        ip.invest_money = self.params.invest_money

        now_stock_cnt = 0
        ip.index_money = ip.invest_money
        all_stock_count = 0
        before_scaled_close = None
        before_invest_predict = None
        now_scaled_close = None
        for i in range(invest_count):

            x = investX[i:i + 1]
            invest_predict = self._get_predict(self.params, comp_code, x)
            invest_predict_all = self._get_predict(params_all, comp_code, x)

            invest_predict = np.mean([invest_predict, invest_predict_all])
            ip.predict_list.append([invest_predict])
            x_last = x[:, -1][0]
            now_scaled_close = x_last[0]
            if i != 0:
                ip.invest_money, now_stock_cnt = self.invest_scaled_money_before(before_scaled_close, before_invest_predict, x,
                                                                       ip.invest_money, now_stock_cnt, scaler_close)

            ip.invest_money, now_stock_cnt = self.invest_scaled_money(invest_predict, now_scaled_close, scaler_close,
                                                            ip.invest_money, now_stock_cnt)
            if i == 0:
                ip.index_money, all_stock_count = self.invest_scaled_money(-1, now_scaled_close, scaler_close,
                                                                      ip.index_money, all_stock_count)
            before_scaled_close = now_scaled_close
            before_invest_predict = invest_predict

        if now_scaled_close != None:
            now_close = DataUtils.inverse_scaled_data(scaler_close, now_scaled_close)
            ip.invest_money += self.to_money(now_stock_cnt, now_close)
            ip.index_money += self.to_money(all_stock_count, now_close)

        # 마지막 예측 값을 구한다.
        ip.last_predict = self._get_predict(self.params, comp_code, dataX_last)
        last_predict_all = self._get_predict(params_all, comp_code, dataX_last)
        ip.last_predict = np.mean([ip.last_predict, last_predict_all])

        return ip


    def _get_predict(self, params, comp_code, investX):
        """종가 예측 값을 가져온다."""
        stacked_rnn = StackedRnn(params)
        graph_params = stacked_rnn.get_model()
        Y_pred = graph_params.Y_pred
        output_keep_prob = graph_params.output_keep_prob
        X = graph_params.X
        learning = Learning(params)
        file_path = learning.get_session_path(comp_code)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, file_path)
            last_predict = sess.run(Y_pred, feed_dict={X: investX, output_keep_prob: 1.0})
        return last_predict


    def invest_4_rule(self, stock_data, invest_count):
        """정해진 규칙으로 주식을 매매한다."""
        stock_len = len(stock_data)
        invest_data = stock_data[stock_len-invest_count:]

        before_close = None
        invest_money = self.params.invest_money
        now_stock_cnt = 0
        for i in range(invest_count):
            now_data = invest_data[i:i+1]
            close_money = now_data['close'].values[0]
            if i != 0:
                invest_money, now_stock_cnt = self.invest_money_4_rule(now_data, before_close, invest_money, now_stock_cnt)
            before_close = close_money
        invest_money += self.to_money(now_stock_cnt, before_close)
        return invest_money


    def invest_money_4_rule(self, now_data, before_close, now_money, now_stock_cnt):
        now_high = now_data['high'].values[0]
        now_low = now_data['low'].values[0]
        rule_trade_percent = self.params.rule_trade_percent
        max = before_close * (1.0 + rule_trade_percent/100)
        min = before_close * (1.0 - rule_trade_percent/100)

        if now_stock_cnt > 0 and now_high > max :
            now_money, now_stock_cnt = self.sell_stock(now_money, max, now_stock_cnt)
        elif now_stock_cnt == 0 and now_low < min:
            now_money, now_stock_cnt = self.buy_stock(now_money, min, now_stock_cnt)

        return now_money, now_stock_cnt