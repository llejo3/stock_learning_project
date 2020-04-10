from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

from data.corp import Corp
from data.data_utils import DataUtils
from data.stocks import Stocks
from data.trains_data import TrainsData
from params.plain_model_params import PlainModelParams
from trains.mock_investment import MockInvestment
import pandas as pd

from visualization.plain_visualizer import PlainVisualizer


class LearningPlainModels:
    """학습시키고 모의투자를 실행한다."""

    # 학습 결과의 컬럼명 정의
    RESULT_COLUMNS = ['no', 'code', 'name', 'invest_result', 'all_invest_result', 'rmse']

    def __init__(self, type):
        self.params = PlainModelParams(type)
        self.type = type
        if self.params.result_file_name == None:
            self.params.result_file_name = type + "_result"

    def trains(self, corps):
        """ 선형회귀 모형 학습  """
        comp_rmses = []
        no = 1
        for index, corp_data in corps.iterrows():
            corp_code = corp_data['종목코드']
            corp_name = corp_data['회사명']
            try :
                result = self.train(corp_code, corp_name, no)
            except Exception as inst:
                print(inst)
                no += 1
                continue

            comp_rmses.append(result)
            no += 1
        df_comp_rmses = pd.DataFrame(comp_rmses, columns=self.RESULT_COLUMNS)
        DataUtils.save_excel(df_comp_rmses, self.get_result_file_path())

    def get_result_file_path(self):
        """결과를 저장할 경로"""
        return './result/' + self.params.result_file_name + '.xlsx'

    def train(self, corp_code, corp_name, no):
        """입력한 회사에 대해서 학습시키고 모의투자를 실행한다."""

        stocks = Stocks(self.params)
        trains_data = TrainsData(self.params)
        stock_data = stocks.get_stock_data(corp_code)
        dts = trains_data.get_train_test_plain(stock_data)

        # if self.type == 'light_gbm':
        #     self.train_light_gbm(dts)
        # else:
        clf = self.get_model()
        clf.fit(dts.trainX, dts.trainY)
        if len(dts.testX) > 0:
            test_pred = clf.predict(dts.testX)
            rmse = metrics.mean_squared_error(dts.testY, test_pred)
        else:
            rmse = -1

        invest = MockInvestment(self.params)
        invest_money, all_invest_money = invest.invest_plain(dts.investX, clf)
        print(no, corp_code, corp_name, invest_money, all_invest_money, rmse)

        visualizer = PlainVisualizer(self.params)
        visualizer.draw_chart(clf, corp_name, dts)
        return [no, corp_code, corp_name, invest_money, all_invest_money, rmse]

    def get_model(self):
        clf = None
        if self.type == 'decision_tree':
            clf = DecisionTreeRegressor()
        elif self.type == 'random_forest':
            clf = RandomForestRegressor(n_estimators=10)
        elif self.type == 'linear_regression':
            clf = LinearRegression()
        elif self.type == 'light_gbm':
            clf = LGBMRegressor()
        return clf

    def train_light_gbm(self, dts):
        # create dataset for lightgbm
        lgb_train = lgb.Dataset(dts.trainX, dts.trainY)
        lgb_test = lgb.Dataset(dts.testX, dts.testY, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'num_leaves': 5,
            'metric': ('l1', 'l2'),
            'verbose': 0
        }

        evals_result = {}  # to record eval results for plotting

        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=[lgb_train, lgb_test],
                        feature_name=['close', 'open', 'high', 'low', 'volume'],
                        categorical_feature=[21],
                        evals_result=evals_result,
                        verbose_eval=10)

        print('Plotting metrics recorded during training...')
        ax = lgb.plot_metric(evals_result, metric='l1')
        plt.show()

        print('Plotting feature importances...')
        ax = lgb.plot_importance(gbm, max_num_features=10)
        plt.show()

        print('Plotting 84th tree...')  # one tree use categorical feature to split
        ax = lgb.plot_tree(gbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
        plt.show()

        print('Plotting 84th tree with graphviz...')
        graph = lgb.create_tree_digraph(gbm, tree_index=83, name='Tree84')
        graph.render(view=True)

if __name__ == '__main__':
    corp_name = "카카오"
    corp = Corp()
    corp_code = corp.get_corp_code(corp_name)
    learning = LearningPlainModels('light_gbm')
    learning.train(corp_code, corp_name, 1)

    # from sklearn.datasets import load_iris
    #
    # iris = load_iris()
    # X = iris.data[:, [2, 3]]
    # y = iris.target
    # print(X, y)