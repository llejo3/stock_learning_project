from data.corp import Corp
from data.data_utils import DataUtils
from trains.learning_plain_models import LearningPlainModels
from trains.learning_n_mock_investment import LearningNMockInvestment
from params.global_params import GlobalParams
from trains.learning_n_mock_top10 import LearningNMockTop10
from utils.date_utils import DateUtils
import pandas as pd
import os
import datetime

from visualization.invest_visualizer import InvestVisualizer


def get_corps(type='EACH'):
    params = GlobalParams(type)
    corp = Corp(params)
    return corp.get_eval_corps_auto()


def train(type:str='EACH', start_no:int=1, train_model:str='rnn', invest_only:bool=False) -> None:
    """하나의 세션으로 학습시키는 기본 모델 """
    corps = get_corps(type)

    params = GlobalParams(type, train_model=train_model)
    invests = LearningNMockInvestment(params)
    invests.train_n_invests(corps, start_no, invest_only)


def train_months(start:str='2018.01', end:str='2018.09', invest_money:float=100000000,
                 train_model:str='rnn') -> None:
    """하나의 세션으로 학습시키는 기본 모델 """
    start_month = DateUtils.to_date(start, '%Y.%m')
    end_month = DateUtils.to_date(end, '%Y.%m')
    between = DateUtils.between_months(start_month, end_month)
    invest_months_result = []
    result_columns = ["month", "invest_money", "result_money"]
    MOCK_MONEY = 10000000
    chart_data = []
    params = None
    index_money = None
    for i in range(between +1):

        params = GlobalParams(train_model=train_model)
        #params.remove_session_file = True
        before_month_start = DateUtils.to_month_str(start_month, i - params.mock_period_months)
        before_month_end = DateUtils.to_month_str(start_month, i - 1)
        params.invest_start_date = before_month_start + '.01'
        params.invest_end_date = before_month_end + '.31'
        params.result_file_name = "MOCK_" + before_month_start + "-" + before_month_end
        params.invest_money = MOCK_MONEY
        corp = Corp(params)
        corps = corp.get_eval_corps_auto(params.invest_end_date)
        invests = LearningNMockInvestment(params)
        invests.train_n_invests(corps)
        before_result = pd.read_csv(invests.get_result_file_path())

        now_month = DateUtils.to_month_str(start_month, i)
        if params.rmse_max_recommend is not None:
            before_result = before_result.query("rmse<" + str(params.rmse_max_recommend))
        before_result = corp.exclude_corps(before_result, now_month)
        before_result = before_result.sort_values(by='invest_result', ascending=False)
        before_result.index = range(len(before_result.index))
        corp10_codes = before_result.loc[:9,'code']
        corp10_codes.index = range(len(corp10_codes.index))
        corp10 = corp.get_corps_for_codes(corp10_codes)
        corp10_len = len(corp10_codes.index)

        params = GlobalParams(train_model=train_model)
        #params.remove_session_file = False

        params.invest_start_date = now_month + '.01'
        params.invest_end_date = now_month + '.31'
        params.result_file_name = "INVEST_" + now_month
        params.invest_money = invest_money/corp10_len
        if index_money is not None:
            params.index_money = index_money/corp10_len
        invests = LearningNMockInvestment(params)
        invest_chart_data = invests.train_n_invests(corp10, invest_only=False)
        chart_data.append(invest_chart_data)
        now_result = pd.read_csv(invests.get_result_file_path())
        invest_money = now_result['invest_result'].sum()
        index_money = now_result['all_invest_result'].sum()
        invest_months_result.append([now_month, params.invest_money*corp10_len, invest_money])
        print(now_month, params.invest_money*corp10_len, invest_money)

    df_imr = pd.DataFrame(invest_months_result, columns=result_columns)
    save_file_name = "recommend_months_" + start + "-" + end + ".xlsx"
    if "_" in train_model:
        save_file_path = os.path.join('result', train_model, params.ensemble_type, save_file_name)
    else:
        save_file_path = os.path.join('result', train_model, save_file_name)
    DataUtils.save_csv(df_imr, save_file_path)

    if len(chart_data) > 1 and params is not None:
        visualizer = InvestVisualizer(params)
        visualizer.draw_invest_months(chart_data, start, end)
        print()


def recommend_corps(recommend_month:str, train_model:str='rnn') -> None:
    """하나의 세션으로 학습시키는 기본 모델 """

    month = DateUtils.to_date(recommend_month, '%Y.%m')
    params = GlobalParams(train_model=train_model)
    #params.remove_session_file = True
    before_month_start = DateUtils.to_month_str(month, -params.mock_period_months)
    before_month_end = DateUtils.to_month_str(month, -1)
    params.invest_start_date = before_month_start + '.01'
    params.invest_end_date = DateUtils.to_date_str(month - datetime.timedelta (days = 1))
    params.result_file_name = "MOCK_" + before_month_start + "-" + before_month_end
    corp = Corp(params)
    corps = corp.get_eval_corps_auto(params.invest_end_date)
    invests = LearningNMockInvestment(params)
    invests.train_n_invests(corps)
    before_result = pd.read_csv(invests.get_result_file_path())

    if params.rmse_max_recommend is not None:
        before_result = before_result.query("rmse<" + str(params.rmse_max_recommend))
    before_result = before_result.sort_values(by='invest_result', ascending=False)
    before_result.index = range(len(before_result.index))
    save_file_name = "recommend_months_" + recommend_month + ".xlsx"
    save_file_path = os.path.join('result', train_model, save_file_name)
    DataUtils.save_csv(before_result, save_file_path)
    print(before_result)


def train_one(corp_name='카카오', train_model='rnn'):
    """하나의 세션으로 학습시키는 기본 모델 """
    corp = Corp(type)
    corps = corp.get_corp(corp_name)

    params = GlobalParams(train_model=train_model)
    invests = LearningNMockInvestment(params)
    invests.train_n_invests(corps)

def train_names(corp_names, train_model='rnn'):
    """하나의 세션으로 학습시키는 기본 모델 """
    corp = Corp(type)
    corps = corp.get_corps_for_names(corp_names)

    params = GlobalParams(train_model=train_model)
    invests = LearningNMockInvestment(params)
    invests.train_n_invests(corps)

def train_codes(corp_codes, train_model='rnn'):
    """하나의 세션으로 학습시키는 기본 모델 """
    corp = Corp(type)
    corps = corp.get_corps_for_codes(corp_codes)

    params = GlobalParams(train_model=train_model)
    invests = LearningNMockInvestment(params)
    invests.train_n_invests(corps)

def train_all_corps(type='ALL_CORPS', start_no=1):
    """하나의 세션으로 모든 회사를 학습시킨다.  """
    corp = Corp(type)
    corps = corp.get_corps()

    params = GlobalParams(type)
    params.result_file_name = "training_" + type.lower() + "_result"
    invests = LearningNMockInvestment(params)
    #invests.train_n_invests(corps, start_no)
    invests.train_n_invests_for_all(corps)


def top10_model(train_model='rnn'):
    """상위10개를 가지고 투자하는 모델"""
    corp = Corp()
    corps = corp.get_eval_corps()

    params = GlobalParams(train_model=train_model)
    params.invest_type = 'top10'
    params.result_file_name = "top10_result"
    invests = LearningNMockTop10(params)
    invests.train_n_invests_top10(corps)


def train_plain_model(model_type= 'decision_tree'):
    """기초적인 통계를 이용한 모델"""
    #corp = Corp()
    #corps = corp.get_corps_for_names(["카카오"])
    corps = get_corps()

    #decision_tree, random_forest, linear_regression, light_gbm
    invests = LearningPlainModels(model_type)
    invests.trains(corps)


if __name__ == '__main__':
    train()
    #train_names(["카카오"], 'cnn_mlp_rnn')
    #train_plain_model('light_gbm')
    #train_names(["신한"])
    #month = DateUtils.to_date('2018.12', '%Y.%m')
    #before = month - datetime.timedelta (days = 1)
    #print(before)_
