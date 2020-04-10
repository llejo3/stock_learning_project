import os
import sys, traceback
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from data.data_utils import DataUtils
from params.global_params import GlobalParams
from utils.date_utils import DateUtils
from visualization.main_visualizer import MainVisualizer
from data.corp import Corp


class InvestVisualizer:
    """투자관련 그래프를 그린다."""

    def __init__(self, params:GlobalParams):
        self.params = params

        self.main = MainVisualizer(params)


    def get_file_path(self, dir:str, corp_name:str, extension:str, start:str=None, end:str=None)->str:
        """저장할 파일 경로 """
        if start is None:
            invest_start_date = self.params.invest_start_date
        else:
            invest_start_date = start

        if end is None:
            invest_end_date = self.params.invest_end_date
        else:
            invest_end_date = end

        if dir == 'invests_total':
            dir = os.path.join(self.main.DIR_CHARTS, dir)
        else:
            dir = os.path.join(self.main.DIR_CHARTS, dir, corp_name)

        DataUtils.create_dir(dir)
        corp_name = corp_name.replace(" ", "")
        return os.path.join(dir, corp_name + "_" + invest_start_date + "-" + invest_end_date  + "." + extension)

    def get_kospi_kosdaq(self, market):
        invest_start_date = self.params.invest_start_date
        invest_end_date = self.params.invest_end_date
        corp = Corp(self.params)
        if market == 'KOSPI':
            data = corp.get_kospi()
        elif market == 'KOSDAQ':
            data = corp.get_kosdaq()
        else:
            data = pd.DataFrame()
        return data.query("date>='{}' & date<='{}'".format(invest_start_date, invest_end_date))

    def get_value(self, data, row_index, column):
        return data.iloc[row_index:row_index+1][column].values[0]


    def get_date_search_value(self, data, date, column):
        value = data.query("date=='{}'".format(date))[column].values
        if len(value) > 0:
            return value[0]
        else:
            return 0


    def draw_invest_daily(self, invest_daily_data:list, corps:pd.DataFrame)->pd.DataFrame:
        """여러종목을 투자한 결과를 그래프로 그린다."""
        corp_cnt = len(corps)
        corp_name = corps[0:1]['회사명'].values[0] + " 외 " + str(corp_cnt-1) + "종목"
        dir = 'invests_total'

        # 데이터를 더하고 합친다.
        daily_data = self.join_daily_data(invest_daily_data)

        # 결과 데이터르 만든다.
        result_data = self.get_invest_chart_data(daily_data)
        self.save_csv(result_data, dir, corp_name)
        self.draw_invest_seaborn(result_data, dir, corp_name)
        return daily_data


    def draw_invest_4reinforcement(self, invest_daily_data:list, corps:pd.DataFrame)->pd.DataFrame:
        """여러종목을 투자한 결과를 그래프로 그린다."""
        corp_cnt = len(corps)
        corp_name = corps[0:1]['회사명'].values[0] + " 외 " + str(corp_cnt-1) + "종목"
        dir = 'invests_total'

        # 데이터를 더하고 합친다.
        chart_daily_data = self.to_chart_daily_data(invest_daily_data)
        daily_data = self.join_daily_data(chart_daily_data)
        #print(daily_data.head())

        # 결과 데이터르 만든다.
        result_data = self.get_invest_chart_data(daily_data)
        self.save_csv(result_data, dir, corp_name)
        self.draw_invest_seaborn(result_data, dir, corp_name)
        return daily_data

    def to_chart_daily_data(self, invest_daily_data)->list:
        kospi = self.get_kospi_kosdaq('KOSPI')
        kosdaq = self.get_kospi_kosdaq('KOSDAQ')
        data_list = []
        for dailies in invest_daily_data:
            data = []
            for daily in dailies:
                kospi_amt = self.get_date_search_value(kospi, daily.date, 'close')
                if kospi_amt == 0:
                    continue
                kosdaq_amt = self.get_date_search_value(kosdaq, daily.date, 'close')
                if kosdaq_amt == 0:
                    continue
                data.append([daily.date, daily.value, daily.index_value, kospi_amt, kosdaq_amt])
            data_list.append(data)
        return data_list


    def get_invest_chart_data(self, daily_data:pd.DataFrame)->pd.DataFrame:
        date = daily_data.loc[0, 0]
        start_money = daily_data.loc[0, 1]
        start_index_money = daily_data.loc[0, 2]
        start_cospi = daily_data.loc[0, 3]
        start_cosdaq = daily_data.loc[0, 4]
        invest_chart_data = [[date, 0, 0, 0, 0]]
        for i in range(1, len(daily_data.index)):
            date = daily_data.loc[i, 0]
            money = daily_data.loc[i, 1]
            index_money = daily_data.loc[i, 2]
            cospi = daily_data.loc[i, 3]
            cosdaq = daily_data.loc[i, 4]
            money_percent = self.get_ratio(money, start_money)
            money_index_percent = self.get_ratio(index_money, start_index_money)
            kospi_percent = self.get_ratio(cospi, start_cospi)
            kosdaq_percent = self.get_ratio(cosdaq, start_cosdaq)
            invest_chart_data.append([date, money_percent, money_index_percent, kospi_percent, kosdaq_percent])

        return pd.DataFrame(invest_chart_data, columns=['date', 'invest', 'index', 'KOSPI', 'KOSDAQ'])


    def join_daily_data(self, invest_daily_data)->pd.DataFrame:
        daily_cnt = len(invest_daily_data)
        merged_data = pd.DataFrame(invest_daily_data[0])
        columns = merged_data.columns.values
        for i in range(1, daily_cnt):
            df_next = pd.DataFrame(invest_daily_data[i])
            merged_data = pd.merge(merged_data, df_next, how='outer', on=[0]).sort_values([0])
            merged_data.index = range(0, len(merged_data.index))
            self.fills_nan(merged_data)
            merged_data = self.sum_columns(columns, merged_data)
        return merged_data

    def fills_nan(self, df):
        columns = df.columns.values
        before_values = {}
        for i in range(len(df.index)):
            for column in columns:
                if df.dtypes[column] == 'object':
                    continue
                value = df.loc[i, column]
                if np.isnan(value):
                    before_value = 0
                    if column in before_values:
                        before_value = before_values[column]
                    else:
                        for j in range(1, len(df.index) - i):
                            before_value = df.loc[i + j, column]
                            if not np.isnan(before_value):
                                break
                    value = before_value
                    df.loc[i, column] = before_value
                before_values[column] = value

    def sum_columns(self, columns, df):
        for i in range(1, len(columns)):
            df[i] = df[str(i) + '_x'] + df[str(i) + '_y']
        return df[columns]


    def draw_invests(self, corp_name, invest_data, all_invest_data, scaler_close, data_params)->list:
        """모의 투자 결과 그래프를 그린다."""
        dir = 'invests'
        kospi = self.get_kospi_kosdaq('KOSPI')
        kosdaq = self.get_kospi_kosdaq('KOSDAQ')

        start_money = self.params.invest_money
        if self.params.index_money is not None:
            start_index_money = self.params.index_money
        else:
            start_index_money = start_money
        start_kospi = self.get_value(kospi, 0, 'close')
        start_kosdaq = self.get_value(kosdaq, 0, 'close')
        investY_date = data_params.investY_date
        invest_index = all_invest_data[0]
        money_index = invest_index[1]
        stock_cnt_index = invest_index[2]
        date = investY_date[0:1].values[0]
        date = DateUtils.add_days(date, -1, '%Y.%m.%d')
        invest_chart_data = [[date, 0, 0, 0, 0]]
        invest_daily = [[date, start_money, start_index_money, start_kospi, start_kosdaq]]
        for i in range(len(invest_data)):
            try:
                date = investY_date[i:i+1].values[0]
                invest = invest_data[i]
                close_scaled = invest[0]
                money = invest[1]
                stock_cnt = invest[2]
                kospi_amt = self.get_date_search_value(kospi, date, 'close')
                if kospi_amt == 0: continue
                kospi_percent = self.get_ratio(kospi_amt,start_kospi)
                kosdaq_amt = self.get_date_search_value(kosdaq, date, 'close')
                if kosdaq_amt == 0: continue
                kosdaq_percent = self.get_ratio(kosdaq_amt, start_kosdaq)
                close = DataUtils.inverse_scaled_data(scaler_close, close_scaled)
                eval_amt = money + close * stock_cnt
                eval_percent = self.get_ratio (eval_amt, start_money)
                eval_index_amt = money_index + close * stock_cnt_index
                eval_index_percent  = self.get_ratio (eval_index_amt, start_money)
                invest_chart_data.append([date, eval_percent, eval_index_percent, kospi_percent, kosdaq_percent])
                invest_daily.append([date, eval_amt, eval_index_amt, kospi_amt, kosdaq_amt])
            except Exception:
                pass
                #exc_type, exc_value, exc_traceback = sys.exc_info()
                #traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)

        #print(invest_chart_data)
        if self.params.debug == True:
            df_data = pd.DataFrame(invest_chart_data, columns=['date', 'invest', 'index', 'KOSPI', 'KOSDAQ'])
            self.draw_invest_seaborn(df_data, dir, corp_name)
            self.save_csv(df_data, dir, corp_name)
        return invest_daily

    def draw_invest_months(self, chart_data:list, start:str, end:str)->None:
        dir = 'invests_total'
        title = "매달 10종목 추천"
        months_chart_data = None
        for i in range(len(chart_data)):
            if months_chart_data is None:
                months_chart_data = chart_data[i]
            else:
                months_chart_data = months_chart_data.append(chart_data[i], ignore_index=True)

        result_data = self.get_invest_chart_data(months_chart_data)
        self.save_csv(result_data, dir, title, start, end)
        self.draw_invest_seaborn(result_data, dir, title, start, end)



    def get_last_multiply(self, value, last):
        return value * (last / 100 + 1)

    def get_ratio(self, eval_amt, start_money):
        return (eval_amt / start_money - 1) * 100

    def draw_invest_seaborn(self, pridects_data:pd.DataFrame, dir:str, title:str, start:str=None, end:str=None)->None:
        file_path = self.get_file_path(dir, title, 'png', start, end)
        pridects_data = pridects_data.copy()
        pridects_data['date'] = pd.to_datetime(pridects_data['date'], format='%Y.%m.%d')
        pridects_data.set_index('date', inplace=True)

        plt = self.main.get_plt()
        fig, ax = plt.subplots(figsize=self.main.FIG_SIZE)
        sns.lineplot(data=pridects_data).set_title(title)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
        ax.set(xlabel='Date', ylabel='Margin(%)')
        #fig.autofmt_xdate()
        plt.grid(color='k', linestyle='dotted', linewidth=1, alpha=0.4)
        fig.savefig(file_path)
        plt.close()


    def save_csv(self, df_rmses, dir, corp_name, start=None, end=None)->None:
        """차트 테이터를 저장한다."""
        file_path = self.get_file_path(dir, corp_name, "csv", start, end)
        DataUtils.save_csv(df_rmses, file_path)


if __name__ == '__main__':
    params = GlobalParams()
    corp = Corp(params)
    kospi = corp.get_kospi()
    kospi = kospi.query("date>='{}' & date<='{}'".format(params.invest_start_date, params.invest_end_date))
    print(kospi[0:1]['close'].values[0])