import pandas as pd
import os
import datetime
import requests
import sys, traceback
import glob

from params.global_params import GlobalParams
from utils.date_utils import DateUtils
from data.data_utils import DataUtils


class Stocks:
    """ 주식데이터  """

    DIR = os.path.dirname(os.path.abspath(__file__))
    DIR_STOCKS = os.path.join(DIR, 'files', 'stocks')

    def __init__(self, params=None):
        self.params = params


    def get_daum_url_before (self, comp_code, start_date=''):
        """ 다음증권 URL """
        if start_date == '':
            return 'http://finance.daum.net/item/quote_yyyymmdd_sub.daum?code={code}&modify=1'.format(code=comp_code)
        else:
            return 'http://finance.daum.net/item/quote_yyyymmdd.daum?code={code}'.format(code=comp_code)

    def get_stock_daum_data_before (self, comp_code, start_date=''):
        """다음증권의 매일 주식정보를 가져온다."""
        url = self.get_daum_url_before(comp_code, start_date)
        df = pd.DataFrame()
        # 다음 웹 크롤링
        page = 1
        while True:
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            page_data = pd.read_html(pg_url, header=0)[0]
            page_data = page_data.dropna()
            if len(page_data) == 0:
                break
            page_data = page_data[['일자', '종가', '시가', '고가', '저가', '거래량']]
            page_data['일자'] = pd.to_datetime(page_data['일자'], format='%y.%m.%d').dt.strftime('%Y.%m.%d')
            last_date = page_data.tail(1)['일자'].to_string(index=False)
            df = df.append(page_data, ignore_index=True)
            if start_date != '':
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    break
            page += 1
        # 필요 없는 날짜 제거
        if start_date != '':
            drop_cnt = 0
            df_len = len(df)
            for i in range(df_len):
                last_date = df.loc[df_len - i - 1, '일자']
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    drop_cnt += 1
                else:
                    break
            if drop_cnt > 0:
                df = df[:-drop_cnt]
        # 정렬 및 컬럼명 변경
        if df.shape[0] != 0:
            df = df.sort_values(by='일자')
            df.rename(columns={'일자': 'date',
                               '종가': 'close',
                               '시가': 'open',
                               '고가': 'high',
                               '저가': 'low',
                               '거래량': 'volume'}, inplace=True)
            return df

    def get_web_data(self, comp_code, url, cols, rename_cols,  start_date='', date_col_name='date', data_type='json', headers = None):
        df = pd.DataFrame()

        # 다음 웹 크롤링
        page = 1
        while True:
            pg_url = url.format(code=comp_code, page=page)
            if data_type == 'json':
                if headers != None:
                    response = requests.get(pg_url, headers=headers)
                else:
                    response = requests.get(pg_url)
                json = response.json()
                page_data = pd.DataFrame.from_dict(json['data'])
            else:
                page_data = pd.read_html(pg_url, header=0)[0]
                page_data = page_data.dropna()

            if len(page_data) == 0:
                break
            page_data = page_data[cols]
            page_data[date_col_name] = page_data[date_col_name].str.slice(0, 10).str.replace("-", ".")
            last_date = page_data.tail(1)[date_col_name].to_string(index=False)
            df = df.append(page_data, ignore_index=True)
            if start_date != '':
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    break
            page += 1

        # 필요 없는 날짜 제거
        if start_date != '':
            drop_cnt = 0
            df_len = len(df)
            for i in range(df_len):
                last_date = df.loc[df_len - i - 1, date_col_name]
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    drop_cnt += 1
                else:
                    break
            if drop_cnt > 0:
                df = df[:-drop_cnt]

        # 정렬 및 컬럼명 변경
        if df.shape[0] != 0:
            df = df.sort_values(by=date_col_name)
            df.rename(columns=rename_cols, inplace=True)
        return df


    def get_stock_daum_data(self, comp_code, start_date=''):
        """다음증권의 매일 주식정보를 가져온다."""
        url = "http://finance.daum.net/api/quote/A{code}/days?symbolCode=A{code}&page={page}&perPage=30&pagination=true"
        cols = ['date','tradePrice', 'openingPrice', 'highPrice', 'lowPrice', 'accTradeVolume']
        rename_cols = {'date': 'date',
                       'tradePrice': 'close',
                       'openingPrice': 'open',
                       'highPrice': 'high',
                       'lowPrice': 'low',
                       'accTradeVolume': 'volume'}
        headers = {
            'Host' : 'finance.daum.net',
            'Referer' : 'http://finance.daum.net/quotes/',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        return self.get_web_data(comp_code, url, cols, rename_cols, start_date, headers = headers)


    def get_investor_daum_data(self, comp_code, start_date=''):
        """다음증권의 매일 주식정보를 가져온다."""
        url = "http://finance.daum.net/api/investor/days?page={page}&perPage=30&symbolCode=A{code}&pagination=true"
        return df


    def get_stock_paxnet_data(self, comp_code, start_date=''):
        """다음증권의 매일 주식정보를 가져온다."""

        url = "http://paxnet.moneta.co.kr/stock/analysis/pagingListAjax?abbrSymbol={code}&currentPageNo={page}&method=listByDate"
        cols = ['tradeDt','closePrice', 'openPrice', 'highPrice', 'lowPrice', 'volume']
        rename_cols = {'tradeDt': 'date',
                       'closePrice': 'close',
                       'openPrice': 'open',
                       'highPrice': 'high',
                       'lowPrice': 'low',
                       'volume': 'volume'}
        return self.get_web_data(comp_code, url, cols, rename_cols, start_date, 'tradeDt')


    def get_stock_naver_data(self, comp_code, start_date):
        """네이버 매일 주식정보를 가져온다."""
        url = "http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
        df = pd.DataFrame()

        # 네이버 웹 크롤링
        page = 1
        bf_date = ''
        while True:
            pg_url = url.format(code=comp_code, page=page)
            page_data = pd.read_html(pg_url, header=0)[0]
            page_data = page_data.dropna()
            page_data = page_data[['날짜', '종가', '시가', '고가', '저가', '거래량']]
            last_date = page_data.tail(1)['날짜'].to_string(index=False)
            if bf_date == last_date:
                break
            df = df.append(page_data, ignore_index=True)
            if start_date != '':
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    break
            if len(page_data) < 10:
                break
            page += 1
            bf_date = last_date

        # 필요 없는 날짜 제거
        if start_date != '':
            drop_cnt = 0
            df_len = len(df)
            for i in range(df_len):
                last_date = df.loc[df_len - i - 1, '날짜']
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    drop_cnt += 1
                else:
                    break
            if drop_cnt > 0:
                df = df[:-drop_cnt]

        # 정렬 및 컬럼명 변경
        if df.shape[0] != 0:
            df = df.sort_values(by='날짜')
            df.rename(columns={'날짜': 'date',
                               '종가': 'close',
                               '시가': 'open',
                               '고가': 'high',
                               '저가': 'low',
                               '거래량': 'volume'}, inplace=True)
        return df

    def get_stock_web_data(self, comp_code, date_next):
        """ 웹에서 주식 데이터를 가져온다."""
        try:
            return self.get_stock_daum_data(comp_code, date_next)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
            try :
                return self.get_stock_naver_data(comp_code, date_next)
            except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
                return self.get_stock_daum_data_before(comp_code, date_next)


    def get_stock_data(self, comp_code:str)->pd.DataFrame:
        comp_code = DataUtils.to_string_corp_code(comp_code)
        file_path = os.path.join(self.DIR_STOCKS, comp_code + '.txt')

        if os.path.isfile(file_path):
            stock_data = pd.read_csv(file_path)
            if hasattr(self.params, 'check_stock_data') and self.params.check_stock_data == True:
                stock_data = stock_data.dropna()
                stock_data = stock_data[:-1]
                date_last = stock_data.tail(1)['date'].to_string(index=False)
                date_next = DateUtils.to_date(date_last) + datetime.timedelta(days=1)
                date_next = date_next.strftime("%Y.%m.%d")
                new_data = self.get_stock_web_data(comp_code, date_next)
                if len(new_data) > 0:
                   stock_data = stock_data.append(new_data, ignore_index=True)
                   stock_data = stock_data.dropna()
                   stock_data.to_csv(file_path, index=False)
        else:
            stock_data = self.get_stock_web_data(comp_code, '')
            stock_data.to_csv(file_path, index=False)

        stock_data = stock_data.dropna()

        if hasattr(self.params, 'forcast_date') and self.params.forcast_date is not None:
            stock_data = stock_data.query("date<'{}'".format(self.params.forcast_date))
        elif hasattr(self.params, 'remove_stock_days') and self.params.remove_stock_days > 0:
            stock_data = stock_data[:-self.params.remove_stock_days]
        return stock_data

    def update_stocks_data(self):
        files = glob.glob(self.DIR_STOCKS + "/*.txt")
        for file_path in files:
            file_name = os.path.basename(file_path)
            stock_data = pd.read_csv(file_path)
            stock_data = stock_data.dropna()
            stock_data = stock_data[:-1]
            date_last = stock_data.tail(1)['date'].to_string(index=False)
            date_next = DateUtils.to_date(date_last) + datetime.timedelta(days=1)
            date_next = date_next.strftime("%Y.%m.%d")
            comp_code = file_name.replace(".txt", "")
            new_data = self.get_stock_web_data(comp_code, date_next)
            if len(new_data) > 0:
               stock_data = stock_data.append(new_data, ignore_index=True)
               stock_data = stock_data.dropna()
               stock_data.to_csv(file_path, index=False)


if __name__ == '__main__':
    params = GlobalParams()
    stocks = Stocks(params)
    #stocks.update_stocks_data()
    df = stocks.get_stock_daum_data('005930')
    #df = stocks.get_stock_data('005930')
    print(len(df))