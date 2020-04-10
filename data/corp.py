import pandas as pd
import numpy as np
import os
import requests
from io import BytesIO
import datetime
from data.data_utils import DataUtils
from params.global_params import GlobalParams
from utils.date_utils import DateUtils


class Corp:
    """ 주식회사 정보  """
    DIR = os.path.dirname(os.path.abspath(__file__))
    CORPS_FILE_PATH = DIR + '/files/corps.xlsx'
    CORPS_FILE_PATH2 = DIR + '/files/corps2.xlsx'
    CORPS_FILE_PATH3 = DIR + '/files/corps3.xlsx'

    URL_CORPS_KRX = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    DAUM_HEADER = {
        'Host': 'finance.daum.net',
        'Referer': 'http://finance.daum.net/quotes/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    KTX_HEADER = {
        #'Host': 'file.krx.co.kr',
        'Referer': 'https://marketdata.krx.co.kr/mdi',
        "User-Agent": "Mozilla/5.0"
        #'Content-Type': 'application/x-www-form-urlencoded',
        #'Origin': 'https://marketdata.krx.co.kr'
    }

    def __init__(self, params=None):
        if params != None:
            self.params = params

    def save_corps(self):
        """ 주식회사 정보를 가져와서 엑셀로 저장한다. """
        code_df = pd.read_html(self.URL_CORPS_KRX, header=0)[0]
        DataUtils.save_excel(code_df, self.CORPS_FILE_PATH)

    def save_corps_csv(self, file_path):
        """ 주식회사 정보를 가져와서 csv로 저장한다. """
        corps = self.get_corps_master()
        corps = corps[['종목코드', '회사명', '상장일']]
        DataUtils.save_csv(corps, file_path)

    def get_corp_code(self, corp_name) -> str:
        """ 엘셀을 불러와서 회사 코드를 가져온다. """
        corps = self.get_corps_all()
        corp_code = corps.query("회사명=='{}'".format(corp_name))['종목코드'].to_string(index=False)
        return format(int(corp_code), "06d")

    def get_corp(self, corp_name):
        """ 엘셀을 불러와서 회사 코드를 가져온다. """
        corps = self.get_corps_all()
        corp = corps.query("회사명=='{}'".format(corp_name))
        return corp

    def get_corp_codes(self, to_listing_date=''):
        corp_codes = self.get_corps(to_listing_date, '종목코드')
        return corp_codes

    def get_corps_for_names(self, corp_nemas):
        """회사명들로 종목을 검색한다."""
        corps = self.get_corps_all()
        values = corps.loc[corps['회사명'].isin(corp_nemas)]
        values.index = range(len(values.index))
        return values

    def get_corps_for_codes(self, corp_codes):
        """회사명들로 종목을 검색한다."""
        corps = self.get_corps()
        values = corps.loc[corps['종목코드'].isin(corp_codes)]
        values.index = range(len(values.index))
        return values

    def get_corp_name(self, corp_code: str or int):
        """회사명들로 종목을 검색한다."""
        if type(corp_code) is str:
            corp_code = corp_code.replace("A", "")
            corp_code = int(corp_code)
        corps = self.get_corps()
        value = corps.loc[corps['종목코드'] == corp_code]
        return value['회사명'].values[0]

    def get_corps_all(self):
        today = DateUtils.today_str('%Y.%m.%d')
        year = today[0:4]
        file_path = os.path.join(self.DIR, 'files', 'corps', year, 'corps_' + today + '.txt')
        if not os.path.isfile(file_path):
            self.save_corps_csv(file_path)
        return pd.read_csv(file_path)

    def get_corps(self, to_listing_date='', columns=None):
        corps = self.get_corps_all()
        if to_listing_date != '':
            corps = corps.query("상장일<='{}'".format(to_listing_date))
        if not (columns is None):
            corps = corps[columns]
        return corps

    def exclude_corps(self, before_result: pd.DataFrame, now_month: str):
        """ 모의 투자시 제외시킬 종목을 제거한다. """
        for exclude_corp in self.params.exclude_corps:
            if now_month == exclude_corp["month"]:
                if before_result['code'].dtype == int:
                    before_result = before_result.query("code!={}".format(int(exclude_corp["code"])))
                else:
                    before_result = before_result.query("code!='{}'".format(exclude_corp["code"]))
        return before_result

    def get_corps2(self):
        return pd.read_csv(self.CORPS_FILE_PATH2)

    def get_corps3(self):
        return pd.read_csv(self.CORPS_FILE_PATH3)

    def get_eval_corps(self):
        data_columns = ['회사명', '종목코드']
        return self.get_corps('1976-05-20', data_columns)

    def get_eval_corps2(self):
        corps = pd.read_csv(self.CORPS_FILE_PATH2)
        corps = corps.query("상장일<'{}'".format('2000-01-01'))
        selected_corps_first = corps[:50]
        selected_corps_last = corps[len(corps) - 60:-10]
        return selected_corps_first.append(selected_corps_last, ignore_index=True)

    def get_eval_corps_auto(self, date_maket_cap=None) -> pd.DataFrame:
        """100개의 주식 종목을 정해진 방법에 의해 가져온다"""

        if hasattr(self.params, 'invest_start_date') == False or self.params.invest_start_date is None:
            invest_start_date_str = DateUtils.today_str('%Y.%m.%d')
        else:
            invest_start_date_str = self.params.invest_start_date
        invest_start_date = DateUtils.to_date(invest_start_date_str)

        if hasattr(self.params, 'max_listing_period_years') == False or self.params.max_listing_period_years is None:
            max_listing_period_years = 20
        else:
            max_listing_period_years = self.params.max_listing_period_years

        max_listing_date = DateUtils.add_years(invest_start_date, -max_listing_period_years)
        max_listing_date = DateUtils.to_date_str(max_listing_date, '%Y-%m-%d')
        corps = self.get_corps_all()
        corps = corps.query("상장일<'{}'".format(max_listing_date))
        corps.loc[:, '종목코드'] = corps['종목코드'].astype(str).str.zfill(6)
        if date_maket_cap is None:
            date_maket_cap = invest_start_date_str
        #corps_cap = self.get_corps_maket_cap(date_maket_cap)
        corps_cap = self.get_now_corps_maket_cap()
        corps = corps.merge(corps_cap, on='종목코드')
        corps = corps.sort_values(by=["시가총액"], ascending=False)

        selected_corps_first = corps[:50]
        selected_corps_last = corps[len(corps) - 60:-10]
        return selected_corps_first.append(selected_corps_last, ignore_index=True)

    def get_corps_master(self):
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
        data = {
            'method': 'download',
            'orderMode': '1',  # 정렬컬럼
            'orderStat': 'D',  # 정렬 내림차순
            'searchType': '13',  # 검색유형: 상장법인
            'fiscalYearEnd': 'all',  # 결산월: 전체
            'location': 'all',  # 지역: 전체
        }

        r = requests.post(url, data=data)
        f = BytesIO(r.content)
        dfs = pd.read_html(f, header=0, parse_dates=['상장일'])
        # df = dfs[0].copy()

        # 숫자를 앞자리가 0인 6자리 문자열로 변환
        # df['종목코드'] = df['종목코드'].astype(np.str).str.zfill(6)
        return dfs[0]

    def get_coprs_master_price_from_krx(self, date=None):
        if date is None:
            date = DateUtils.today_str('%Y%m%d')
        else:
            date = date.replace(".", "").replace("-", "")
        df = None
        for i in range(30):
            date = DateUtils.add_days(date, -i, '%Y%m%d')

            # STEP 01: Generate OTP
            gen_otp_url = 'https://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
            gen_otp_data = {
                'name': 'fileDown',
                'filetype': 'csv',
                'url': 'MKD/04/0404/04040200/mkd04060200_01',
                'market_gubun': 'ALL',  # 시장구분: ALL=전체
                'indx_ind_cd': '',
                'sect_tp_cd': '',
                'schdate': date,
                'pagePath': '/contents/MKD/04/0404/04040200/MKD04040200.jsp',
            }

            r = requests.post(gen_otp_url, gen_otp_data, headers=self.KTX_HEADER)
            code = r.content  # 리턴받은 값을 아래 요청의 입력으로 사용.

            #STEP 02: download
            down_url = 'http://file.krx.co.kr/download.jspx'
            down_data = {
                'code': code
            }
            r = requests.post(down_url, down_data, headers=self.KTX_HEADER)
            df = pd.read_csv(BytesIO(r.content), header=0, thousands=',')
            if len(df) > 0:
                break

        return df

    def get_now_coprs_master_price_from_krx(self):
        # STEP 01: Generate OTP
        gen_otp_url = 'https://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
        gen_otp_data = {
            'name': 'fileDown',
            'filetype': 'xls',
            'url': 'MKD/04/0406/04060100/mkd04060100_01',
            'market_gubun': 'ALL',  # 시장구분: ALL=전체
            'pagePath': '/contents/MKD/04/0406/04060100/MKD04060100.jsp',
            'sort_type': 'A',
            'lst_stk_vl': 1,
            'cpt': 1,
            'isu_cdnm': '전체'
        }

        r = requests.post(gen_otp_url, gen_otp_data, headers=self.KTX_HEADER)
        code = r.content  # 리턴받은 값을 아래 요청의 입력으로 사용.

        #STEP 02: download
        down_url = 'http://file.krx.co.kr/download.jspx'
        down_data = {
            'code': code
        }
        r = requests.post(down_url, down_data, headers=self.KTX_HEADER)
        df = pd.read_excel(BytesIO(r.content), header=0, thousands=',')
        return df

    def get_corps_maket_cap(self, date=None):
        if date == None:
            date = DateUtils.today_str()  # 오늘 날짜
        else:
            date = date.replace(".", "").replace("-", "")
        for i in range(30):
            try:
                year = date[0:4]
                file_path = os.path.join(self.DIR, "files", "corps_market_cap", year, "market_cap_" + date + ".txt")
                if not os.path.isfile(file_path):
                    master_data = self.get_coprs_master_price_from_krx(date)
                    market_cap = master_data[['종목코드', '시가총액']]
                    DataUtils.save_csv(market_cap, file_path)
                else:
                    market_cap = pd.read_csv(file_path)
                break
            except:
                print(date, "에 시가총액 데이터를 가져오는데 에러 발생하여 이전 데이터 사용")
                date = DateUtils.add_days(date, -i, '%Y%m%d')
        return market_cap

    def get_now_corps_maket_cap(self):
        date = DateUtils.today_str('%Y%m%d')  # 오늘 날짜
        year = date[0:4]
        file_path = os.path.join(self.DIR, "files", "corps_market_cap", year, "market_cap_" + date + ".txt")
        if not os.path.isfile(file_path):
            master_data = self.get_now_coprs_master_price_from_krx()
            market_cap = master_data[['종목코드', '자본금(원)']]
            market_cap.rename(columns={'자본금(원)': '시가총액'}, inplace=True)
            DataUtils.save_csv(market_cap, file_path)
        else:
            market_cap = pd.read_csv(file_path)
        market_cap.loc[:, '종목코드'] = market_cap['종목코드'].astype(str).str.zfill(6)
        return market_cap

    def get_kospi(self):
        return self.get_kospi_kosdaq('KOSPI')

    def get_kosdaq(self):
        return self.get_kospi_kosdaq('KOSDAQ')

    def get_kospi_kosdaq(self, market='KOSPI'):
        file_path = os.path.join(self.DIR, 'files', market + '.txt')

        if os.path.isfile(file_path):
            kos_data = pd.read_csv(file_path)
            if hasattr(self.params, 'check_kos_data') and self.params.check_kos_data == True:
                kos_data = kos_data.dropna()
                kos_data = kos_data[:-1]
                date_last = kos_data.tail(1)['date'].to_string(index=False)
                date_next = DateUtils.to_date(date_last) + datetime.timedelta(days=1)
                date_next = date_next.strftime("%Y.%m.%d")
                new_data = self.get_kospi_kosdaq_from_daum(market, date_next)
                if len(new_data) > 0:
                    kos_data = kos_data.append(new_data, ignore_index=True)
                    kos_data = kos_data.dropna()
                    kos_data.to_csv(file_path, index=False)
        else:
            kos_data = self.get_kospi_kosdaq_from_daum(market, '')
            kos_data.to_csv(file_path, index=False)

        kos_data = kos_data.dropna()
        return kos_data

    def get_kospi_kosdaq_from_daum(self, market='KOSPI', start_date=''):
        daum_url = 'http://finance.daum.net/api/market_index/days?page={page}&perPage=10&market={market}&pagination=true'

        df = pd.DataFrame()
        # 다음 웹 크롤링
        page = 1
        while True:
            pg_url = daum_url.format(market=market, page=page)
            response = requests.get(pg_url, headers=self.DAUM_HEADER)
            json = response.json()
            page_data = pd.DataFrame.from_dict(json['data'])
            if len(page_data) == 0:
                break
            page_data = page_data[['date', 'tradePrice']]
            page_data['date'] = page_data['date'].str.slice(0, 10).str.replace("-", ".")
            last_date = page_data.tail(1)['date'].to_string(index=False)
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
                last_date = df.loc[df_len - i - 1, 'date']
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    drop_cnt += 1
                else:
                    break
            if drop_cnt > 0:
                df = df[:-drop_cnt]

        # 정렬 및 컬럼명 변경
        if df.shape[0] != 0:
            df = df.sort_values(by='date')
            df.rename(columns={'date': 'date',
                               'tradePrice': 'close'}, inplace=True)
        return df


if __name__ == '__main__':
    params = GlobalParams()
    corp = Corp(params)
    # corps = corp.get_corps_for_codes([9200,40300])
    # corps = corp.get_corps_for_names(['카카오'])
    # print(corps)
    # df = corp.corps_maket_cap()

    df = corp.get_now_coprs_master_price_from_krx()
    # df = corp.get_eval_corps_auto()
    # df = corp.get_corps_all()
    # df = corp.get_kospi()
    print(df[['종목코드', '자본금(원)']].head())
    #value = corp.get_corp_name("A206400")

