import pandas as pd
import numpy as np
import os


class DataUtils:
    """ 데이터 처리 관련 메소드"""

    @staticmethod
    def save_excel(df_data, file_path, sheet_name = 'Sheet1'):
        """ excel로 저장한다. """
        DataUtils.create_dir_4path(file_path)
        writer = pd.ExcelWriter(file_path)
        df_data.to_excel(writer, sheet_name, index=False)
        writer.save()

    @staticmethod
    def save_csv(df_data, file_path):
        """ csv로 저장한다. """
        DataUtils.create_dir_4path(file_path)
        df_data.to_csv(file_path, index=False)

    @staticmethod
    def create_dir_4path(file_path):
        """데렉토리를 생성한다."""
        dir = os.path.dirname(file_path)
        DataUtils.create_dir(dir)

    @staticmethod
    def create_dir(dir):
        """데렉토리를 생성한다."""
        if not os.path.exists(dir):
            os.makedirs(dir)

    @staticmethod
    def to_string_corp_code(corp_code):
        """주식회사 코드를 문자열로 바꾼다."""
        return format(int(corp_code), "06d")

    @staticmethod
    def inverse_scaled_data(scaler, scaled_data):
        if type(scaled_data) is np.ndarray:
            scaled_data = scaled_data[0]
        #print(scaler.inverse_transform([[scaled_data]]))
        return scaler.inverse_transform([[scaled_data]])[0][0]


