from datetime import datetime
from dateutil import relativedelta


class DateUtils:

    DATE_FORMAT = '%Y.%m.%d'

    @staticmethod
    def to_date(date_str, date_format=None):
        """문자열을 데이트 형대로 변환한다."""
        date_str = date_str.replace(" ", "")

        if date_format is None:
            split = ""
            if date_str.find(".") > -1:
                split = "."
            elif date_str.find("-") > -1:
                split = "-"
            date_format = '%Y' + split + '%m' + split + '%d'
        try :
            result = datetime.strptime(date_str, date_format)
        except Exception as inst:
            print(date_str)
            print(inst)
            result = datetime.now()
        return result

    @staticmethod
    def to_date_str(date, date_format=None):
        """데이트 문자열  형대로 변환한다."""

        if date_format is None:
            date_format = DateUtils.DATE_FORMAT

        return datetime.strftime(date, date_format)

    @staticmethod
    def between_months(from_month, to_month):
        if type(from_month) == str:
            from_month = datetime.strptime(str(from_month), '%Y.%m')
        if type(to_month) == str:
            to_month = datetime.strptime(str(to_month), '%Y.%m')
        r = relativedelta.relativedelta(to_month, from_month)
        return r.years * 12 + r.months

    @staticmethod
    def add_days(date, add=1, format=None):
        if add == 0:
            return date
        elif format is None:
            return date + relativedelta.relativedelta(days=add)
        else:
            date = DateUtils.to_date(date, format)
            date = date + relativedelta.relativedelta(days=add)
            return DateUtils.to_date_str(date, format)

    @staticmethod
    def add_months(date, add=1):
        return date + relativedelta.relativedelta(months=add)

    @staticmethod
    def add_years(date, add=1):
        return date + relativedelta.relativedelta(years=add)

    @staticmethod
    def to_month_str(month, add=0):
        if add != 0:
            month = DateUtils.add_months(month, add)
        return datetime.strftime(month, '%Y.%m')

    @staticmethod
    def today_str(format=None):
        if format is None:
            format = DateUtils.DATE_FORMAT
        return datetime.today().strftime(format)


if __name__ == '__main__':
    date = DateUtils.to_date('2018.01', '%Y.%m')
    # print(date)
    between = DateUtils.between_months('2017.01', '2018.08')
    print(between)
    # added_date = DateUtils.add_months(date)
    # print(added_date)
    # month_str = DateUtils.to_month_str(added_date)
    # print(month_str)